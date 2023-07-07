import os
import json
import torch
from utils.prompter import Prompter
from collections import defaultdict
from peft import PeftModel

from datasets import load_dataset
from transformers import Pipeline, PreTrainedTokenizer, AutoModelForCausalLM, AutoTokenizer, AutoModel,AutoConfig, GPTJForCausalLM
from transformers import GenerationConfig

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['NVIDIA_VISIBLE_DEVICES']='all'
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


def generate_prompt(data_point):
    # taken from https://github.com/tloen/alpaca-lora
    if data_point["instruction"]:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
                ### Instruction:
                {data_point["instruction"]}                
                ### Input:
                {data_point["input"]}                
                ### Response:
                {data_point["output"]}"""
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.
                ### Instruction:
                {data_point["instruction"]}
                ### Response:
                {data_point["output"]}"""


def evaluate(
        instruction,
        model,
        prompter,
        input=None,
        temperature=0.1,
        top_p=0.75,
        top_k=40,
        num_beams=4,
        max_new_tokens=128,
        **kwargs,
):
    prompt = prompter.generate_prompt(instruction, input)
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    generation_config = GenerationConfig(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        num_beams=num_beams,
        **kwargs,
    )
    with torch.no_grad():
        generation_output = model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=max_new_tokens,
        )
    s = generation_output.sequences[0]
    output = tokenizer.decode(s)
    # print(output)
    return prompter.get_response(output)


if __name__ == '__main__':
    CUTOFF_LEN = 256
    base_model = "databricks/dolly-v2-3b"
    trained_model = "alpaca-lora-dolly-2.0"
    output_dir = "dolly-ft-rebel-output/"
    # output_dir = "./"
    output_file = "generated_response.json"
    data_path = "rebel/instruction/en_val.json"

    template_name = "alpaca"
    prompter = Prompter(template_name)

    tokenizer = AutoTokenizer.from_pretrained(base_model, padding_side="left")
    # load val data
    data = load_dataset("json", data_files=data_path)
    # For debugging only, Slice the dataset
    data = data['train'].select(range(0, 10))
    # data = data['train']
    data = data.map(
        lambda data_point: tokenizer(
            generate_prompt(data_point),
            truncation=True,
            max_length=CUTOFF_LEN,
            padding="max_length",
        )
    )

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(
        model,
        trained_model,
        torch_dtype=torch.float16,
    )

    model.eval()

    print("running for validations")
    res_sequences = []

    for sample in data:
        # print(sample)
        result = {}
        instruction = sample['instruction']
        input = sample['input']
        response = evaluate(instruction=instruction, model=model, input=input, prompter=prompter)
        # print(response)
        result["instruction"] = instruction
        result["input"] = input
        result["label"] = sample['output']
        result["response"] = response

        # print("Instruction: ", instruction)
        # print("Input: ", input)
        # print("Response: ", response)
        res_sequences.append(result)

    with open(output_dir + output_file, 'w') as f:
        json.dump(res_sequences, f)
