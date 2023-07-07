import logging
import re
import os
import fire

import numpy as np
import torch
import torch.nn as nn
import bitsandbytes as bnb
from datasets import load_dataset
import transformers

from peft import prepare_model_for_int8_training, LoraConfig, get_peft_model, set_peft_model_state_dict
from transformers import Pipeline, PreTrainedTokenizer, AutoModelForCausalLM, AutoTokenizer


os.environ['CUDA_VISIBLE_DEVICES'] = '0'
logger = logging.getLogger(__name__)

INSTRUCTION_KEY = "### Instruction:"
RESPONSE_KEY = "### Response:"
END_KEY = "### End"
INTRO_BLURB = (
    "Below is an instruction that describes a task. Write a response that appropriately completes the request."
)

# This is the prompt that is used for generating responses using an already trained model.  It ends with the response
# key, where the job of the model is to provide the completion that follows it (i.e. the response itself).
PROMPT_FOR_GENERATION_FORMAT = """{intro}
{instruction_key}
{instruction}
{response_key}
""".format(
    intro=INTRO_BLURB,
    instruction_key=INSTRUCTION_KEY,
    instruction="{instruction}",
    response_key=RESPONSE_KEY,
)


def get_special_token_id(tokenizer: PreTrainedTokenizer, key: str) -> int:
    """Gets the token ID for a given string that has been added to the tokenizer as a special token.
    When training, we configure the tokenizer so that the sequences like "### Instruction:" and "### End" are
    treated specially and converted to a single, new token.  This retrieves the token ID each of these keys map to.
    Args:
        tokenizer (PreTrainedTokenizer): the tokenizer
        key (str): the key to convert to a single token
    Raises:
        RuntimeError: if more than one ID was generated
    Returns:
        int: the token ID for the given key
    """
    token_ids = tokenizer.encode(key)
    if len(token_ids) > 1:
        raise ValueError(f"Expected only a single token for '{key}' but found {token_ids}")
    return token_ids[0]


class InstructionTextGenerationPipeline(Pipeline):
    def __init__(
        self, *args, do_sample: bool = True, max_new_tokens: int = 256, top_p: float = 0.92, top_k: int = 0, **kwargs
    ):
        super().__init__(*args, do_sample=do_sample, max_new_tokens=max_new_tokens, top_p=top_p, top_k=top_k, **kwargs)

    def _sanitize_parameters(self, return_instruction_text=False, **generate_kwargs):
        preprocess_params = {}

        # newer versions of the tokenizer configure the response key as a special token.  newer versions still may
        # append a newline to yield a single token.  find whatever token is configured for the response key.
        tokenizer_response_key = next(
            (token for token in self.tokenizer.additional_special_tokens if token.startswith(RESPONSE_KEY)), None
        )

        response_key_token_id = None
        end_key_token_id = None
        if tokenizer_response_key:
            try:
                response_key_token_id = get_special_token_id(self.tokenizer, tokenizer_response_key)
                end_key_token_id = get_special_token_id(self.tokenizer, END_KEY)

                # Ensure generation stops once it generates "### End"
                generate_kwargs["eos_token_id"] = end_key_token_id
            except ValueError:
                pass

        forward_params = generate_kwargs
        postprocess_params = {
            "response_key_token_id": response_key_token_id,
            "end_key_token_id": end_key_token_id,
            "return_instruction_text": return_instruction_text,
        }

        return preprocess_params, forward_params, postprocess_params

    def preprocess(self, instruction_text, **generate_kwargs):
        prompt_text = PROMPT_FOR_GENERATION_FORMAT.format(instruction=instruction_text)
        inputs = self.tokenizer(
            prompt_text,
            return_tensors="pt",
        )
        inputs["prompt_text"] = prompt_text
        inputs["instruction_text"] = instruction_text
        return inputs

    def _forward(self, model_inputs, **generate_kwargs):
        input_ids = model_inputs["input_ids"]
        attention_mask = model_inputs.get("attention_mask", None)
        generated_sequence = self.model.generate(
            input_ids=input_ids.to(self.model.device),
            attention_mask=attention_mask,
            pad_token_id=self.tokenizer.pad_token_id,
            **generate_kwargs,
        )[0].cpu()
        instruction_text = model_inputs.pop("instruction_text")
        return {"generated_sequence": generated_sequence, "input_ids": input_ids, "instruction_text": instruction_text}

    def postprocess(self, model_outputs, response_key_token_id, end_key_token_id, return_instruction_text):
        sequence = model_outputs["generated_sequence"]
        instruction_text = model_outputs["instruction_text"]

        # The response will be set to this variable if we can identify it.
        decoded = None

        # If we have token IDs for the response and end, then we can find the tokens and only decode between them.
        if response_key_token_id and end_key_token_id:
            # Find where "### Response:" is first found in the generated tokens.  Considering this is part of the
            # prompt, we should definitely find it.  We will return the tokens found after this token.
            response_pos = None
            response_positions = np.where(sequence == response_key_token_id)[0]
            if len(response_positions) == 0:
                logger.warn(f"Could not find response key {response_key_token_id} in: {sequence}")
            else:
                response_pos = response_positions[0]

            if response_pos:
                # Next find where "### End" is located.  The model has been trained to end its responses with this
                # sequence (or actually, the token ID it maps to, since it is a special token).  We may not find
                # this token, as the response could be truncated.  If we don't find it then just return everything
                # to the end.  Note that even though we set eos_token_id, we still see the this token at the end.
                end_pos = None
                end_positions = np.where(sequence == end_key_token_id)[0]
                if len(end_positions) > 0:
                    end_pos = end_positions[0]

                decoded = self.tokenizer.decode(sequence[response_pos + 1 : end_pos]).strip()
        else:
            # Otherwise we'll decode everything and use a regex to find the response and end.

            fully_decoded = self.tokenizer.decode(sequence)

            # The response appears after "### Response:".  The model has been trained to append "### End" at the
            # end.
            m = re.search(r"#+\s*Response:\s*(.+?)#+\s*End", fully_decoded, flags=re.DOTALL)

            if m:
                decoded = m.group(1).strip()
            else:
                # The model might not generate the "### End" sequence before reaching the max tokens.  In this case,
                # return everything after "### Response:".
                m = re.search(r"#+\s*Response:\s*(.+)", fully_decoded, flags=re.DOTALL)
                if m:
                    decoded = m.group(1).strip()
                else:
                    logger.warn(f"Failed to find response in:\n{fully_decoded}")

        if return_instruction_text:
            return {"instruction_text": instruction_text, "generated_text": decoded}

        return decoded


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


def train(
        model_name="databricks/dolly-v2-3b",
        save_model_name="lora-dolly-rebel-ep1/rebel-lora-dolly-2.0-ep1",
        output_dir="lora-dolly-rebel-ep1",
        data_path="rebel/instruction/en_train.json",
        # Finetuning
        # Settings for A100 - For 3090
        MICRO_BATCH_SIZE = 4,  # change to 4 for 3090
        BATCH_SIZE = 128,
        EPOCHS = 1,  # paper uses 3
        LEARNING_RATE = 2e-5,
        CUTOFF_LEN = 256,
        LORA_R = 4,
        LORA_ALPHA = 16,
        LORA_DROPOUT = 0.05,
        # wandb params
        wandb_project="dolly-3b-rebel-ep1",
        wandb_run_name="dolly-3b-rebel-ep1",
        wandb_watch="gradients",  # options: false | gradients | all
        wandb_log_model="false", # options: false | true
        resume_from_checkpoint=None,  # either training checkpoint or final adapter
        prompt_template_name="re",
):
    GRADIENT_ACCUMULATION_STEPS = BATCH_SIZE // MICRO_BATCH_SIZE
    # Check if parameter passed or if set within environ
    use_wandb = len(wandb_project) > 0 or (
            "WANDB_PROJECT" in os.environ and len(os.environ["WANDB_PROJECT"]) > 0
    )
    # Only overwrite environ if wandb param passed
    if len(wandb_project) > 0:
        os.environ["WANDB_PROJECT"] = wandb_project
    if len(wandb_watch) > 0:
        os.environ["WANDB_WATCH"] = wandb_watch
    if len(wandb_log_model) > 0:
        os.environ["WANDB_LOG_MODEL"] = wandb_log_model

    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")

    model = AutoModelForCausalLM.from_pretrained(model_name,
                                                 device_map="auto",
                                                 torch_dtype=torch.bfloat16)


    model = prepare_model_for_int8_training(model, use_gradient_checkpointing=True)

    config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)
    tokenizer.pad_token_id = 0  # unk. we want this to be different from the eos token

    # load train data
    data = load_dataset("json", data_files=data_path)

    data = data.shuffle().map(
        lambda data_point: tokenizer(
            generate_prompt(data_point),
            truncation=True,
            max_length=CUTOFF_LEN,
            padding="max_length",
        )
    )

    if resume_from_checkpoint:
        # Check the available weights and load them
        checkpoint_name = os.path.join(
            resume_from_checkpoint, "pytorch_model.bin"
        )  # Full checkpoint
        if not os.path.exists(checkpoint_name):
            checkpoint_name = os.path.join(
                resume_from_checkpoint, "adapter_model.bin"
            )  # only LoRA model - LoRA config above has to fit
            resume_from_checkpoint = (
                False  # So the trainer won't try loading its state
            )
        # The two files above have a different name depending on how they were saved, but are actually the same.
        if os.path.exists(checkpoint_name):
            print(f"Restarting from {checkpoint_name}")
            adapters_weights = torch.load(checkpoint_name)
            model = set_peft_model_state_dict(model, adapters_weights)
        else:
            print(f"Checkpoint {checkpoint_name} not found")

    model.print_trainable_parameters()  # Be more transparent about the % of trainable params.

    trainer = transformers.Trainer(
        model=model,
        train_dataset=data["train"],
        args=transformers.TrainingArguments(
            per_device_train_batch_size=MICRO_BATCH_SIZE,
            gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
            warmup_steps=100,
            num_train_epochs=EPOCHS,
            learning_rate=LEARNING_RATE,
            fp16=True,
            logging_steps=1,
            output_dir=output_dir,
            save_total_limit=3,
            optim="adamw_torch",
            save_strategy="steps",
            save_steps=200,
            report_to="wandb" if use_wandb else None,
            run_name=wandb_run_name if use_wandb else None,
        ),
        data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    model.config.use_cache = False
    trainer.train(resume_from_checkpoint=False)

    model.save_pretrained(save_model_name)


if __name__ == '__main__':
    fire.Fire(train)




