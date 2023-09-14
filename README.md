# KGC-LLM

This repository contains codes and access to dataset used for our accepted [LLM-KBC workshop](https://lm-kbc.github.io/workshop2023/) paper titled Do **Instruction-tuned Large Language Models Help
with Relation Extraction?**

The REBEL dataset we used for fine-tuning Dolly-v2-3b can be found here:
https://zenodo.org/record/6139236#.YhJdiJPMJhH

In the `sample_data` folder, you can find the 100 samples we used for manual evaluation.

To convert a relation extraction dataset to an instruction-tuning dataset, run the following command.

```
python data_transform.py
```

For fine-tuning Dolly with LoRA, run the following code in a command line with providing your own transformed data path in the script.

```
python fine-tuning.py
```

Once the adapter is trained, you can run inference using the following command.

```
python inference.py
```

If you do have further questions about running the code, feel free to raise an issue or drop me an email at: x DOT li3 AT uva DOT nl .

