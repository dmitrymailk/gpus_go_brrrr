from transformers import AutoTokenizer, AutoConfig
from datasets import load_dataset
import torch

from torch.utils.data import Dataset, DataLoader
from transformers import DataCollatorWithPadding, DataCollatorForLanguageModeling
import numpy as np
import random
import tqdm
from transformers import Trainer, TrainingArguments

from ebany_research.llm_lora.original_svd import (
    freeze_params,
    count_parameters,
    random_seed,
    OpenOrcaDataset,
    assign_new_weights,
    get_L_R,
    pad_datacollator,
)
from ebany_research.llm_lora.changed_mistral import (
    LinearLora,
    ChangedMistralForCausalLM,
)
from functools import partial

if __name__ == "__main__":
    seed = random.randint(0, 2**31 - 1)
    random_seed(seed=seed)

    model_name = "Open-Orca/Mistral-7B-OpenOrca"
    lora_model_name = "ebany_research/llm_lora/models/"
    lora_model_name += "openorca_lora_[17]"

    config = AutoConfig.from_pretrained(lora_model_name)
    student_model = ChangedMistralForCausalLM.from_pretrained(
        lora_model_name,
        device_map={"": 0},
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
    )
    print(count_parameters(student_model))
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    dataset = load_dataset("dim/openaccess-ai-collective-oo-gpt4-filtered")
    # dataset = dataset["train"].to_list()

    valid_elements = 1000
    batch_size = 2

    train_dataset = OpenOrcaDataset(
        dataset=dataset["train"].to_list(),
        tokenizer=tokenizer,
    )

    valid_dataset = OpenOrcaDataset(
        dataset=dataset["test"].to_list()[:valid_elements],
        tokenizer=tokenizer,
    )

    callibration_layers = [
        17,
    ]

    freeze_params(
        student_model,
        layers=callibration_layers,
    )
    print(count_parameters(student_model))

    save_path = lora_model_name
    save_path += "["
    layers_names = []
    for layer_id in sorted(config.lora_layers):
        if layer_id in callibration_layers:
            layers_names.append(f"{layer_id}c")
        else:
            layers_names.append(f"{layer_id}")
    layers_names = "_".join(layers_names)
    save_path += layers_names
    save_path += "]"
    print(save_path)

    max_steps = 100
    training_args = TrainingArguments(
        output_dir=save_path,
        evaluation_strategy="steps",
        save_strategy="no",
        num_train_epochs=1,
        max_steps=max_steps,
        report_to="none",
        # report_to="wandb",
        logging_steps=max_steps,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=16,
        bf16=True,
        metric_for_best_model="eval_loss",
    )

    trainer = Trainer(
        model=student_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=partial(
            pad_datacollator,
            tokenizer=tokenizer,
        ),
    )

    # print(trainer.evaluate())
    result = trainer.train()
    # trainer.save_model()
