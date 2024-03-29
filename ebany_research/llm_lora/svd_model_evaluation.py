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
    random_seed()
    model_name = "Open-Orca/Mistral-7B-OpenOrca"
    lora_model_name = "ebany_research/llm_lora/models/"
    lora_model_name += "52[6c_11c_14_17c_20c_22c_25c_26][6_10_11_14_17_20_22_25_26_30][6c_10c_11c_14c_17c_20_22c_25c_26c_30c]"

    config = AutoConfig.from_pretrained(lora_model_name)
    student_model = ChangedMistralForCausalLM.from_pretrained(
        lora_model_name,
        device_map={"": 0},
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
    )
    student_model = student_model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    dataset = load_dataset("dim/openaccess-ai-collective-oo-gpt4-filtered")
    # dataset = dataset["train"].to_list()

    train_elements = 10000
    valid_elements = 10000
    batch_size = 2

    train_dataset = OpenOrcaDataset(
        dataset=dataset["train"].to_list()[:train_elements],
        tokenizer=tokenizer,
    )

    valid_dataset = OpenOrcaDataset(
        dataset=dataset["test"].to_list()[:valid_elements],
        tokenizer=tokenizer,
    )

    pad_datacollator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    print(count_parameters(student_model))

    save_path = "ebany_research/llm_lora/models/openorca_lora_eval"

    max_steps = 20
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
    result = trainer.evaluate()
    print(
        result["eval_loss"],
        torch.exp(torch.tensor(result["eval_loss"])),
    )
