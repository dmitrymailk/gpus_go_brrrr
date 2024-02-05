from transformers import AutoTokenizer, AutoConfig
from ebany_research.llm_lora.changed_mistral import MistralForCausalLM
from datasets import load_dataset
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DataCollatorWithPadding, DataCollatorForLanguageModeling

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
    LinearLora,
)

if __name__ == "__main__":
    random_seed()
    model_name = "Open-Orca/Mistral-7B-OpenOrca"
    config = AutoConfig.from_pretrained(model_name)
    student_model = MistralForCausalLM.from_pretrained(
        model_name,
        device_map={"": 0},
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    dataset = load_dataset("openaccess-ai-collective/oo-gpt4-filtered")
    dataset = dataset["train"].to_list()

    train_elements = 10000
    valid_elements = 10000
    batch_size = 2

    train_dataset = OpenOrcaDataset(
        dataset=dataset[:train_elements],
        tokenizer=tokenizer,
    )

    valid_dataset = OpenOrcaDataset(
        dataset=dataset[train_elements : train_elements + valid_elements],
        tokenizer=tokenizer,
    )

    pad_datacollator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    distill_layer = 17
    student_mlp = student_model.model.layers[distill_layer].mlp

    student_gate_proj = student_mlp.gate_proj.weight.data
    student_up_proj = student_mlp.up_proj.weight.data
    student_down_proj = student_mlp.down_proj.weight.data

    # test
    L, R = get_L_R(student_down_proj, rank=16)
    print(L.shape, R.shape)

    # replace weights
    lora_gate_proj = LinearLora(
        in_dim=config.hidden_size,
        out_dim=config.intermediate_size,
        bias=False,
    )
    lora_up_proj = LinearLora(
        in_dim=config.hidden_size,
        out_dim=config.intermediate_size,
        bias=False,
    )
    lora_down_proj = LinearLora(
        in_dim=config.intermediate_size,
        out_dim=config.hidden_size,
        bias=False,
    )

    assign_new_weights(
        original_module=lora_gate_proj,
        original_weights=student_gate_proj,
    )
    assign_new_weights(
        original_module=lora_up_proj,
        original_weights=student_up_proj,
    )
    assign_new_weights(
        original_module=lora_down_proj,
        original_weights=student_down_proj,
    )

    student_mlp.gate_proj = lora_gate_proj
    student_mlp.up_proj = lora_up_proj
    student_mlp.down_proj = lora_down_proj

    freeze_params(student_model)
    print(count_parameters(student_model))

    training_args = TrainingArguments(
        output_dir="ebany_research/llm_lora/train_results",
        evaluation_strategy="steps",
        save_strategy="steps",
        num_train_epochs=1,
        save_total_limit=1,
        report_to="none",
        # report_to="wandb",
        logging_steps=2,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=2,
        bf16=True,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        # gradient_checkpointing=True,
    )

    trainer = Trainer(
        model=student_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=pad_datacollator,
    )
    # print(trainer.evaluate())
    trainer.train()
