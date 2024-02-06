from transformers import AutoTokenizer, AutoConfig
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
from transformers import MistralForCausalLM
from ebany_research.llm_lora.changed_mistral import (
    LinearLora,
    ChangedMistralForCausalLM,
)


def random_seed(seed=42, rank=0):
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)


class OpenOrcaDataset(Dataset):
    def __init__(
        self,
        dataset=None,
        tokenizer=None,
    ):
        self.dataset = dataset
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        dataset_item = self.dataset[idx]
        chat = [
            {"role": "system", "content": dataset_item["system_prompt"]},
            {"role": "user", "content": dataset_item["question"]},
            {"role": "assistant", "content": dataset_item["response"]},
        ]
        inputs = self.tokenizer.apply_chat_template(
            chat,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer(
            inputs,
            return_tensors="pt",
            truncation=True,
            # max_length=4096,
            max_length=2048,
        )
        for key in inputs.keys():
            inputs[key] = inputs[key].squeeze(0)
        # print(inputs['input_ids'].shape)
        return inputs


# train_dataset[0]


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_L_R(weights=None, rank=64):
    U, S, Vh = torch.linalg.svd(
        weights.to(torch.float32),
        full_matrices=False,
    )

    U = U[:, :rank]
    S = torch.diag(S[:rank])
    Vh = Vh[:rank, :]

    L = U @ S
    R = Vh
    return L, R


def assign_new_weights(original_module, original_weights):
    L, R = get_L_R(original_weights, rank=16)
    original_module.L.weight.data = L
    original_module.R.weight.data = R


def freeze_params(model, layers=None):
    for param in model.named_parameters():
        if "L" in param[0] or "R" in param[0]:
            param[1].requires_grad_(False)
            for layer_id in layers:
                if str(layer_id) == param[0].split(".")[2]:
                    print(param[0])
                    param[1].requires_grad_(True)
        else:
            param[1].requires_grad_(False)


def eval_model(model):
    total_eval_loss = 0
    with torch.no_grad():
        with torch.autocast(device_type="cuda"):
            for eval_batch in tqdm.tqdm(valid_dataloader):
                # print(eval_batch)

                for key in eval_batch.keys():
                    eval_batch[key] = eval_batch[key].to(model.device)

                loss = model(
                    **eval_batch,
                )
                total_eval_loss += loss.loss.item()
    # break
    total_eval_loss = total_eval_loss / len(valid_dataloader)
    return total_eval_loss


if __name__ == "__main__":
    random_seed()
    model_name = "Open-Orca/Mistral-7B-OpenOrca"
    lora_model_name = "ebany_research/llm_lora/models/"
    lora_model_name += "openorca_lora_[17][11_17_22_26][11c_17_22_26c][11_17c_22_26][6_11_14_17_22_26][6c_11_14c_17_22_26][6_11c_14_17c_22c_26][6_11_14_17_20_22_25_26][6c_11c_14_17c_20c_22c_25c_26]"
    config = AutoConfig.from_pretrained(lora_model_name)
    device = 0
    teacher_model = MistralForCausalLM.from_pretrained(
        model_name,
        device_map={"": device},
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    teacher_model = teacher_model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    dataset = load_dataset("dim/openaccess-ai-collective-oo-gpt4-filtered")

    train_elements = 1000
    valid_elements = 1000
    batch_size = 2

    valid_dataset = OpenOrcaDataset(
        dataset=dataset["test"].to_list()[:valid_elements],
        tokenizer=tokenizer,
    )

    pad_datacollator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    valid_dataloader = DataLoader(
        dataset=valid_dataset,
        batch_size=batch_size,
        collate_fn=pad_datacollator,
        shuffle=False,
    )

    # test
    next(iter(valid_dataloader))
    # 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 [17] 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32
    # 1 2 3 4 5 [6] 7 8 9 10 [11] 12 13 [14] 15 16 [17] 18 19 [20] 21 [22] 23 24 [25] [26] 27 28 29 30 31 32
    distill_layers = [
        30,
        10,
    ]
    config.lora_layers = config.lora_layers + distill_layers

    device = 1
    student_model = ChangedMistralForCausalLM.from_pretrained(
        lora_model_name,
        device_map={"": device},
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        config=config,
    )
    student_model = student_model.eval()
    print(student_model)
    for distill_layer in distill_layers:
        mlp = teacher_model.model.layers[distill_layer].mlp

        student_gate_proj = mlp.gate_proj.weight.data.clone()
        student_up_proj = mlp.up_proj.weight.data.clone()
        student_down_proj = mlp.down_proj.weight.data.clone()

        # test
        # L, R = get_L_R(student_down_proj)
        # print(L.shape, R.shape)

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

        student_model.model.layers[distill_layer].mlp.gate_proj = lora_gate_proj
        student_model.model.layers[distill_layer].mlp.up_proj = lora_up_proj
        student_model.model.layers[distill_layer].mlp.down_proj = lora_down_proj
        student_model.to(f"cuda:{device}")

    # show trainable parameters
    print(count_parameters(teacher_model))
    print(count_parameters(student_model))

    # teacher_loss = eval_model(teacher_model)
    # print("teacher_loss", teacher_loss, torch.exp(torch.tensor(teacher_loss)))
    student_loss = eval_model(student_model)
    print("student_loss", student_loss, torch.exp(torch.tensor(student_loss)))

    name = lora_model_name.split("[")[-1][:-1]
    name = name.split("_")
    name = [int(item.replace("c", "")) for item in name]
    name.extend(distill_layers)
    name = sorted(name)
    config.lora_layers = name
    name = [str(item) for item in name]
    name = "_".join(name)
    lora_model_name += f"[{name}]"
    print(lora_model_name)
    student_model.save_pretrained(lora_model_name)
