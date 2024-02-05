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
            max_length=4096,
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


class LinearLora(torch.nn.Module):
    def __init__(self, in_dim=768, out_dim=768, r=16, bias=False):
        super().__init__()
        self.L = torch.nn.Linear(in_dim, r, bias=bias)
        self.R = torch.nn.Linear(r, out_dim, bias=bias)

    def forward(self, hidden_states):
        hidden_states = self.R(hidden_states)
        hidden_states = self.L(hidden_states)
        return hidden_states


def assign_new_weights(original_module, original_weights):
    L, R = get_L_R(original_weights, rank=16)
    original_module.L.weight.data = L
    original_module.R.weight.data = R


def freeze_params(model):
    for param in model.named_parameters():
        if "L" in param[0] or "R" in param[0]:
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
    config = AutoConfig.from_pretrained(model_name)
    device = 0
    model = MistralForCausalLM.from_pretrained(
        model_name,
        device_map={"": device},
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    model = model.eval()
    print(count_parameters(model))

    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    dataset = load_dataset("openaccess-ai-collective/oo-gpt4-filtered")
    dataset = dataset["train"].to_list()

    train_elements = 100000
    valid_elements = 100000
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

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        collate_fn=pad_datacollator,
    )
    valid_dataloader = DataLoader(
        dataset=valid_dataset,
        batch_size=batch_size,
        collate_fn=pad_datacollator,
    )

    # test
    next(iter(train_dataloader))
    next(iter(valid_dataloader))

    device = 1
    student_model = MistralForCausalLM.from_pretrained(
        model_name,
        device_map={"": device},
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
    )
    student_model = student_model.eval()

    distill_layer = 17
    student_mlp = student_model.model.layers[distill_layer].mlp

    student_gate_proj = student_mlp.gate_proj.weight.data
    student_up_proj = student_mlp.up_proj.weight.data
    student_down_proj = student_mlp.down_proj.weight.data

    # test
    L, R = get_L_R(student_down_proj)
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

    # show trainable parameters
    print(count_parameters(model))
    print(count_parameters(student_model))

    teacher_loss = eval_model(model)
    print("teacher_loss", teacher_loss, torch.exp(torch.tensor(teacher_loss)))
    student_loss = eval_model(student_model)
    print("student_loss", student_loss, torch.exp(torch.tensor(student_loss)))
    # 1000
    # teacher_loss 2.297937617301941 tensor(9.9536) batch = 2
    # student_loss 2.2953803930282595 tensor(9.9282) batch = 2
    # 10000
    # teacher_loss 2.3349632108330725 tensor(10.3291) batch = 2
    # student_loss 2.3332279333114623 tensor(10.3112) batch = 2
    # 100000
    # teacher_loss 2.3350424207425116 tensor(10.3299) batch = 2
    # student_loss 2.3333492525792123 tensor(10.3124) batch = 2
