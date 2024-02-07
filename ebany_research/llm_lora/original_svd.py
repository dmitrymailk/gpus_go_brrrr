from transformers import AutoTokenizer, AutoConfig
from datasets import load_dataset
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DataCollatorWithPadding, DataCollatorForLanguageModeling

import abc
from transformers import BatchEncoding, PreTrainedTokenizer
from typing import Dict, List, Tuple, Union

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
import sys
from functools import partial


def random_seed(seed=42, rank=0):
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)


class OpenOrcaSystemDataPrompter:
    """
    Alpaca Style Prompter that uses system prompts from the dataset, with OpenOrca prompts
    """

    def get_prompt(
        self,
        instruction="",
        system="",
    ):
        self.instruction_prompt = (
            f"<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n"
        )
        self.system_prompt = f"<|im_start|>system\n{system}<|im_end|>\n"
        return self.system_prompt + self.instruction_prompt


class PromptTokenizingStrategy(abc.ABC):
    """
    Abstract class for tokenizing strategies
    """

    def __init__(
        self,
        prompter=None,
        tokenizer=None,
        train_on_inputs: bool = False,
        sequence_len: int = 2048,
    ):
        self.prompter = prompter
        self.tokenizer: PreTrainedTokenizer = tokenizer
        self.train_on_inputs = train_on_inputs
        # sequence_len and max_length can be different for CompletionPromptTokenizingStrategy.
        # TODO: Document how they are different.
        self.sequence_len = sequence_len
        self.max_length = sequence_len

        self.prompter = OpenOrcaSystemDataPrompter()

    @abc.abstractmethod
    def tokenize_prompt(self, prompt):
        pass

    @property
    def supports_batched(self):
        return False

    def _tokenize(
        self, prompt: str, add_eos_token: bool = True, strip_bos_token: bool = False
    ) -> BatchEncoding:
        empty = BatchEncoding(data={"input_ids": [], "attention_mask": []})
        if not prompt:
            print("Empty text requested for tokenization.")
            return empty

        result = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors=None,
        )
        if len(result["input_ids"]) == 0:
            print("Tokenizer result is empty. You may want to audit your dataset")
            return empty

        if (
            result["input_ids"][-1] != self.tokenizer.eos_token_id
            and len(result["input_ids"]) < self.max_length
            and add_eos_token
        ):
            result["input_ids"].append(self.tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        if result["input_ids"][0] == self.tokenizer.bos_token_id and strip_bos_token:
            result["input_ids"] = result["input_ids"][1:]
            result["attention_mask"] = result["attention_mask"][1:]

        result["labels"] = result["input_ids"].copy()
        return result


class InstructionPromptTokenizingStrategy(PromptTokenizingStrategy):
    """
    Tokenizing strategy for instruction-based prompts.
    """

    def parse_instruction_fields(
        self, prompt
    ) -> Union[Tuple[str, str, str], Tuple[str, str, str, str]]:
        return (prompt["system_prompt"], prompt["question"], prompt["response"])

    def tokenize_prompt(self, prompt):
        (
            instruction,
            input,  # pylint: disable=redefined-builtin
            response,
        ) = self.parse_instruction_fields(prompt)

        user_prompt = self.prompter.get_prompt(
            instruction=input,
            system=instruction,
        )
        tokenized_prompt = self._tokenize(user_prompt, add_eos_token=False)
        if not self.train_on_inputs:
            user_prompt_len = len(tokenized_prompt["input_ids"])
            # TODO this could be sped up using numpy array slicing
            tokenized_prompt["labels"] = [-100] * user_prompt_len

        tokenized_res_prompt = self._tokenize(
            response, strip_bos_token=True, add_eos_token=True
        )
        tokenized_prompt["input_ids"] += tokenized_res_prompt["input_ids"]
        tokenized_prompt["attention_mask"] += tokenized_res_prompt["attention_mask"]
        tokenized_prompt["labels"] += tokenized_res_prompt["input_ids"]

        return tokenized_prompt


class OpenOrcaDataset(Dataset):
    def __init__(
        self,
        dataset=None,
        tokenizer=None,
    ):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.prompt_tokenizer = InstructionPromptTokenizingStrategy(tokenizer=tokenizer)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        dataset_item = self.dataset[idx]
        inputs = self.prompt_tokenizer.tokenize_prompt(dataset_item)

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


def assign_new_weights(original_module, original_weights, r=16):
    L, R = get_L_R(original_weights, rank=r)
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


def pad_datacollator(batch, tokenizer=None):
    inputs = {}
    max_length = 0
    for item in batch:
        max_length = max(max_length, len(item["input_ids"]))

    for item in batch:
        input_ids = item["input_ids"]
        diff = max_length - len(input_ids)
        item["input_ids"] = [tokenizer.pad_token_id] * diff + input_ids
        item["attention_mask"] = [0] * diff + item["attention_mask"]
        item["labels"] = [-100] * diff + item["labels"]
        for key in item.keys():
            if key in inputs:
                inputs[key].append(item[key])
            else:
                inputs[key] = [item[key]]

    for key in inputs.keys():
        inputs[key] = torch.tensor(inputs[key])

    return inputs


if __name__ == "__main__":
    model_name = "Open-Orca/Mistral-7B-OpenOrca"
    lora_model_name = "ebany_research/llm_lora/models/"
    lora_model_name += "openorca_lora_[17][17c]"
    # lora_model_name = model_name
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

    valid_dataloader = DataLoader(
        dataset=valid_dataset,
        batch_size=batch_size,
        collate_fn=partial(
            pad_datacollator,
            tokenizer=tokenizer,
        ),
        shuffle=False,
    )

    # test
    next(iter(valid_dataloader))
    
    # 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 [17] 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32    
    distill_layers = [
        11,
        17,
        22,
        26,
    ]
    r = 256
    config.lora_layers = config.to_dict().get('lora_layers', []) + distill_layers

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
            r=r,
            bias=False,
        )
        lora_up_proj = LinearLora(
            in_dim=config.hidden_size,
            out_dim=config.intermediate_size,
            r=r,
            bias=False,
        )
        lora_down_proj = LinearLora(
            in_dim=config.intermediate_size,
            out_dim=config.hidden_size,
            r=r,
            bias=False,
        )

        assign_new_weights(
            original_module=lora_gate_proj,
            original_weights=student_gate_proj,
            r=r,
        )
        assign_new_weights(
            original_module=lora_up_proj,
            original_weights=student_up_proj,
            r=r,
        )
        assign_new_weights(
            original_module=lora_down_proj,
            original_weights=student_down_proj,
            r=r,
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
    if "[" in lora_model_name:
        name = lora_model_name.split("[")[-1][:-1]
        name = name.split("_")
        name = [int(item.replace("c", "")) for item in name]
        name.extend(distill_layers)
        name = sorted(list(set(name)))
        config.lora_layers = name
        name = [str(item) for item in name]
        name = "_".join(name)
        lora_model_name += f"[{name}]"
        print(lora_model_name)
    else:
        name = sorted(list(set(distill_layers)))
        config.lora_layers = name
        name = [str(item) for item in name]
        name = "_".join(name)
        lora_model_name = "ebany_research/llm_lora/models/"
        lora_model_name += f"openorca_lora_[{name}]"
    print(lora_model_name)    
    # student_model.save_pretrained(lora_model_name)
