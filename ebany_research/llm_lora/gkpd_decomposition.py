from transformers import AutoTokenizer, AutoConfig, MistralForCausalLM
import torch
from ebany_research.llm_lora.generalized_kronecker_product_decomposition.gkpd import (
    gkpd,
    kron,
)


if __name__ == "__main__":
    model_name = "Open-Orca/Mistral-7B-OpenOrca"
    # model_name = "ebany_research/llm_lora/train_results"
    config = AutoConfig.from_pretrained(model_name)
    device = 0
    model = MistralForCausalLM.from_pretrained(
        model_name, device_map={"": device}, torch_dtype=torch.bfloat16
    )
    model = model.eval()
    gate_proj = model.model.layers[17].mlp.gate_proj.weight.data.cpu().to(torch.float32)
    print('gate_proj')
    
    m1, m2 = 224, 64
    n1, n2 = 64, 64
    a_shape = (m1, n1)
    b_shape = (m2, n2)

    a_hat, b_hat = gkpd(gate_proj, a_shape, b_shape)
    w_hat = torch.kron(a_hat, b_hat)

    print(
        "Reconstruction error: {}".format(
            round(
                (
                    torch.linalg.norm((gate_proj.reshape(-1) - w_hat.reshape(-1)))
                    .detach()
                    .numpy()
                ).item(),
                4,
            )
        )
    )
