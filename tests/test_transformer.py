import os

import torch


def test_forward():
    from vlora.models import VisionTransformerVLoRA

    vit = VisionTransformerVLoRA()
    x = torch.randn((1, 3, 224, 224))
    y = vit(x)


def test_save_and_load():
    from vlora.models import VisionTransformerVLoRA

    vit = VisionTransformerVLoRA()
    torch.save(vit.state_dict(), "vlora_vit.pt")
    vit.load_state_dict(torch.load("vlora_vit.pt"))
    os.remove("vlora_vit.pt")
