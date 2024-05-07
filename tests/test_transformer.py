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


def test_vlora_params_count():
    from vlora.models import VisionTransformerVLoRA
    from timm.models import VisionTransformer
    
    def count_params(model):
        n = 0
        for param in model.parameters():
            n += param.nelement()
        return n

    vlora_vit = VisionTransformerVLoRA()
    vit = VisionTransformer()
    assert count_params(vlora_vit) * 2 < count_params(vit)

