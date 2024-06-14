import os

import torch


def test_forward():
    from vlora.models import TransformerVLoRA

    model = TransformerVLoRA()
    x = torch.randn((1, 3, 224, 224))
    y = model(x)


def test_save_and_load():
    from vlora.models import TransformerVLoRA

    model = TransformerVLoRA()
    torch.save(model.state_dict(), "vlora.pt")
    model.load_state_dict(torch.load("vlora.pt"))
    os.remove("vlora.pt")


def test_vlora_params_count():
    from vlora.models import TransformerVLoRA
    from torch.nn import Transformer

    def count_params(model):
        n = 0
        for param in model.parameters():
            if param.requires_grad:
                n += param.nelement()
        return n

    model_vlora = TransformerVLoRA()
    model = Transformer()
    assert count_params(model_vlora) * 2 < count_params(model)
