import torch


def test_transformer():
    from timm.models import VisionTransformerVLoRA

    vit = VisionTransformerVLoRA()
    x = torch.randn((1, 3, 224, 224))
    y = vit(x)
    print(y.shape)


if __name__ == "__main__":
    test_transformer()
