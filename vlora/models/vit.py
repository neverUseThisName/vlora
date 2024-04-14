from functools import partial

from torch import nn
from timm.models.vision_transformer import VisionTransformer, Block

from ..utils import vlorafy


DEFAULT_PARAM_NAMES = [
    "attn.qkv.weight",
    "attn.proj.weight",
    "mlp.fc1.weight",
    "mlp.fc2.weight",
]


class VisionTransformerVLoRA(VisionTransformer):
    def __init__(
        self,
        *args,
        depth: int = 3,
        vl_size: int = 4,
        vl_rank: int | list[int] = 8,
        vl_param_name: str | list[str] = DEFAULT_PARAM_NAMES,
        vl_alpha: int | list[int] = 1,
        vl_enabled: bool = True,
        **kwargs,
    ):
        block_fn = partial(
            VLoRACompound,
            vl_base_cls=Block,
            vl_param_name=vl_param_name,
            vl_size=vl_size,
            vl_rank=vl_rank,
            vl_alpha=vl_alpha,
            vl_enabled=vl_enabled,
        )
        super().__init__(*args, depth=depth, block_fn=block_fn, **kwargs)


class VLoRACompound(nn.Module):
    def __init__(
        self,
        *args,
        vl_base_cls: type[nn.Module],
        vl_param_name: str | list[str],
        vl_size: int,
        vl_rank: int | list[int],
        vl_alpha: int | list[int],
        vl_enabled: bool = True,
        **kwargs,
    ):
        super().__init__()
        vl_cls = vlorafy(vl_base_cls)
        base_block = vl_base_cls(*args, **kwargs)
        self.blocks = nn.ModuleList([base_block])
        for _ in range(vl_size - 1):
            vl_block = vl_cls(
                *args,
                vl_base_module=base_block,
                vl_param_name=vl_param_name,
                vl_rank=vl_rank,
                vl_alpha=vl_alpha,
                vl_enabled=vl_enabled,
                **kwargs,
            )
            self.blocks.append(vl_block)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x
