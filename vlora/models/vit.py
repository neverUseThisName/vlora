from functools import partial

from timm.models.vision_transformer import VisionTransformer, Block

from .common import VLoRACompound


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
