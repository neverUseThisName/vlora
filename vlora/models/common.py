import torch.nn as nn

from ..utils import vlorafy


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
