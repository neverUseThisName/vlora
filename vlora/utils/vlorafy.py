import torch
from torch import nn
import torch.nn.utils.parametrize as parametrize


def vlorafy(
    base: type[nn.Module],
):
    return type("_vlorafied", (base,), {"__init__": init_fn})


def init_fn(
    self,
    *args,
    vl_base_module: nn.Module,
    vl_param_name: str | list[str],
    vl_rank: int | list[int],
    vl_alpha: int | list[int] = 1,
    vl_enabled: bool = True,
    **kwargs,
):
    super(self.__class__, self).__init__(*args, **kwargs)

    if isinstance(vl_param_name, str):
        vl_param_name = [vl_param_name]
    if isinstance(vl_rank, int):
        vl_rank = [vl_rank] * len(vl_param_name)
    if isinstance(vl_alpha, int):
        vl_alpha = [vl_alpha] * len(vl_param_name)
    assert len(vl_param_name) == len(vl_rank) == len(vl_alpha)
    name2param = dict(vl_base_module.named_parameters())
    assert all(
        name in name2param for name in vl_param_name
    ), f"{vl_param_name}\n{list(name2param.keys())}"
    for name, r, a in zip(vl_param_name, vl_rank, vl_alpha):
        *mod_names, param_name = name.split(".")
        base_mod, self_mod = vl_base_module, self
        for mod_name in mod_names:
            base_mod = base_mod.__getattr__(mod_name)
            self_mod = self_mod.__getattr__(mod_name)
        param = base_mod.__getattr__(param_name)
        self_mod.__setattr__(param_name, param)
        if len(param.shape) == 1:
            parametrize.register_parametrization(
                self_mod,
                param_name,
                VLoRAParam1D(param.shape[0], param.dtype, param.device, vl_enabled),
            )
        elif len(param.shape) == 2:
            dim_out, dim_in = param.shape
            parametrize.register_parametrization(
                self_mod,
                param_name,
                VLoRAParam2D(
                    dim_in, dim_out, r, param.dtype, param.device, a, vl_enabled
                ),
            )
        else:
            raise NotImplementedError


class VLoRAParam1D(nn.Module):
    def __init__(
        self,
        dim: int,
        dtype: torch.dtype,
        device: torch.device | str,
        enabled: bool = True,
    ):
        super().__init__()
        self.enabled = enabled
        self.A = nn.Parameter(torch.zeros(dim, dtype=dtype, device=device))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.zero_(self.A)

    def forward(self, param: torch.Tensor) -> torch.Tensor:
        if self.enabled:
            return param.detach() + self.A
        return param


class VLoRAParam2D(nn.Module):
    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        rank: int,
        dtype: torch.dtype,
        device: torch.device | str,
        alpha: int = 1,
        enabled: bool = True,
    ):
        assert rank <= dim_in and rank <= dim_out
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.enabled = enabled
        self.A = nn.Parameter(torch.zeros((rank, dim_in), dtype=dtype, device=device))
        self.B = nn.Parameter(torch.zeros((rank, dim_out), dtype=dtype, device=device))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.A, a=5**0.5)
        nn.init.kaiming_uniform_(self.B, a=5**0.5)

    def forward(self, param: torch.Tensor) -> torch.Tensor:
        if self.enabled:
            return param.detach() + (self.B.T @ self.A) * (self.alpha / self.rank)
        return param
