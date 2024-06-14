# Intro
Official repository for the paper [Vertical LoRA: Dense Expectation-Maximization Interpretation of Transformers](https://arxiv.org/abs/2406.09315).
VLoRA reduces the parameter count of Transformers by about 75% while preserving their performance.

# Install
Run
```
pip install vlora
```

# Out-of-box usage
VLoRA has some built-in models you can use directly. The following example imports, init a VLoRA-ViT model and performs a forward pass.


```python
from vlora.models import VisionTransformerVLoRA

vit = VisionTransformerVLoRA()
x = torch.randn((1, 3, 224, 224))
y = vit(x)

```

# Customized usage
You can apply VloRA to your own layer and use it as the building block of your model.

Assume 
- you have a model of class `Model` consisting of layers of class `Layer`.
- `Layer` is instantialized by `layer = Layer(a, b=b, c=c)`.
- `Layer` has a 2D parameter `L.mod.param`, which you want to vlorafy.

You can create a VLoRA Compound from your layer:

```python
from vlora.models import VLoRACompound

comp = VLoRACompound(
    a,
    b=b,
    c=c,
    vl_base_cls=Layer,
    vl_param_name='mod.param',
    vl_size=3, # compound size
    vl_rank=2, # low rank r
    vl_alpha=1,
)
```

Then replace the layer with the compound in your model. Note if your model has `L` layers, you should have `L/vl_size` compounds in the vlorofied model.
