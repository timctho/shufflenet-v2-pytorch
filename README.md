# Shufflenet-v2-pytorch
Pytorch implementation of ECCV 2018 paper ShuffleNet V2. [[Paper]](https://arxiv.org/abs/1807.11164)

# Support architectures
  * ShuffleNetV2 with 0.5, 1.0, 1.5, 2.0 channel multipliers
  * ShuffleNetV2-50, ShuffleNetV2-164 with residual connections and SE modules

# Usage
```python
from net import ShuffleNetV2, ShuffleResNetV2
import torch

if __name__ == '__main__':
    # Create dummy input
    size = 224
    dummy = torch.rand(2, 3, size, size)

    # Create model
    net = ShuffleNetV2(size, size, 3, class_num=1000, model_scale=1.0)
    # net = ShuffleResNetV2(size, size, 3, class_num=1000, model_arch=50,
    #                       use_se_block=False, se_reduction=2)
    print(net)

    # Inference
    out = net(dummy)
    print(out.size())
```
