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
