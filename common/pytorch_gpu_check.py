import torch
import torch.nn as nn


class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(1, 2)

    def forward(self, x):
        return self.l1(x)


def main():
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(torch.cuda.current_device())
        print(torch.cuda.device(0))
        print(torch.cuda.device_count())
        print(torch.cuda.get_device_name(0))

    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    t1 = torch.randn(1, 2)
    t2 = torch.randn(1, 2).to(dev)
    print(t1)
    print(t2)
    t1.to(dev)  # not in-place: t1 is still on CPU
    print(t1.is_cuda)  # False
    t1 = t1.to(dev)
    print(t1)
    print(t1.is_cuda)  # True only when CUDA is available

    model = M()  # created on CPU
    model.to(dev)  # moves all parameters to the device
    print(next(model.parameters()).device)


if __name__ == '__main__':
    main()
