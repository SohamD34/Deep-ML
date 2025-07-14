import math

class CosineAnnealingLRScheduler:
    def __init__(self, initial_lr, T_max, min_lr):
        self.init_lr = initial_lr
        self.t_max = T_max
        self.min_lr = min_lr

    def get_lr(self, epoch):
        return self.min_lr + 0.5 * (self.init_lr - self.min_lr) * (1 + math.cos(epoch * math.pi / self.t_max))



scheduler = CosineAnnealingLRScheduler(initial_lr=0.1, T_max=10, min_lr=0.001)
print(f"{scheduler.get_lr(epoch=0):.4f}")
print(f"{scheduler.get_lr(epoch=2):.4f}")
print(f"{scheduler.get_lr(epoch=5):.4f}")
print(f"{scheduler.get_lr(epoch=7):.4f}")
print(f"{scheduler.get_lr(epoch=10):.4f}")