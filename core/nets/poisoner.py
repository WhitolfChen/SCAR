import torch
import torch.nn as nn
from torchvision.utils import save_image

class Poisoner(nn.Module):
    def __init__(self, size=32):
        super().__init__()
        self.trigger = nn.Parameter(torch.rand(3, size, size), requires_grad=True)

    def forward(self, inputs):
        poisoned_inputs = inputs + self.trigger
        return torch.clamp(poisoned_inputs, 0.0, 1.0)
    
    def project(self, epsilon=0.1):
        self.trigger.data = torch.clamp(self.trigger.data, -epsilon, epsilon)
    
    def visualize(self, path):
        save_image(self.trigger.detach(), path)