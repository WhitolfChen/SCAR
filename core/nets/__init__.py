from . import resnet_cifar
from . import mobilenetv2
from . import shufflenetv2
from . import shufflenetv2_cifar
from . import efficientvit
from .poisoner import Poisoner
import torchvision.models as models
import torch

def get_network(dataset, model, ckp_path=None, device='cpu'):
    # The network of cifar10
    if dataset == "cifar10":
        if model == "resnet18":
            network = resnet_cifar.resnet18()
        elif model == "resnet50":
            network = resnet_cifar.resnet50()
        elif model == "mobilenetv2":
            network = mobilenetv2.mobilenetv2_cifar()
        elif model == "shufflenetv2":
            network = shufflenetv2_cifar.shufflenetv2()
        elif model == "efficientvit":
            network = efficientvit.efficientvit_cifar()
        elif model == "poisoner":
            network = Poisoner(size=32)
        else:
            raise NotImplementedError
    
    # The network of imagenet50
    elif dataset == "imagenet50":
        if model == "resnet18":
            network = models.resnet18(weights=None, num_classes=50)
        elif model == "resnet50":
            network = models.resnet50(weights=None, num_classes=50)
        elif model == "mobilenetv2":
            network = mobilenetv2.mobilenetv2_imagenet()
        elif model == "shufflenetv2":
            network = shufflenetv2.shufflenetv2_imagenet()
        elif model == "efficientvit":
            network = efficientvit.efficientvit_imagenet()
        elif model == "poisoner":
            network = Poisoner(size=224)
        else:
            raise NotImplementedError
                
    else:
        raise NotImplementedError
    
    if ckp_path:
        network.load_state_dict(torch.load(ckp_path, weights_only=True, map_location=device))

    return network
