import torch

from models import caption
from configuration import Config

dependencies = ['torch', 'torchvision']

def v1(pretrained=False):
    config = Config()
    model, _ = caption.build_model(config)
    
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            map_location='cpu'
        )
        model.load_state_dict(checkpoint['model'])
    
    return model

def v2(pretrained=False):
    config = Config()
    model, _ = caption.build_model(config)
    
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            map_location='cpu'
        )
        model.load_state_dict(checkpoint['model'])
    
    return model

def v3(pretrained=False):
    config = Config()
    model, _ = caption.build_model(config)
    
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            map_location='cpu'
        )
        model.load_state_dict(checkpoint['model'])
    
    return model