"""Define your architecture here."""
import torch
from models import SimpleNet
from torch import nn
import torchvision.models as models


def my_competition_model(model_args,load):
    """Override the model initialization here.

    Do not change the model load line.
    """
    


    model = models.regnet_x_400mf(pretrained=True)
    model.classifier = nn.Sequential(
        nn.Linear(576, 256),
        nn.ReLU(),
        nn.Linear(256, 64),
        nn.ReLU(),
        nn.Linear(64, 2))


    # load your model using exactly this line (don't change it):
    model.load_state_dict(torch.load('checkpoints/fakes_dataset_my_competition_model_Adam.pt')['model'])
    return model
