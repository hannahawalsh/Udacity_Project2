### Austomatically detect the input size from the architecture
# That way we don't have to hard code anything
import torch
from torch import nn, optim
from collections import OrderedDict
from torchvision import models


def find_input_size(arch, model):
    """
    Austomatically detect the input size from the architecture of the model
    Inputs:
        arch: string specifiying pretrained model architecture
        model: downloaded model of the architecture arch
    Output:
        input_size: an int of the input size of the classifier
    """
    if "vgg" in arch or arch == "alexnet":
        for i in range(4):
            if type(model.classifier[i]) == torch.nn.modules.linear.Linear:
                input_size = model.classifier[i].in_features
                break
    elif "resnet" in arch or arch == "inception_v3":
        input_size = model.fc.in_features
    elif "densenet" in arch:
        input_size= model.classifier.in_features

    return input_size



def create_model(arch, hidden_units, output_size, dropout, pytorch_models):
    """
    Create a Pytorch model with a base of a pretrained model and a classifier
    with specified parameters
    Inputs:
        arch: string specifiying pretrained model architecture
        hidden_units: int or list of hidden units for the classifier
        output_size: output size of the classifier
        dropout: dropout for the classifier
        pytorch_models: a dictionary mapping arch string to pytorch model
    Output:
        model: a pytorch model with a classifier
    """
    # Download pretrained model from given architecture:
    model = pytorch_models[arch](pretrained=True)

    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False

    # Change the classifier to fit our needs, according to given parameters
    if isinstance(hidden_units, int):
        hidden_units = [hidden_units]
    input_size = find_input_size(arch, model) # Find input size
    layer_sizes = [input_size] + hidden_units
    layers = [[('linear_{}'.format(i),
                nn.Linear(layer_sizes[i], layer_sizes[i+1])),
        ('relu_{}'.format(i), nn.ReLU()),
        ('dropout_{}'.format(i),
         nn.Dropout(dropout))] for i in range(len(layer_sizes)-1)]
    layers = [item for sublist in layers for item in sublist]
    layers += [('linear', nn.Linear(layer_sizes[-1], output_size)),
               ('logsoftmax', nn.LogSoftmax(dim=1))]
    model.classifier = nn.Sequential(OrderedDict(layers))

    return model
