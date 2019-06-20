"""
Training Script for Project 2
"""

### Imports
from __future__ import print_function

import warnings
warnings.filterwarnings("ignore")

import argparse
import os
import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
import json
from collections import OrderedDict
import build_model


### Create a parser for the command line
parser = argparse.ArgumentParser(description='Train a flower classifier')
parser.add_argument('data_dir', metavar='data directory', type=str,
                    help='the directory in which the data is located')
parser.add_argument('--save_dir', metavar='save directory', type=str,
                    default=os.getcwd(),
                    help='directory in which to save the trained model')
parser.add_argument('--arch', metavar='model architecture', type=str,
                    default='vgg13', help='architecture of the base model')
parser.add_argument('--learning_rate', metavar='learning rate', type=float,
                    default=0.01, help='learning rate for model')
parser.add_argument('--hidden_units', metavar='hidden units', type=int,
                    nargs='+', default=512, help='hidden units in the model')
parser.add_argument('--epochs', metavar='epochs', type=int,
                    default=20, help='number of epochs to train the model for')
parser.add_argument('--gpu', dest='device', metavar='device',
                    action='store_const', const='gpu',
                    default = 'cpu', help='use the gpu (default: cpu)')

args = parser.parse_args()


### Define the directories
data_dir = args.data_dir
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'


### Transform and load the data
# Define transforms for the data
means = [0.485, 0.456, 0.406]
stds = [0.229, 0.224, 0.225]
train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize(means, stds)])
test_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize(means, stds)])

# Load the datasets with ImageFolder
train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
valid_data = datasets.ImageFolder(valid_dir, transform=test_transforms)
test_data = datasets.ImageFolder(test_dir, transform=test_transforms)

# Define the dataloaders
trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
validloader = torch.utils.data.DataLoader(valid_data, batch_size=64, shuffle=True)
testloader = torch.utils.data.DataLoader(test_data, batch_size=64)


### Load a pre-trained model
# Define the supported architectures
pytorch_models = {"vgg11": models.vgg11,
                  "vgg13": models.vgg13,
                  "vgg16": models.vgg16,
                  "vgg19": models.vgg19,
                  "vgg11_bn": models.vgg11_bn,
                  "vgg13_bn": models.vgg13_bn,
                  "vgg16_bn": models.vgg16_bn,
                  "vgg19_bn": models.vgg19_bn,
                  "alexnet": models.alexnet,
                  "resnet18": models.resnet18,
                  "resnet34": models.resnet18,
                  "resnet50": models.resnet18,
                  "resnet101": models.resnet18,
                  "resnet152": models.resnet18,
                  "densenet121": models.densenet121,
                  "densenet169": models.densenet169,
                  "densenet201": models.densenet201,
                  "densenet161": models.densenet161,
                  "inception_v3": models.inception_v3}

### Load mapping
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)


### Create our model
output_size = len(cat_to_name)
dropout = 0.4
model = build_model.create_model(args.arch, args.hidden_units, output_size,
                                 dropout, pytorch_models)
criterion = nn.NLLLoss()
model.class_to_idx = train_data.class_to_idx


### Choose device and transfer over model
device = torch.device("cuda" if args.device == "gpu" and
                      torch.cuda.is_available() else "cpu")
model.to(device)
print("Using {}".format(device))
print("GPU is {}available".format(" " if torch.cuda.is_available() else "not "))


### Train and validate the final model
print("Training model...")

optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)

epochs = args.epochs
steps = 0
running_loss = 0
print_every = 10

for epoch in range(epochs):
    for inputs, labels in trainloader:
        steps += 1
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma = 0.95)

        logps = model.forward(inputs)
        loss = criterion(logps, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        valid_loss = 0
        accuracy = 0
        model.eval()
        with torch.no_grad():
            for inputs, labels in validloader:
                inputs, labels = inputs.to(device), labels.to(device)
                logps = model.forward(inputs)
                batch_loss = criterion(logps, labels)

                valid_loss += batch_loss.item()

                # Calculate accuracy
                ps = torch.exp(logps)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

        scheduler.step()


        if steps % print_every == 0:
            print("EPOCH {epoch+1}/{epochs}\n" +\
                  "Training loss:       {running_loss/print_every:.3f}\n" +\
                  "Validation loss:     {valid_loss/len(validloader):.3f}\n" +\
                  "Validation accuracy: {accuracy/len(validloader)  100:.2f}%\n" +\
                  "-"*22 + "\n")
            running_loss = 0
        model.train()
print("DONE!")

### Save the model to checkpoint.pth in the save_dir
checkpoint = {'input_size': build_model.find_input_size(args.arch, model),
              'hidden_units': args.hidden_units,
              'output_size': len(cat_to_name),
              'dropout': args.dropout,
              'model_state_dict': model.state_dict(),
              'optimizer_state_dict': optimizer.state_dict(),
              'class_to_idx' : model.class_to_idx,
              'epoch': epoch,
              'arch': args.arch,
              'learning_rate': args.learning_rate}

torch.save(checkpoint, os.path.join(args.save_dir, 'checkpoint.pth'))
print("Model saved to {}".format(os.path.join(args.save_dir, 'checkpoint.pth')))
