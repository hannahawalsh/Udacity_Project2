"""
Prediction Script for Project 2
"""

### Imports
import warnings
warnings.filterwarnings("ignore")

import argparse
import os
import torch
from torch import nn, optim
from torchvision import models
import numpy as np
from PIL import Image
import json
import build_model


### Create a parser for the command line
parser = argparse.ArgumentParser(description='Predict flower using classifier')
parser.add_argument('img_path', metavar='path to image', type=str,
                    help='the image to be classified')
parser.add_argument('ckp_path', metavar='path to checkpoint', type=str,
                    help='the checkpoint file of the trained model')
parser.add_argument('--topk', metavar='save directory', type=int,
                    default=1, help='predict the top k classes of the image')
parser.add_argument('--category_names', metavar='category names', type=str,
                    default = 'cat_to_name.json',
                    help='path to json file with category names')
parser.add_argument('--gpu', dest='device', metavar='device',
                    action='store_const', const='gpu',
                    default = 'cpu', help='use the gpu (default: cpu)')
args = parser.parse_args()


### Load model from checkpoint path
checkpoint = torch.load(args.ckp_path, map_location=args.device)
ep = checkpoint['epoch']

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

### Build the model from the checkpoint

##################
checkpoint['arch'] = 'vgg11_bn'
##################



def load_model(checkpoint):
    model = build_model.create_model(checkpoint['arch'],
                                     checkpoint['hidden_units'],
                                     checkpoint['output_size'],
                                     checkpoint['dropout'],
                                     pytorch_models)
    for param in model.parameters():
        param.requires_grad = False

    model.load_state_dict(checkpoint['model_state_dict'])
    model.class_to_idx = checkpoint["class_to_idx"]

    lr = checkpoint["learning_rate"]
    op = optim.Adam(model.classifier.parameters(), lr=lr)
    op.load_state_dict(checkpoint['optimizer_state_dict'])

    return model, op

model, optimizer = load_model(checkpoint)
criterion = nn.NLLLoss()
model.to(args.device)


### Process the image
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # Open the image
    im = Image.open(image)

    # Resize the image, where the shortest side is 256 pixels, keeping aspect ratio
    size = (256, 256)
    im = im.resize(size)

    # Center crop 224x224
    w, h = im.size
    crop_size = 224
    left = (w - crop_size)/2
    top = (h - crop_size)/2
    right = (w + crop_size)/2
    bottom = (h + crop_size)/2
    im = im.crop((left, top, right, bottom))

    # Convert to numpy array and normalize
    means = [0.485, 0.456, 0.406]
    sds = [0.229, 0.224, 0.225]
    np_image = np.array(im)
    np_image = np.true_divide(np_image, 255)
    for i in range(3):
        np_image[i,:,:] = (np_image[i,:,:] - means[i]) / sds[i]

    return np_image.transpose()


def predict(image_path, model, topk=1):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    with torch.no_grad():
        # Map the index to class
        idx_to_class = {v:k for k, v in model.class_to_idx.items()}

        # Create an image tensor
        im_tensor = torch.tensor(process_image(image_path)).float()
        im_tensor = im_tensor.view(1, 3, 224, 224)

        # Predict all the probabilities
        probs = torch.exp(model.forward(im_tensor))

        # Find the top probabilities and indexes
        top_probs, top_k_idx = probs.topk(topk+1, dim=1)

        # Transform the top indexes into the top classes
        classes = [idx_to_class[k] for k in top_k_idx.data.numpy().squeeze()]

    return top_probs.data.numpy().squeeze().tolist()[:-1], classes[:-1]



### Make predictions and print them out
probs, classes = predict(args.img_path, model, args.topk)

try:
    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)
    categories = [cat_to_name[c] for c in classes] # Map class numbers to names
except:
    print("No file detected to map classes to flower names.")
    print("Showing category numers instead:\n")

if len(classes) == 1:
    try:
        print("Top Prediction: {} - probability {:.4f}".format(categories[0], probs[0]))
    except:
        print("Top Prediction: class {} - probability {:.4f}".format(classes[0], probs[0]))
else:
    print("Top {} Predictions".format(args.topk))
    print("-"*17)
    try:
        for i, e in enumerate(categories):
            print("{}. {} - probability {:.4f}".format(i+1, e, probs[i]))
    except:
        for i, e in enumerate(classes):
            print("{}. class {} - probability {:.4f}".format(i+1, e, probs[i]))
