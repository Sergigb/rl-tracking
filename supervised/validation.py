import argparse

import torch
import numpy as np
import matplotlib.pyplot as plt

from model import CNN


def get_action(action):
    if(action == 0):
        return"go left"
    if(action == 1):
        return"go right"
    if(action == 2):
        return"go up"
    if(action == 3):
        return"go down"
    if(action == 4):
        return"scale up"
    if(action == 5):
        return"scale down"
    if(action == 6):
        return"do nothing"


parser = argparse.ArgumentParser()
parser.add_argument('-v', action='store_true',
                    help='Visualize the patch and the result')
args = parser.parse_args()


patches = np.load("datasets/simple/val_supervised.npy")
patches = torch.from_numpy(patches).type(torch.FloatTensor).cuda().unsqueeze(1)
gt = np.load("datasets/simple/val_supervised_gt.npy")

model_path = "models/model-epoch-40.pth"
model = CNN()
model.load_state_dict(torch.load(model_path))

if torch.cuda.is_available():
    model.cuda()
model.eval()

out = model(patches)

patches = patches.cpu()

right = 0
wrong = 0

for i in range(patches.shape[0]):
    pred = torch.argmax(out[i])
    if int(pred) == int(gt[i]):
        right += 1
    else:
        wrong += 1

    if args.v:
        print("predicted: ", get_action(pred))
        print("gt: ", get_action(gt[i]))
        fig, ax = plt.subplots(1)
        ax.imshow(patches[i, 0, :])
        plt.show()

print(right, wrong)

