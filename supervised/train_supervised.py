import argparse
import os
import sys

import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

from model import CNN
from data_loader import get_data_loader

def main(args):
    print(sys.argv)

    if not os.path.exists('models'):
        os.mkdir('models')

    num_epochs = args.ne
    lr_decay = args.decay
    learning_rate = args.lr

    data_loader = get_data_loader(args.patches_path, args.gt_path, args.bs, num_workers=8)
    model = CNN()
    if torch.cuda.is_available():
        model.cuda()
    model.train()

    if args.rms:
        optimizer = optim.RMSprop(model.parameters(), lr=args.lr, momentum=args.mm)
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    model_loss = torch.nn.CrossEntropyLoss()

    losses = []
    try:
        for epoch in range(num_epochs):
            if epoch % args.decay_epoch == 0 and epoch > 0:
                learning_rate = learning_rate * lr_decay
                for param_group in optimizer.param_groups:
                    param_group['lr'] = learning_rate


            loss_epoch = []
            for step, (patches, gt) in enumerate(data_loader):
                if torch.cuda.is_available():
                    patches = patches.cuda()
                    gt = gt.cuda()
                model.zero_grad()

                out = model(patches)
                loss = model_loss(out, gt)
                loss.backward()
                optimizer.step()

                loss_step = loss.cpu().detach().numpy()
                loss_epoch.append(loss_step)

                print('Epoch ' + str(epoch + 1) + '/' + str(num_epochs) + ' - Step '
                       + str(step + 1) + '/' + str(len(data_loader)) + " - Loss: " + str(loss_step))

            loss_epoch_mean = np.mean(np.array(loss_epoch))
            losses.append(loss_epoch_mean)
            print('Total epoch loss: ' + str(loss_epoch_mean))

            if (epoch + 1) % args.save_epoch == 0 and epoch > 0:
                filename = 'model-epoch-' + str(epoch + 1) + '.pth'
                model_path = os.path.join('models/', filename)
                torch.save(model.state_dict(), model_path)
    except KeyboardInterrupt:
        pass

    filename = 'model-epoch-last.pth'
    model_path = os.path.join('models', filename)
    torch.save(model.state_dict(), model_path)
    plt.plot(losses)
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--patches_path', type=str, default='../datasets/simple/train_supervised.npy',
                        help='Path to the patches')
    parser.add_argument('--gt_path', type=str, default='../datasets/simple/train_supervised_gt.npy',
                        help='Path to the ground truth labels')
    parser.add_argument('--n_workers', type=int, default=8,
                        help='Number of subprocesses used for data loading')
    parser.add_argument('-lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('-mm', type=float, default=0.9,
                        help='Momentum')
    parser.add_argument('--save_epoch', type=int, default=2,
                        help='Epoch where we want our model to be saved')
    parser.add_argument('-ne', type=int, default=150, help='Number of epochs')
    parser.add_argument('-bs', type=int, default=512, help='Size of the batch')
    parser.add_argument('--decay', type=float, default=0.1,
                        help='Decay of the learning rate')
    parser.add_argument('--decay_epoch', type=int, default=10,
                        help='Indicates the epoch where we want to reduce the learning rate')
    parser.add_argument('-rms', action='store_true',
                        help='RMSProp option')
    args = parser.parse_args()

    main(args)
