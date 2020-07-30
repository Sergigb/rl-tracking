import argparse
import sys
import random
from math import floor
import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as pat
import cv2


def fill_sequences(sequences, args):
    """
    This ugly function generates n sequences where the target roams around and changes sizes, 
    should allways work. Well it very rarely violates the margins and produces an empty frame,
    but it's not common. Check the available arguments to change the size of the image, the 
    max. size of the raget and the max. speed of the target.

    The target never goes outside of the margins, which have a value of args.maxsize / 2

    The behaviour of the target can be changed by messing around with the different probabilities
    and the variance of the gaussian functions that are used. For example increasing the variance
    of the gaussian that generates the value added to dx/dy (eg. dx += random.gauss(0, variance)) 
    makes the movement more erratic, specially if the max. vel. is too low, so maybe you should
    decrease it. Lowering it makes the speed changes smaller and the target moves more smoothly 
    (but it barely steers), so I guess the idea is big image -> small variance and max speed, and 
    vice versa.

    This probably could be done faster if it was done purely with numpy arrays but I am lazy and
    speed doesn't seem to be a problem.

    Some values are clamped because they are generated from gaussian functions, and they may be
    too large (or too -large).
    """
    gt = np.zeros((sequences.shape[0], sequences.shape[1], 4))

    for i in range(sequences.shape[0]):
        x = random.uniform(args.maxsize / 2, args.size - (args.maxsize / 2))
        y = random.uniform(args.maxsize / 2, args.size - (args.maxsize / 2))
        # having velocity and delta-v helps smoothing the movement and changes in acceleration
        vx = random.gauss(0, 1.0)
        vy = random.gauss(0, 1.0)
        dx = random.gauss(0, 0.2)
        dx = max(min(dx, (args.size / 4) - 1), -(args.size / 4) - 1) # clamped d-v change
        dy = random.gauss(0, 0.2)
        dy = max(min(dy, (args.size / 4) - 1), -(args.size / 4) - 1)

        w = random.uniform(2, args.maxsize / 2)
        h = random.uniform(2, args.maxsize / 2)
        vw = random.gauss(0, 0.1) # not as much smoothing needed
        vh  = random.gauss(0, 0.1)
        vw = max(min(vw, (args.maxsize / 4) - 1), -(args.maxsize / 4) - 1) # clamped w/h change
        vh = max(min(vh, (args.maxsize / 4) - 1), -(args.maxsize / 4) - 1)

        for j in range(sequences.shape[1]):

            # increase or decrease dx/dy with p > 0.25
            if random.random() > 0.25:
                dx += random.gauss(0, 0.05)
                dx = max(min(dx, (args.size / 4) - 1), -(args.size / 4) - 1) # clamped d-v change
            if random.random() > 0.25:
                dy += random.gauss(0, 0.05)
                dy = max(min(dy, (args.size / 4) - 1), -(args.size / 4) - 1)

            # decrese vx/vy if we are going over the max vel. (or p < 0.1)
            if abs(vx + dx) > args.maxvel or random.random() < 0.1:
                dx = -dx
            if abs(vy + dy) > args.maxvel or random.random() < 0.1:
                dy = -dy

            vx += dx
            vy += dy

            # change direction if we are going over the margin
            if x + vx > args.size - (args.maxsize / 2) or x + vx < args.maxsize / 2:
                vx = -vx * random.random() * 0.5
                dx = -dx
            if y + vy > args.size - (args.maxsize / 2) or y + vy < args.maxsize / 2:
                vy= -vy * random.random() * 0.5
                dy = -dy

            x += vx
            y += vy

            # similar stuff with the size of the target
            if random.random() > 0.25:
                vw += random.gauss(0, 0.01)
                vw = max(min(vw, (args.maxsize / 4)), -(args.maxsize / 4)) # clamped w/h change
            if random.random() > 0.25:
                vh += random.gauss(0, 0.01)
                vh = max(min(vh, (args.maxsize / 4)), -(args.maxsize / 4))

            # check the max/min sizes, the min size = 2, this can be changed but then you should change the
            # clamping of the vh/vw values (or change vw = -vw to something else).
            if w + vw < 2 or w + vw > args.maxsize / 2.:
                vw = -vw
            if h + vh < 2 or h + vh > args.maxsize / 2.:
                vh = -vh

            w += vw
            h += vh

            sequences[i, j, int(y)-int(h):int(y)+int(h), int(x)-int(w):int(x)+int(w)] = 1
            gt[i, j] = np.array([floor(x) / args.size, floor(y) / args.size, (floor(w) * 2.) / args.size, (floor(h) * 2) / args.size])

            if np.sum(sequences[i, j]) == 0:
                print("empty frame?")
                print(i, j, x, y, w ,h, dx, dy, vw, vh)

    return gt


def main(args):
    sequences = np.zeros((args.number, args.length, args.size, args.size))
    gt = fill_sequences(sequences, args)

    if not os.path.exists("datasets"):
        os.mkdir("datasets")
        os.mkdir("datasets/simple")
    if not os.path.exists("datasets/simple"):
        os.mkdir("datasets/simple")

    np.save("datasets/simple/" + args.output + "_rl.npy", sequences)
    np.save("datasets/simple/" + args.output + "_rl_gt.npy", gt)


def visualize(args):
    sequence = np.zeros((1, args.length, args.size, args.size))

    gt = fill_sequences(sequence, args)

    if args.dumpvideo:
        for i in range(sequence.shape[0]):
            sequence = sequence * 255
            out = cv2.VideoWriter("videos/output.avi",cv2.VideoWriter_fourcc(*"DIVX"), 15, (args.size, args.size), 0)
            for j in range(sequence.shape[1]):
                out.write(np.uint8(sequence[i, j]))
            out.release

    else:
        fig, ax = plt.subplots(1, figsize=(15,15))
        gt = gt * args.size

        for i in range(args.length):
            plt.cla()
            ax.imshow(sequence[0, i, :])

            loc = (gt[0, i, 0] - gt[0, i, 2] / 2, gt[0, i, 1] - gt[0, i, 3] / 2)

            if args.visualizegt:
                rect = pat.Rectangle(loc, gt[0, i, 2], gt[0, i, 3], linewidth=1, edgecolor='r', facecolor='none')
                ax.add_patch(rect)

            plt.show(block=False)
            plt.pause(0.00001)

            sys.stdout.write(("\rFrame " + str(i+1) + "/" + str(args.length)))
            sys.stdout.flush()

        plt.close()
        print("")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--length", "-l", type=int, default=30,
                        help="Length of the generated sequences")
    parser.add_argument("--number", "-n", type=int, default=1,
                        help="Number of sequences to be generated")
    parser.add_argument("--size", "-s", type=int, default=64,
                        help="Size of the frames (equal for both sides)")
    parser.add_argument("--visualize", "-v", action='store_true',
                        help="Visualize one generated sequence")
    parser.add_argument("--visualizegt", "-vgt", action='store_true',
                        help="Shows the gt generated in the visualization")
    parser.add_argument("--dumpvideo", "-d", action='store_true',
                        help="Dumps the result into a video if -v is selected")
    parser.add_argument("--maxsize", "-ms", type=int, default=16,
                        help="Maximum lateral size of the target")
    parser.add_argument("--maxvel", "-mv", type=int, default=5,
                        help="Maximum velocity of the target")
    parser.add_argument("--output", "-o", type=str, default="output",
                        help="Name of the split")

    args = parser.parse_args()

    if args.visualize:
        visualize(args)
    else:
        main(args)

