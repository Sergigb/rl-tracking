import argparse
import os

import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.patches as pat
import cv2

x_max = 48
x_min = 16

bbox_marg = 2


def iou(bbox1, bbox2):
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])

    intersection = abs(max((x2 - x1, 0)) * max((y2 - y1), 0))

    bbox1area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    bbox2area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])

    iou_score = intersection / float(bbox1area + bbox2area - intersection)
    return iou_score


# actions: [disp -x, disp +x, disp -y, disp + y, size +, size -, none]
# displacements are of 2 pxs
def pick_action(bbox_gt, bbox_noise):

    x, y, w, h = bbox_gt
    bbox_gt_t = [x - int(w/2), y - int(h/2), x + int(w/2),  y + int(h/2)]

    x, y, w, h = bbox_noise
    bbox_noise_t = [x - int(w/2), y - int(h/2), x + int(w/2),  y + int(h/2)]

    x1, y1, x2, y2 = bbox_noise_t

    best_action = 6
    max_score = iou(bbox_gt_t, bbox_noise_t)
    # 1 -x
    score_action = iou(bbox_gt_t, [x1 - 2, y1, x2 - 2, y2])
    if score_action > max_score:
        max_score = score_action
        best_action = 0
    # 2 +x
    score_action = iou(bbox_gt_t, [x1 + 2, y1, x2 + 2, y2])
    if score_action > max_score:
        max_score = score_action
        best_action = 1
    # 3 -y
    score_action = iou(bbox_gt_t, [x1, y1 - 2, x2, y2 - 2])
    if score_action > max_score:
        max_score = score_action
        best_action = 2
    # 4 +y
    score_action = iou(bbox_gt_t, [x1, y1 + 2, x2, y2 + 2])
    if score_action > max_score:
        max_score = score_action
        best_action = 3
    # 5 +s
    score_action = iou(bbox_gt_t, [x1 - 2, y1 - 2, x2 + 2, y2 + 2])
    if score_action > max_score:
        max_score = score_action
        best_action = 4
    # 6 -s
    score_action = iou(bbox_gt_t, [x1 + 2, y1 + 2, x2 - 2, y2 - 2])
    if score_action > max_score:
        max_score = score_action
        best_action = 5

    return best_action



def main(args):
    patches = np.zeros((5000, 64, 64))
    actions = np.zeros((5000, 1))
    
    for i in range(patches.shape[0]):
        im = np.zeros((256, 256))

        origin = 128

        w = 16 + int(abs(min(random.gauss(0, 16), x_max)))
        h = 16 + int(abs(min(random.gauss(0, 16), x_max)))

        bbox_gt = [origin, origin, w * 2 + bbox_marg * 2, h * 2 + bbox_marg * 2]

        im[origin - h: origin + h, origin - w:origin + w] = 1

        # put this in a function
        if random.random() < 1/3.:
            origin_x = origin + min(int(random.gauss(0, 16)), w)
            origin_y = origin + min(int(random.gauss(0, 16)), h)
            w_d = w
            h_d = h
        elif random.random() < 1/4.:
            origin_x = origin
            origin_y = origin
            d = min(int(random.gauss(0, 16)), 32)
            if abs(d) < 3:
                d = random.choice([3, -3])
            w_d = w + d
            h_d = h + d
            if w_d < 2 or h_d < 2:
                w_d = 2
                h_d = 2
        else:
            origin_x = origin + min(int(random.gauss(0, 16) + bbox_marg * 2 * random.choice([1, -1])), w)
            origin_y = origin + min(int(random.gauss(0, 16) + bbox_marg * 2 * random.choice([1, -1])), h)
            d = min(int(random.gauss(0, 16)), 32)
            if abs(d) < 3:
                d = random.choice([3, -3])
            w_d = w + d
            h_d = h + d
            if w_d < 2 or h_d < 2:
                w_d = 2
                h_d = 2

        bbox_noise = [origin_x, origin_y, w_d * 2 + bbox_marg, h_d * 2 + bbox_marg]

        actions[i] = pick_action(bbox_gt, bbox_noise)
        patch = im[int(bbox_noise[1] - bbox_noise[3] / 2):int(bbox_noise[1] + bbox_noise[3] / 2),
                   int(bbox_noise[0] - bbox_noise[2] / 2):int(bbox_noise[0] + bbox_noise[2] / 2)]

        patches[i] = cv2.resize(patch, (64, 64), interpolation = cv2.INTER_NEAREST)

        if args.v:
            if(actions[i] == 0):
                print("go left")
            if(actions[i] == 1):
                print("go right")
            if(actions[i] == 2):
                print("go up")
            if(actions[i] == 3):
                print("go down")
            if(actions[i] == 4):
                print("grow")
            if(actions[i] == 5):
                print("shrink")
            if(actions[i] == 6):
                print("do nothing")

            fig2, ax2 = plt.subplots(1)
            ax2.imshow(patches[i])

            fig, ax = plt.subplots(1)
            ax.imshow(im)

            rect = pat.Rectangle((bbox_gt[0] - bbox_gt[2] / 2, bbox_gt[1] - bbox_gt[3] / 2), bbox_gt[2], bbox_gt[3], linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            rect = pat.Rectangle((bbox_noise[0] - bbox_noise[2] / 2, bbox_noise[1] - bbox_noise[3] / 2), bbox_noise[2], bbox_noise[3], linewidth=1, edgecolor='g', facecolor='none')
            ax.add_patch(rect)

            plt.show()

    if not os.path.exists("../datasets"):
        os.mkdir("../datasets")
        os.mkdir("../datasets/simple")
    if not os.path.exists("../datasets/simple"):
        os.mkdir("../datasets/simple")

    np.save("../datasets/simple/" + args.output + "_supervised.npy", patches)
    np.save("../datasets/simple/" + args.output + "_supervised_gt.npy", actions)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", action="store_true",
                        help="Visualize the generated patches")
    parser.add_argument("--output", "-o", type=str, default="train",
                        help="Name of the split")
    parser.add_argument("--number", "-n", type=int, default=5000,
                        help="Number of images generated")
    args = parser.parse_args()

    main(args)


