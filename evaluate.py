import os
import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

env_path = os.path.join(os.path.dirname(__file__), '..')
if env_path not in sys.path:
    sys.path.append(env_path)

from pytracking.config import local_env_settings

def evaluate(parameter_name, seq_name):
    env = local_env_settings()
    results_path = '{}/atom/{}/{}.txt'.format(env.results_path, parameter_name, seq_name)
    groundtruth_path = '{}/{}/groundtruth_rect.txt'.format(env.otb_path, seq_name)
    try:
        results = np.loadtxt(str(results_path), dtype=np.float64)
    except:
        results = np.loadtxt(str(results_path), delimiter=',', dtype=np.float64)
    try:
        groundtruth = np.loadtxt(str(groundtruth_path), dtype=np.float64)
    except:
        groundtruth = np.loadtxt(str(groundtruth_path), delimiter=',', dtype=np.float64)

    frame_num = np.shape(groundtruth)[0]
    threshold = np.arange(0, 1, 0.05)
    N = np.size(threshold)
    success_rate = np.zeros(np.shape(threshold))

    for f in np.arange(frame_num):
        score = iou(groundtruth[f,:], results[f,:])
        for i in np.arange(N):
            if (score > threshold[i]):
                success_rate[i] += 1
    
    success_rate = success_rate / frame_num
    
    for i in np.arange(N):
        print(threshold[i], success_rate[i])


def iou(gt, bb):
    endx = max(gt[0]+gt[2]/2, bb[0]+bb[2]/2)
    startx = min(gt[0]-gt[2]/2, bb[0]-bb[2]/2)
    width = gt[2] + bb[2] - (endx - startx)

    endy = max(gt[1]+gt[3]/2, bb[1]+bb[3]/2)
    starty = min(gt[1]-gt[3]/2, bb[1]-bb[3]/2)
    height = gt[3] + bb[3] - (endy - starty)

    if width <=0 or height <=0:
       ratio = 0
    else:
        Area = width*height
        Area1 = gt[2]*gt[3]
        Area2 = bb[2]*bb[3]
        ratio = Area / ( Area1 + Area2 -Area)

    return(ratio)



def main():
    parser = argparse.ArgumentParser(description='evaluate success point')
    parser.add_argument('tracker_param', type=str, help='Name of parameter file.')
    parser.add_argument('--dataset', type=str, default='otb', help='Name of dataset (otb, nfs, uav, tpl, vot, tn, gott, gotv, lasot).')
    parser.add_argument('--sequence', type=str, default=None, help='Sequence number or name.')

    args = parser.parse_args()

    evaluate(args.tracker_param,  args.sequence)


if __name__ == '__main__':
    main()

