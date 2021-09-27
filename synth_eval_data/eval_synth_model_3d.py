#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 11:56:56 2020

@author: jwang,vrudnev
"""

import os

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import shutil
import json
import re
import time
from glob import glob
from tqdm import tqdm, trange
import torch
import torch.nn as nn
import cv2


def plot_PCK(fname, errors_list, labels, color_list, title=None, xrange=[0,100], showAUC=True):
    """
    errors_list: list of errors from each method
    labels: what each method should be called.
    """

    #configure plot
    font_size = 28
    # font_size = 14
    lim_min, lim_max = xrange
    x_step = 0.1 #step size for PCK

    #create plot
    fig = plt.figure(figsize=(10, 7))
    plt.xlim((lim_min, lim_max))
    plt.ylim((0., 1.))

    plt.grid()
    axes = fig.axes
    for ax in axes:
        for item in ax.get_xticklabels() + ax.get_yticklabels():
            item.set_fontsize(font_size)

    all_legends = []
    legend_labels = []
    method_lines = []

    for method_id in trange(len(errors_list)):
        color = color_list[method_id]
        errors = errors_list[method_id]
        label = labels[method_id]

        xaxis = np.arange(lim_min, lim_max, x_step)
        errors_sorted = np.sort(errors)
        counts = np.array([errors_sorted[errors_sorted < xval].size for xval in xaxis]) / errors.size
        AUC = (counts * x_step / (lim_max - lim_min)).sum()

        label = labels[method_id]
        if showAUC:
            label += " AUC:%.02f" % (AUC)

        line = plt.plot(xaxis, counts, color=color, zorder=100 if method_id==0 else 1)
        method_lines.append(line[0])
        legend_labels.append(label)

    # legend_location = (.49, 0.01)
    legend_location = 4
    # legend_title = r"$\bf{PCK}$"
    plt.legend(method_lines, legend_labels, loc=legend_location, title_fontsize=font_size - 2,
               prop={'size': font_size - 3})
    # plt.legend(method_lines, legend_labels, loc=legend_location, title=legend_title, title_fontsize=font_size - 2,
    #            prop={'size': font_size - 3})

    plt.xlabel('3D Error (mm)',fontsize=font_size)
    plt.ylabel('3D Root-Aligned PCK',fontsize=font_size)
    # plt.title(title,fontsize=font_size)
    plt.tight_layout()
    plt.savefig(fname)

gt = None
def compute_event_stream_errors(gt_list, pred_list, shift=0.):
    global gt

    assert len(gt_list) == len(pred_list), "Must provide the same number of gt file and predictions"

    errors = []
    for seq_id in range(len(gt_list)):
        seq_errors = []
        gt_file = gt_list[seq_id]
        predictions_file = pred_list[seq_id]

        # LIMIT = 10000
        LIMIT = 1000000
        print('loading')
        tstart = time.time()
        if gt is None:
            gt = np.loadtxt(gt_file, max_rows=LIMIT)
        print('loaded gt')
        pred = np.loadtxt(predictions_file, max_rows=LIMIT)
        print('loaded pred')
        print('loaded in', time.time()-tstart, 's')

        # gt_timestamp = np.arange(len(gt))
        # pred_timestamp = np.arange(len(pred))
        # gt_dict = {gt_timestamp[i]: gt[i,:].reshape(-1,3) for i in range(gt.shape[0])}
        # for i in trange(pred_timestamp.shape[0]):
        for i in trange(len(pred)):
            if i < 100.: #our method need a history of 100 ms to work
                continue

            # gt_values = gt_dict.get(pred_timestamp[i]+shift, None)
            gt_values = gt[i+shift].reshape(-1, 3) if i+shift<len(gt) and i+shift>=0 else None
            if (gt_values is None):
                continue
            else:
                pred_values = pred[i,:].reshape(-1,3)

                target_kpts = [21, 52, 53, 54, 55, 56]
                target_kpts += list(range(37, 52))

                gt_values = gt_values[target_kpts,:]
                pred_values = pred_values[target_kpts,:]

                # subtract base
                gt_values = gt_values[1:]-gt_values[0][None,:]
                pred_values = pred_values[1:]-pred_values[0][None,:]

                all_joint_errors = np.linalg.norm(pred_values - gt_values,axis=-1)
                # from meters to millimeters
                all_joint_errors = all_joint_errors * 1000

                seq_errors.append(all_joint_errors)
        seq_errors = np.concatenate(seq_errors)
        print("seq {} error: {}".format(seq_id, seq_errors.mean()))
        errors.append(seq_errors)
    print("total:{}".format(np.concatenate(errors).mean()))
    return np.concatenate(errors)

if __name__ == "__main__":
    gt_files = ['./synthjoints/synthdemo_joints.txt']


    errors_list = []
    color_list = []
    labels = []

    pr_files = [sys.argv[1]]
    print(pr_files)

    shift = 100
    if 'nolnes100' in model_name:
        shift = 100
    elif 'nolnes30' in model_name:
        shift = 33
    elif 'lnes300' in model_name:
        shift = 300
    elif 'lnes100' in model_name:
        shift = 100
    elif 'lnes30' in model_name:
        shift = 33
    elif 'frame100' in model_name:
        shift = 100
    elif 'frame30' in model_name:
        shift = 33
    elif 'count100' in model_name:
        shift = 100
    elif 'count30' in model_name:
        shift = 33
    else:
        print('Cannot determine the shift from the model name, assuming 100 ms window length')

    print('Using shift:', shift)

    errors = compute_event_stream_errors(gt_files, pr_files, shift)
    errors_list.append(errors)
    color_list.append('C'+str(len(color_list)))
    label = sys.argv[1]

    fname = sys.argv[3]

    plot_PCK(fname, errors_list, labels, color_list, title="Avg. Keypoint Error 3D", xrange=[0, 100], showAUC=True)
