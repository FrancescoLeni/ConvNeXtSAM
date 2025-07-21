import sys
import os
import h5py

from pathlib import Path

import matplotlib.pyplot as plt
import torch
from functions import plot_boxplots

parent = Path(r'C:\Users\franc\OneDrive - Politecnico di Milano\ConvNeXtSAM\data')

iou = {}
conf = {}

for names in os.listdir(parent):
    if os.path.isdir(parent / names):
        with h5py.File(parent / f'{names}/IoU.h5', 'r') as file:
            iou[names] = {key: list(file[key][:]) for key in file.keys()}
for names in os.listdir(parent):
    if os.path.isdir(parent / names):
        with h5py.File(parent / f'{names}/conf.h5', 'r') as file:
            conf[names] = {key: list(file[key][:]) for key in file.keys()}

finetune = {'iou': {key: [iou[c][key] for c in iou if 'finetune' in c] for key in ['0', '1', '2']},
            'conf': {key: [conf[c][key] for c in conf if 'finetune' in c] for key in ['0', '1', '2']},
            'names': [c for c in iou if 'finetune' in c]}
pretrained = {'iou': {key: [iou[c][key] for c in iou if 'pretrained' in c] for key in ['0', '1', '2']},
              'conf': {key: [conf[c][key] for c in conf if 'pretrained' in c] for key in ['0', '1', '2']},
              'names': [c for c in conf if 'pretrained' in c]}


for n in ['iou', 'conf']:
    dst = Path(parent) / f'finetune_{n}.png'

    fig, axs = plt.subplots(3, 1, figsize=(20, 11))
    axs[0].boxplot(finetune[n]['0'], vert=True, patch_artist=True, labels=finetune['names'])
    axs[0].set_title('shaft')
    axs[1].boxplot(finetune[n]['1'], vert=True, patch_artist=True, labels=finetune['names'])
    axs[1].set_title('wrist')
    axs[2].boxplot(finetune[n]['2'], vert=True, patch_artist=True, labels=finetune['names'])
    axs[2].set_title('tip')
    plt.tight_layout()
    plt.savefig(dst, dpi=96)

    dst = Path(parent) / f'pretrained_{n}.png'

    fig, axs = plt.subplots(3, 1, figsize=(20, 11))
    axs[0].boxplot(pretrained[n]['0'], vert=True, patch_artist=True, labels=pretrained['names'])
    axs[0].set_title('shaft')
    axs[1].boxplot(pretrained[n]['1'], vert=True, patch_artist=True, labels=pretrained['names'])
    axs[1].set_title('wrist')
    axs[2].boxplot(pretrained[n]['2'], vert=True, patch_artist=True, labels=pretrained['names'])
    axs[2].set_title('tip')
    plt.tight_layout()
    plt.savefig(dst, dpi=96)
