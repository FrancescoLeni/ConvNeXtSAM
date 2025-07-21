import sys
import os
import torch
import numpy as np
import math
from torchmetrics.detection import IntersectionOverUnion
from torchmetrics.functional.detection import intersection_over_union
from pathlib import Path
from PIL import Image
from matplotlib import pyplot as plt
import h5py


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
    return y


def get_iou(pred, img_path, return_list=False, save_path='custom_scripts/data', file_name=None, device='cpu',
            lab_path=Path(r'C:\Users\franc\OneDrive - Politecnico di Milano\dataset_mmi\labels\val')):
    """
        Args:
            pred: pred object outbut of model (.pred)
            img_path: path to images (needed to retrieve targets)
            return_list: whether to return a list (def = False => dict)
            save_path: path for saving as .h5 file (ONLY if file_path NOT None)
            file_name: name of file to be saved (if None => not saved)
            device: where to compute metric ('cuda:0' or 'cpu')
            lab_path: path to labels

        Returns:
            iou: dictionary {name: iou_values} (return_list=False)
                 list of ious per image (return_list=True)

    """
    # list of images
    list_file = sorted(os.listdir(img_path))

    preds = []
    t = []

    # creating preds  list of dictionaries (one dict per img)
    for i, p in enumerate(pred):
        preds.append({"boxes": p[:, 0:4].to(device), "labels": p[:, -1].to(torch.uint8).to(device)})

    # creating labels list of dictionaries
    for imgs in list_file:
        lab = Path(imgs).stem + '.txt'
        im = Image.open(img_path / imgs)
        w, h = im.size
        l = []
        c = []
        with open(lab_path / lab, "r") as file:
            for line in file:
                line.strip()
                line = line.split(' ')

                # bringing to absolute coords
                line = [float(x) * y for x, y in zip(line, [1, w, h, w, h])]

                l.append([x for x in line[1:]])
                c.append(int(line[0]))

        t.append({'boxes': xywh2xyxy(torch.tensor(l).to(device)), 'labels': torch.tensor(c).to(device)})

    # finally computing iou (considering the max iou per EVERY predicted box)
    iou = []
    for x, y in zip(preds, t):
        m = intersection_over_union(x['boxes'], y['boxes'], aggregate=False)
        classes = x['labels'].to('cpu').tolist()
        u_classes = np.unique(classes)
        m_list = m.to('cpu').tolist()
        iou_d = {str(c): [] for c in u_classes}
        for i, c in enumerate(classes):
            iou_d[str(c)].append(max(m_list[i]))
        iou.append(iou_d)

    iou_tot = {'0': [], '1': [], '2': []}
    for i in iou:
        for c in i:
            iou_tot[c] += i[c]
    for c in iou_tot:
        iou_tot[c] = [x for x in iou_tot[c] if not math.isnan(x)]

    if file_name:
        if not os.path.isdir(save_path):
            os.mkdir(save_path)
        save_path = Path(save_path) / f'{file_name}.h5'
        print(f'saving {save_path}.h5')
        with h5py.File(save_path, 'w') as file:
            for c in iou_tot:
                file.create_dataset(c, data=iou_tot[c])

    if not return_list:
        return iou
    else:
        iou_list = [iou_tot[x] for x in iou_tot]
        return iou_list


def get_conf(pred, img_path, return_list=False, save_path='custom_scripts/data', file_name=None):
    """

    Args:
        pred: pred object output of model (.pred)
        img_path: path to images
        return_list: whether to return list, default = dict
        save_path: path for saving as .h5 file (ONLY if file_path NOT None)
        file_name: name of file to be saved (if None => not saved)

    Returns:
        conf: dictionary {name: iou_values} (return_list=False)
              list of ious per image (return_list=True)

    """
    confs = []
    list_file = sorted(os.listdir(img_path))

    # creating preds and confs list of dictionaries (one dict per img)
    for i, p in enumerate(pred):
        confs.append({'name': list_file[i], 'conf': p[:, -2].to('cpu').tolist(),
                      'labels': p[:, -1].to(torch.uint8).to('cpu').tolist()})

    conf = {'0': [], '1': [], '2': []}
    for i in confs:
        for c, l in zip(i['conf'], i['labels']):
            conf[str(l)].append(c)

    if file_name:
        if not os.path.isdir(save_path):
            os.mkdir(save_path)
        save_path = Path(save_path) / f'{file_name}.h5'
        print(f'saving {save_path}.h5')
        with h5py.File(save_path, 'w') as file:
            for c in conf:
                file.create_dataset(c, data=conf[c])

    if not return_list:
        return conf
    else:
        conf_list = [conf[x] for x in conf]
        return conf_list


def plot_boxplots(data: list, names: list, save_path, file_name, title):
    """

        Args:
            data: list of data to be plotted (even if single boxplot provide as list)
            names: list of names for data (even if single boxplot provide as list)
            save_path: path to save
            file_name: name of file to be saved WITH prefix
            title: title for boxplot

        Returns:

    """

    dst = Path(save_path) / file_name
    print(f'saving {dst}')
    plt.figure(figsize=(20, 11))
    # plt.imshow(img)
    plt.boxplot(data, vert=True, patch_artist=True, labels=names)
    plt.title(title)
    # plt.text(1.01, 0.97, 'gt = red \npred = green', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5),
    #          transform=plt.gca().transAxes, va='top', ha='left', fontsize=12)
    plt.savefig(dst, dpi=96)

