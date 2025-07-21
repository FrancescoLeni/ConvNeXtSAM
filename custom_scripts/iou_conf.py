import torch
from torchmetrics.detection import IntersectionOverUnion
from pathlib import Path
import sys
import os
from PIL import Image
import cv2 as cv
from matplotlib import pyplot as plt

# Get the current directory of script_1.py
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory (A) of the current directory
parent_dir = os.path.dirname(current_dir)
# Add the parent directory to the Python path
sys.path.append(parent_dir)

from models.common import *
from utils.general import xywh2xyxy

# PUT HERE 'cpu' or 'cuda:0'
device = 'cuda:0'

img_path = Path(r'C:\Users\franc\OneDrive - Politecnico di Milano\dataset_mmi\images\valid')
lab_path = Path(r'C:\Users\franc\OneDrive - Politecnico di Milano\dataset_mmi\labels\val')
fold = Path(r'C:\Users\franc\OneDrive - Politecnico di Milano\ConvNeXtSAM\cluster_runs\train')

# per avere stesso ordine delle pred
list_file = sorted(os.listdir(img_path))

model = torch.hub.load('', 'custom', fold / r"ConvNextSAM_finetune_mmi\weights\best.pt", source='local').to(device)
out = model(img_path)
pred = out.pred

preds = []
t = []
confs = []

# creating preds and confs list of dictionaries (one dict per img)
for i, p in enumerate(pred):
    preds.append({"boxes": p[:, 0:4].to(device), "labels": p[:, -1].to(torch.uint8).to(device)})
    confs.append({'name': list_file[i], 'conf': p[:, -2].to('cpu').tolist(), 'class': p[:, -1].to(torch.uint8).to('cpu').tolist()})

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

metric = IntersectionOverUnion(class_metrics=True).to(device)

iou = []
# m = metric(preds, t)
# metric.reset()
# print(m)

for x, y in zip(preds, t):
    m = metric([x], [y])
    metric.reset()
    iou_d = {}
    if 'iou/cl_0' in m:
        iou_d['0'] = m['iou/cl_0'].to('cpu').tolist()
    if 'iou/cl_1' in m:
        iou_d['1'] = m['iou/cl_1'].to('cpu').tolist()
    if 'iou/cl_2' in m:
        iou_d['2'] = m['iou/cl_2'].to('cpu').tolist()

    iou.append(iou_d)

img = Image.open(img_path / list_file[0])
img = np.asarray(exif_transpose(img))
for box in preds[0]['boxes']:
    pt1 = box[0:2].to(torch.int).to('cpu').tolist()
    pt1 = tuple(pt1)
    pt2 = box[2:4].to(torch.int).to('cpu').tolist()
    pt2 = tuple(pt2)
    img = cv.rectangle(img, pt1, pt2, (0, 255, 0), thickness=2)
for box in t[0]['boxes']:
    pt1 = box[0:2].to(torch.int).to('cpu').tolist()
    pt1 = tuple(pt1)
    pt2 = box[2:4].to(torch.int).to('cpu').tolist()
    pt2 = tuple(pt2)
    img = cv.rectangle(img, pt1, pt2, (255, 0, 0), thickness=2)

plt.figure(figsize=(20, 11))
plt.imshow(img)
plt.title('culo')
plt.text(1.01, 0.97, 'gt = red \npred = green', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5),
         transform=plt.gca().transAxes, va='top', ha='left', fontsize=12)
plt.savefig('runs/prova.png', dpi=96)


