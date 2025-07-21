import sys
import os

# Get the current directory of script_1.py
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory (A) of the current directory
parent_dir = os.path.dirname(current_dir)
# Add the parent directory to the Python path
sys.path.append(parent_dir)

from pathlib import Path
import torch

from functions import get_conf, get_iou, plot_boxplots

# PUT HERE 'cpu' or 'cuda:0'
device = 'cuda:0'

img_path = Path(r'C:\Users\franc\OneDrive - Politecnico di Milano\dataset_mmi\images\valid')
lab_path = Path(r'C:\Users\franc\OneDrive - Politecnico di Milano\dataset_mmi\labels\val')
fold = Path(r'C:\Users\franc\OneDrive - Politecnico di Milano\ConvNeXtSAM\cluster_runs\train')

for models in os.listdir(fold):
    if "pretrained" in models:
        if not os.path.isdir(fr'C:\Users\franc\OneDrive - Politecnico di Milano\ConvNeXtSAM\data\{models}'):
            model = torch.hub.load('', 'custom', fold / fr"{models}\weights\best.pt", source='local').to(device)
            out = model(img_path)
            pred = out.pred

            iou = get_iou(pred, img_path, True, save_path=fr'C:\Users\franc\OneDrive - Politecnico di Milano\ConvNeXtSAM\data\{models}',
                          file_name='IoU')
            names = ['shaft', 'wrist', 'tip']
            conf = get_conf(pred, img_path, True, save_path=fr'C:\Users\franc\OneDrive - Politecnico di Milano\ConvNeXtSAM\data\{models}',
                            file_name='conf')

            plot_boxplots(conf, names, fr'C:\Users\franc\OneDrive - Politecnico di Milano\ConvNeXtSAM\data\{models}',
                          f'conf.png', 'conf')
            plot_boxplots(iou, names, fr'C:\Users\franc\OneDrive - Politecnico di Milano\ConvNeXtSAM\data\{models}',
                          f'IoU.png', 'IoU')






