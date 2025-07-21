import subprocess
from pathlib import Path

# w_list1 = ['pesi/ConvNeXtSAM_finetuning.pt',
#            'pesi/yolov5l_finetuning.pt'
#            ]
#
# n_list1 = ['ConvNextSAM_transfer_mmi',
#            'yolov5l_transfer_mmi'
#            ]

w_list2 = ['pesi/ConvNeXtSAM_finetuning.pt',
           'pesi/yolov5l_finetuning.pt'
           ]

n_list2 = ['ConvNextSAM_pretrained_mmi',
           'yolov5l_pretrained_mmi'
           ]

# f_list = ['14', '10']
#
#
# for w, n, f in zip(w_list1, n_list1, f_list):
#     subprocess.run(['python', 'train.py'] + ['--epochs', '100', '--batch', '16', '--img', '640', '--data', 'mmi_2.yaml',
#                                              '--weights', f'{w}', '--hyp', 'transfer.yaml', '--optimizer', 'AdamW',
#                                              '--cos-lr', '--seed', '12345', '--device', 'cuda:0', '--name', f'{n}',
#                                              '--freeze', f'{f}'])

for w, n in zip(w_list2, n_list2):
    subprocess.run(['python', 'train.py'] + ['--epochs', '100', '--batch', '16', '--img', '640', '--data', 'mmi_2.yaml',
                                             '--weights', f'{w}', '--hyp', 'runs/train/ConvNext_pretrained_mmi/hyp.yaml', '--optimizer', 'AdamW',
                                             '--cos-lr', '--seed', '12345', '--device', 'cuda:0', '--name', f'{n}'])
