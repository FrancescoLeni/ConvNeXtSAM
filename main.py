trainings = [
    "segment/train.py --weights ConvNeXt_cholect.pt --cfg models/segment/ConvNext_seg.yaml --data data/mmi_nov24.yaml --hyp data/hyps/segmentation/fine_tuning_head_from_cholect.yaml --epochs 120 --optimizer AdamW --name convnext_finetuning_solotesta_su_mmi --freeze 14 --cos-lr --seed 12345",
    "segment/train.py --weights ConvNeXtSAM_cholect.pt --cfg models/segment/SAM_seg.yaml --data data/mmi_nov24.yaml --hyp data/hyps/segmentation/fine_tuning_head_from_cholect.yaml --epochs 120 --optimizer AdamW --name convnextsam_finetuning_solotesta_su_mmi --freeze 14 --cos-lr --seed 12345",
    "segment/train.py --weights yolov5l_cholect.pt --cfg models/segment/yolov5l-seg.yaml --data data/mmi_nov24.yaml --hyp data/hyps/segmentation/fine_tuning_head_from_cholect.yaml --epochs 120 --optimizer AdamW --name yolov5l_finetuning_solotesta_su_mmi --freeze 10 --cos-lr --seed 12345",
]

for command in trainings:
    exec(f"import os; os.system('python {command}')")