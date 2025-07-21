import os
import torch
from PIL import Image
from utils.metrics import box_iou
from utils.general import xywh2xyxy



# model load
model = torch.hub.load('', 'custom', r"C:\Users\User\dopotesi\ConvNeXtSAM\runs_cluster\train\ConvNextSAM_finetune_cholect\weights\best.pt", source='local')

#paths
img_path = r"C:\Users\User\dopotesi\dataset\images\valid"
label_path = r"C:\Users\User\dopotesi\ConvNeXtSAM\datasets\dataset_mmi_v2_new\labels\val"
# Percorso del file in cui scrivere
cartella = r"C:\Users\User\dopotesi\boxplots_iou\ConvNextSAM_finetune_cholect"


out = model(img_path)

#detections
det_vect = out.pred

#itero sulle predizioni
boxplot_data=[]
confidence=[]
fals_neg=0
fals_pos=0

for i, img in enumerate(os.listdir(img_path)):
    im_path = os.path.join(img_path,img)
    lab_path = os.path.join(label_path,img.replace('png','txt'))

    # Apre l'immagine
    image = Image.open(im_path)

    # Ottiene le dimensioni dell'immagine
    w, h = image.size

    # porto le coordinate tra 0 e 1
    det = det_vect[i].clone()
    det[..., 0] = det[..., 0] / w
    det[..., 2] = det[..., 2] / w
    det[..., 1] = det[..., 1] / h
    det[..., 3] = det[..., 3] / h
    class_pred = det[..., 5].tolist()
    conf = det[..., 4].tolist()

    print(det)

    #tiro fuori le labels
    values=[]
    classes=[]
    with open(lab_path, 'r') as file:
        for line in file:
            line_values = [float(val) for val in line.split()[1:]]
            classes.append(float(line.split()[0]))
            values.append(line_values)

    # Converte la lista in un tensore PyTorch
    label_tens = torch.tensor(values)
    label_tens = xywh2xyxy(label_tens)
    print(label_tens)

    #calcolo la matrice delle iou tra le predizioni e labels
    iou = box_iou(det[:, :3], label_tens)
    print(iou)

    '''per ogni classe della pred devo prendere la iou con la stessa classe,
    poi prendo i più valori più alti. quando ho finito per tutte le classi faccio la 
    differenza tra i cosi predetti e quelli veri e aggiungo tot zeri alla 
    lista delle iou'''

    max_iou=[]
    elementi_condivisi = list(set([elemento for elemento in class_pred if elemento in classes]))
    iou_values = []
    for value in elementi_condivisi:
        # conto quante volte il valore appare nelle pred e nelle labels
        value_pred = class_pred.count(value)
        value_label = classes.count(value)

        #se ci sono più labels che predizioni
        if value_label >= value_pred:
            # prendo le righe corrispondenti alla classe giusta (value)
            indices_pred = [j for j, x in enumerate(class_pred) if x == value]
            indices_label = [j for j, x in enumerate(classes) if x == value]

            # prendo solo le righe e colonne corrispondenti alla stessa classe (value)
            iou_copy = iou[indices_pred][:,indices_label]

            # per ogni riga prendo il massimo e elimino la riga
            for n in range(len(indices_pred)):
                iou_values.append(torch.max(iou_copy).item())
                row_to_remove=torch.argmax(iou_copy) // iou_copy.size(1)
                iou_copy = torch.cat((iou_copy[:row_to_remove],
                                      iou_copy[row_to_remove + 1:]), dim=0)

            # aggiungo gli zeri se value labels > value preds
            for n in range(value_label-value_pred):
                iou_values.append(0)

        # se invece ci sono più predizioni che labels (per quella classe)
        if value_pred > value_label:
            indices_pred = [j for j, x in enumerate(class_pred) if x == value]
            indices_label = [j for j, x in enumerate(classes) if x == value]

            # prendo solo le righe e colonne corrispondenti alla stessa classe (value)
            iou_copy = iou[indices_pred][:, indices_label]

            # per ogni colonna prendo il massimo e elimino la riga
            for n in range(len(indices_label)):
                iou_values.append(torch.max(iou_copy).item())
                col_to_remove = torch.argmax(iou_copy) % iou_copy.size(1)
                iou_copy = torch.cat((iou_copy[:, :col_to_remove],
                                      iou_copy[:, col_to_remove + 1:]), dim=1)

            # aggiungo gli zeri se value labels > value preds
            for n in range(value_pred - value_label):
                iou_values.append(0)


    # aggiungo zeri anche se una classe non è stata predetta dall'uno e dall'altro si
    for value in classes:
        if value not in elementi_condivisi:
            iou_values.append(0)

    for value in class_pred:
        if value not in elementi_condivisi:
            iou_values.append(0)

# Apri il file in modalità scrittura
with open(os.path.join(cartella, 'boxpl_data.txt'), 'w') as file:
    # Scrivi ogni valore della lista su una nuova riga
    for value in iou_values:
        file.write(str(value) + '\n')

'''
    for k,cl in enumerate(class_pred):
        indices = [j for j, x in enumerate(classes) if x == cl]
        iou_values=[]
        for j in indices:
            iou_values.append(iou[int(k),int(j)].item())
        max_iou.append(max(iou_values))
        iou[..., int(indices[iou_values.index(max(iou_values))])] = 0

    # calcolo i falsi positivi e negativi e aggiungo uno 0 nel calcolo per ogni falso
    false_positives=0
    false_negatives=0
    if len(class_pred)-len(classes)>0:
        # falsi positivi
        false_positives=false_positives+len(class_pred)-len(classes)
    elif len(class_pred)-len(classes)<0:
        # falsi negativi
        false_negatives=false_negatives-len(class_pred)+len(classes)
    for false in range(false_positives+false_negatives):
        max_iou.append(0)

    #salvo i vettori finali
    boxplot_data.extend(max_iou)
    fals_neg=fals_neg+false_negatives
    fals_pos=fals_pos+false_positives
    confidence.extend(conf)

# Apri il file in modalità scrittura
with open(os.path.join(cartella, 'boxpl_data.txt'), 'w') as file:
    # Scrivi ogni valore della lista su una nuova riga
    for value in boxplot_data:
        file.write(str(value) + '\n')

with open(os.path.join(cartella, 'confidence.txt'), 'w') as file:
    # Scrivi ogni valore della lista su una nuova riga
    for value in confidence:
        file.write(str(value) + '\n')

with open(os.path.join(cartella, 'falsi.txt'), 'w') as file:
    # Scrivi ogni valore della lista su una nuova riga
    for value in (fals_pos,fals_neg):
        file.write(str(value) + '\n')
'''