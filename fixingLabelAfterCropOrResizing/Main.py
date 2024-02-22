from functions import Preprocessing
import os
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
from functions import controller


img_folder = r'C:\Users\zafer\OneDrive\Desktop\FinalProcessing\images'
label_folder = r'C:\Users\zafer\OneDrive\Desktop\FinalProcessing\labels'

img_files = os.listdir(img_folder)
label_files = os.listdir(label_folder)
labels = []
sayac= 0
# print(controller('fasdfsa.jpg','fdasf.txt'))
for img, label in tqdm(zip(img_files, label_files)):
    if controller (img,label) == False:
        print(img,label)
        break
    try:
        full_img_path = os.path.join(img_folder,img)
        full_label_path = os.path.join(label_folder,label)
        process  = Preprocessing(img_path=full_img_path,label_path=full_label_path)
        
        x = process.Unnormalize_labels(image_type='orginal_image')

        process.crop_img()
        process.Resize((512,512))
        process.update_bounding_boxes()
        bbox = process.bbox_resize_cropped
        templabel = []
        templabel.append(bbox['x1']),templabel.append(bbox['y1']),templabel.append(bbox['x2']),templabel.append(bbox['y2'])
        labels.append(templabel)

        cv2.rectangle(process.Resized_cropped_img, (bbox['x1'], bbox['y1']), (bbox['x2'], bbox['y2']), (255, 255, 0), 2)

        cv2.imshow('after_P_Processing',process.Resized_cropped_img)
        cv2.imshow('orginal',process.img)
        sayac +=1
        key= cv2.waitKey(1)
        if key ==ord('q'):
            break
    except:
        print('an error')

# print(labels)
cv2.destroyAllWindows()
print(sayac)