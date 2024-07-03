import cv2
import pandas as pd
import cv2
import pandas as pd
import matplotlib.pyplot as plt


class Preprocessing:
    def __init__(self, img_path,label_path):
        self.img_path = img_path
        self.img = cv2.imread(img_path)
        self.label = pd.read_csv(label_path)
        self.contors = self.label.columns[0].split(' ')

        self.orginal_image_center_x_normalized, self.orginal_image_center_y_normalized, self.orginal_image_width_normalized, self.orginal_image_height_normalized = float(self.contors[1]),float(self.contors[2]),float(self.contors[3]),float(self.contors[4])
        
        self.orginal_image_height = self.img.shape[0]
        self.orginal_image_width = self.img.shape[1]

        # self.orginal_image_x1, self.orginal_image_y1, self. orginal_image_x2, self.orginal_image_y2 = None,None,None,None
        self.Unnormalized_cords =None
        self.bounding_boxes_cropped_image = None
        self.cropped_img = None
        self.Resized_cropped_img =None
        self.bbox_resize_cropped = None

        self.Resized_cropped_normalized_labels= None
        
    def Unnormalize_labels (self,image_type):
        if image_type=='orginal_image':
            center_x_unnormalized = int(self.orginal_image_center_x_normalized * self.orginal_image_width)
            center_y_unnormalized = int(self.orginal_image_center_y_normalized * self.orginal_image_height)
            width_unnormalized = int(self.orginal_image_width_normalized * self.orginal_image_width)
            height_unnormalized = int(self.orginal_image_height_normalized * self.orginal_image_height)
            self.Unnormalized_cords = {
            'x1' : center_x_unnormalized - width_unnormalized // 2,
            'y1' : center_y_unnormalized - height_unnormalized // 2,
            'x2' : center_x_unnormalized + width_unnormalized // 2,
            'y2' : center_y_unnormalized + height_unnormalized // 2
            }
    def Normalize_labels (self,image_type):
        if image_type=='resized_cropped_image':
            center_x_normalized_again = self.bbox_resize_cropped['x1'] / self.Resized_cropped_img.shape[1]
            center_y_normalized_again = self.bbox_resize_cropped['y1'] / self.Resized_cropped_img.shape[0]
            width_normalized_again = self.bbox_resize_cropped['x2'] / self.Resized_cropped_img.shape[1]
            height_normalized_again = self.bbox_resize_cropped['y2'] / self.Resized_cropped_img.shape[0]
            self.Resized_cropped_normalized_labels = {
                "center_x_normalized_again": center_x_normalized_again,
                "center_y_normalized_again": center_y_normalized_again,
                "width_normalized_again": width_normalized_again,
                "height_normalized_again": height_normalized_again
            }
        return self.Resized_cropped_normalized_labels

    def crop_img(self):
        img = self.img
        gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        hrch, binary_image = cv2.threshold(gray_img, 10, 255, cv2.THRESH_BINARY)
        contours,hrch = cv2.findContours(binary_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        crop_details = {
            'top': y,
            'left': x,
            'right': img.shape[1] - (x + w),
            'bottom': img.shape[0] - (y + h)
        }
        #print(crop_details)
        self.crop_details = crop_details
        n_cropped_img = img[y:y+h, x:x+w]
        #resized = cv2.resize(cropped_img, (512, 512))
        self.cropped_img = n_cropped_img
    def Resize(self,Size):
        self.Resized_cropped_img = cv2.resize(self.cropped_img,(Size))

    def update_bounding_boxes(self):
        
        self.bounding_boxes_cropped_image = {    
        'x1' : self.Unnormalized_cords['x1'] - self.crop_details['left'],
        'y1' : self.Unnormalized_cords['y1'] - self.crop_details['top'],
        'x2' : self.Unnormalized_cords['x2'] - self.crop_details['left'],
        'y2' : self.Unnormalized_cords['y2'] - self.crop_details['top']
        }
        # print('hello')
        # print(self.Resized_cropped_img.shape)


        # # Adjust bounding box coordinates
        if self.Resized_cropped_img is not None:
            resized_shape = self.Resized_cropped_img.shape
            original_shape = self.cropped_img.shape

            # # # Bounding box coordinates before resizing
            bbox_before_resizing = ((self.bounding_boxes_cropped_image['x1'], self.bounding_boxes_cropped_image['y1']), (self.bounding_boxes_cropped_image['x2'], self.bounding_boxes_cropped_image['y2']))

        # # # Calculate scaling factors for width and height
            width_scale = resized_shape[1] / original_shape[1]
            height_scale = resized_shape[0] / original_shape[0]
            self.bbox_resize_cropped = {
                'x1':int(bbox_before_resizing[0][0] * width_scale), 'y1':int(bbox_before_resizing[0][1] * height_scale),
                'x2':int(bbox_before_resizing[1][0] * width_scale), 'y2':int(bbox_before_resizing[1][1] * height_scale)
            }
            # print(self.bbox_resize_cropped)

        # return  adjusted_bbox
    
def controller (imgpath,labelpath):
    labelpath= labelpath[0:-4]
    imgpath = imgpath[0:-4]
    if labelpath == imgpath:
        return True
    else:
        return False








def crop_img(img_path):
    img = cv2.imread(img_path)
    gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    hrch, binary_image = cv2.threshold(gray_img, 10, 255, cv2.THRESH_BINARY)
    contours,hrch = cv2.findContours(binary_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    crop_details = {
        'top': y,
        'left': x,
        'right': img.shape[1] - (x + w),
        'bottom': img.shape[0] - (y + h)
    }
    #print(crop_details)
    cropped_img = img[y:y+h, x:x+w]
    #resized = cv2.resize(cropped_img, (512, 512))
    return cropped_img,crop_details


def update_bounding_boxes_for_cropped_image(cords,x1,y1,x2,y2):
    bounding_boxes = {    
    'new_x1' : x1 - cords['left'],
    'new_y1' : y1 - cords['top'],
    'new_x2' : x2 - cords['left'],
    'new_y2' : y2 - cords['top']
    }
    return bounding_boxes


def update_bounding_boxes_for_resized_image(original_image,resized_image,cropped_x1_bbox,cropped_y1_bbox,cropped_x2_bbox,cropped_y2_bbox):


    # Original image shape
    original_shape = original_image.shape

    # Resized image shape
    resized_shape = resized_image.shape

    # Bounding box coordinates before resizing
    bbox_before_resizing = ((cropped_x1_bbox, cropped_y1_bbox), (cropped_x2_bbox, cropped_y2_bbox))

    # Calculate scaling factors for width and height
    width_scale = resized_shape[1] / original_shape[1]
    height_scale = resized_shape[0] / original_shape[0]

    # Adjust bounding box coordinates
    adjusted_bbox = {
        'x1':int(bbox_before_resizing[0][0] * width_scale), 'y1':int(bbox_before_resizing[0][1] * height_scale),
        'x2':int(bbox_before_resizing[1][0] * width_scale), 'y2':int(bbox_before_resizing[1][1] * height_scale)
    }

    return  adjusted_bbox

























''''''''''''''''''''''''''''
Updated
'''''''''''''''''''''''''''''''''
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import os
inputstring = "-902.946,399.9102;-902.946,83.00023;-1175.955,83.00023;-1175.955,399.9102"
# Her koordinat çifti noktalı virgül ile ayrılmıştır

def conf_the_str(input_coords):
    pairs = input_coords.split(';')
    # Koordinat çiftlerini ayrıştırıp, listeye ekleyelim
    coordinates = []
    for pair in pairs:
        x, y = map(float, pair.split(','))
        coordinates.append([x, y])
    return coordinates


def label_conf_and_draw(labels,image):
    '''Output order: x1,y1,x2,y2'''
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Get image dimensions
    height, width, _ = image.shape

    # Given coordinates of the label corners
    label_coords = labels 

    # Calculate actual positions without loop
                    #x              + half_width        y+ half_height
    x1, y1 = int(label_coords[0][0] + width / 2), int(label_coords[0][1] + height / 2)
    x2, y2 = int(label_coords[2][0] + width / 2), int(label_coords[2][1] + height / 2)

    return x1,y1,x2,y2

def convert_to_yolo(x1, y1, x2, y2, image):
    'return x_center, y_center, width, height'
    # Calculate the center of the bounding box
    img_height, img_width, _ = image.shape
    x_center = (x1 + x2) / 2.0
    y_center = (y1 + y2) / 2.0
         
    # Calculate the width and height of the bounding box
    width = x2 - x1
    height = y2 - y1
    
    # Normalize the values by the image width and height
    x_center /= img_width
    y_center /= img_height
    width /= img_width
    height /= img_height
    
    return x_center, y_center, width, height



def unnormalize_Yolo_cords(label,image):
    'return: x1, y1, x2, y2'
    img_height, img_width, rgb = image.shape

    cls,x_zip, y_zip, genislik_bbox_zip, yukseklik_bbox_zip = label

    img_height, img_width, rgb = image.shape #shape fonksyonu Goruntunun gensligi, yuksekligi ve kanal sayisini alir
    x = x_zip * img_width
    y = y_zip * img_height
    bbox_genislik= genislik_bbox_zip * img_width
    bbox_yukseklik = yukseklik_bbox_zip * img_height

    x1 = int(x - bbox_genislik/2)
    x2 = int(x + bbox_genislik/2)
    y1 = int(y - bbox_yukseklik/2)
    y2 = int(y + bbox_yukseklik/2)
    # print(f'x1:{x1} ,x2:{x2},y1:{y1},y2:{y2}')
    # cv2.putText(image,class_name,(x1+5,y1+50),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),2)

    return x1, y1, x2, y2


def update_yolo_after_cropped(crop_labels,normalized_cancer_label,orginalimage,crop_image):
    '''2D crop_labels, 1Dc cancer_label, imageMatrix
    return classID,new_x_center, new_y_center, new_width, new_height'''
    


    crop_cords = crop_labels
    # original_image= image
    crop_x1,crop_y1,crop_x2,crop_y2 = unnormalize_Yolo_cords(label=crop_cords[0], image= orginalimage)
    # cancer_labels = read_txt(txt_konumu=label_path)
    Unnormalized_cords_x1,Unnormalized_cords_y1,Unnormalized_cords_x2,Unnormalized_cords_y2 = unnormalize_Yolo_cords(label=normalized_cancer_label,image=orginalimage)
    crop_details = {
        'left':crop_x1,# x1
        'top':crop_y1,# y1
        'right':crop_x2,# x2
        'bottom':crop_y2,# y2
        }

    new_x1 = Unnormalized_cords_x1 - crop_details['left']
    new_y1 = Unnormalized_cords_y1 - crop_details['top']
    new_x2 = Unnormalized_cords_x2 - crop_details['left']
    new_y2 = Unnormalized_cords_y2 - crop_details['top']
    classID = normalized_cancer_label[0]
    new_x_center, new_y_center, new_width, new_height = convert_to_yolo (x1 = new_x1 ,y1=new_y1,x2=new_x2 ,y2= new_y2,image=crop_image)
    # x_center, y_center, width, height = convert_to_yolo(x1,y1,x2,y2,crop_image)
    return classID,new_x_center, new_y_center, new_width, new_height





def Predict_frame(frame,model):
    'labels[[x, y, w,h]]'
    predictions = model(frame, save_txt=None)
    labels = []
    for idx, prediction in enumerate(predictions[0].boxes.xywhn):  # Change final attribute to desired box format
        cls = int(predictions[0].boxes.cls[idx].item())
        # Write line to file in YOLO label format : cls x y w h
        line = f"{cls} {prediction[0].item()} {prediction[1].item()} {prediction[2].item()} {prediction[3].item()}"
        numbers = list(map(float, line.split()))
        labels.append(numbers)
    return labels

def read_txt (txt_konumu):
    labels = []
    with open(txt_konumu, 'r') as file:
        # Read the entire file
        for line in file:
            numbers  = list(map(float, line.split()))
            labels.append(numbers)
    return labels






print('runned')



import cv2
import matplotlib.pyplot as plt
import pandas as pd
import os
inputstring = "-902.946,399.9102;-902.946,83.00023;-1175.955,83.00023;-1175.955,399.9102"
# Her koordinat çifti noktalı virgül ile ayrılmıştır

def conf_the_str(input_coords):
    pairs = input_coords.split(';')
    # Koordinat çiftlerini ayrıştırıp, listeye ekleyelim
    coordinates = []
    for pair in pairs:
        x, y = map(float, pair.split(','))
        coordinates.append([x, y])
    return coordinates


def label_conf_and_draw(labels,image):
    '''Output order: x1,y1,x2,y2'''
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Get image dimensions
    height, width, _ = image.shape

    # Given coordinates of the label corners
    label_coords = labels 

    # Calculate actual positions without loop
                    #x              + half_width        y+ half_height
    x1, y1 = int(label_coords[0][0] + width / 2), int(label_coords[0][1] + height / 2)
    x2, y2 = int(label_coords[2][0] + width / 2), int(label_coords[2][1] + height / 2)

    return x1,y1,x2,y2

def convert_to_yolo(x1, y1, x2, y2, image):
    'return x_center, y_center, width, height'
    # Calculate the center of the bounding box
    img_height, img_width, _ = image.shape
    x_center = (x1 + x2) / 2.0
    y_center = (y1 + y2) / 2.0
         
    # Calculate the width and height of the bounding box
    width = x2 - x1
    height = y2 - y1
    
    # Normalize the values by the image width and height
    x_center /= img_width
    y_center /= img_height
    width /= img_width
    height /= img_height
    
    return x_center, y_center, width, height



def unnormalize_Yolo_cords(label,image):
    'return: x1, y1, x2, y2'
    img_height, img_width, rgb = image.shape

    cls,x_zip, y_zip, genislik_bbox_zip, yukseklik_bbox_zip = label

    img_height, img_width, rgb = image.shape #shape fonksyonu Goruntunun gensligi, yuksekligi ve kanal sayisini alir
    x = x_zip * img_width
    y = y_zip * img_height
    bbox_genislik= genislik_bbox_zip * img_width
    bbox_yukseklik = yukseklik_bbox_zip * img_height

    x1 = int(x - bbox_genislik/2)
    x2 = int(x + bbox_genislik/2)
    y1 = int(y - bbox_yukseklik/2)
    y2 = int(y + bbox_yukseklik/2)
    # print(f'x1:{x1} ,x2:{x2},y1:{y1},y2:{y2}')
    # cv2.putText(image,class_name,(x1+5,y1+50),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),2)

    return x1, y1, x2, y2


def update_yolo_after_cropped(crop_labels,normalized_cancer_label,orginalimage,crop_image):
    '''2D crop_labels, 1Dc cancer_label, imageMatrix
    return classID,new_x_center, new_y_center, new_width, new_height'''
    


    crop_cords = crop_labels
    # original_image= image
    crop_x1,crop_y1,crop_x2,crop_y2 = unnormalize_Yolo_cords(label=crop_cords[0], image= orginalimage)
    # cancer_labels = read_txt(txt_konumu=label_path)
    Unnormalized_cords_x1,Unnormalized_cords_y1,Unnormalized_cords_x2,Unnormalized_cords_y2 = unnormalize_Yolo_cords(label=normalized_cancer_label,image=orginalimage)
    crop_details = {
        'left':crop_x1,# x1
        'top':crop_y1,# y1
        'right':crop_x2,# x2
        'bottom':crop_y2,# y2
        }

    new_x1 = Unnormalized_cords_x1 - crop_details['left']
    new_y1 = Unnormalized_cords_y1 - crop_details['top']
    new_x2 = Unnormalized_cords_x2 - crop_details['left']
    new_y2 = Unnormalized_cords_y2 - crop_details['top']
    classID = normalized_cancer_label[0]
    new_x_center, new_y_center, new_width, new_height = convert_to_yolo (x1 = new_x1 ,y1=new_y1,x2=new_x2 ,y2= new_y2,image=crop_image)
    # x_center, y_center, width, height = convert_to_yolo(x1,y1,x2,y2,crop_image)
    return classID,new_x_center, new_y_center, new_width, new_height





def Predict_frame(frame,model):
    'labels[[x, y, w,h]]'
    predictions = model(frame, save_txt=None)
    labels = []
    for idx, prediction in enumerate(predictions[0].boxes.xywhn):  # Change final attribute to desired box format
        cls = int(predictions[0].boxes.cls[idx].item())
        # Write line to file in YOLO label format : cls x y w h
        line = f"{cls} {prediction[0].item()} {prediction[1].item()} {prediction[2].item()} {prediction[3].item()}"
        numbers = list(map(float, line.split()))
        labels.append(numbers)
    return labels

def read_txt (txt_konumu):
    labels = []
    with open(txt_konumu, 'r') as file:
        # Read the entire file
        for line in file:
            numbers  = list(map(float, line.split()))
            labels.append(numbers)
    return labels






print('runned')









































