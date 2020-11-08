# User/Nc717

# First error was here -> Common functionalities
import pandas as pd 
import os
from os import listdir, walk
from os.path import join
import numpy as np
import keras
import math
from pathlib import Path
import pandas as pd 
import tensorflow as tf
import PIL.Image
import PIL.ImageOps
from tqdm import tqdm, tqdm_notebook
import matplotlib.pyplot as plt
print("All common packages loaded")
# DL libraries
import keras
import keras.backend as K
import tensorflow as tf
from keras.models import load_model
import cv2
from efficientnet.keras import EfficientNetB3, EfficientNetB4
#Reproducible results
np.random.seed(17)
tf.random.set_seed(17)
seed_value = 17
os.environ['PYTHONHASHSEED']=str(seed_value)


print("DL libraries loaded")
#Keras-Retinanet modules
from keras_retinanet.models import backbone, convert_model
from keras_retinanet.utils.config import read_config_file, parse_anchor_parameters
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
print("keras-retinanet packages loaded, moving towards convert model module where we got an error")
from keras_retinanet.bin.train import create_models

#set WORKING DIRECTORY
os.chdir('/usr/local/bin/src/')
os.environ['CUDA_VISIBLE_DEVICES'] = '0'



# ============= Inference model for object detection ====================

def get_joint_detection_model(model_path, model_type):
    """
    Input -> Model path for the object detection model
            Model type-> Foot or Hand
    Output -> Inference model for getting the predictions on test images
    
    """
    #config_file_path = '/usr/local/bin/config'
    if model_type == 'Foot_detection':

        model, training_model, prediction_model = create_models(
        backbone_retinanet=backbone('resnet50').retinanet,
        num_classes=5,
        weights=None,
        multi_gpu=False,
        freeze_backbone=True,
        lr=1e-3,
        config=read_config_file('/usr/local/bin/config/config_foot.ini'))

        training_model.load_weights(model_path)
        infer_model = convert_model(training_model, anchor_params = parse_anchor_parameters(read_config_file('/usr/local/bin/config/config_foot.ini')))

    elif model_type == 'Hand_detection':

        model, training_model, prediction_model = create_models(
            backbone_retinanet=backbone('resnet50').retinanet,
            num_classes=6,
            weights=None,
            multi_gpu=False,
            freeze_backbone=True,
            lr=1e-3,
            config=read_config_file('/usr/local/bin/config/config_hand.ini'))
        training_model.load_weights(model_path)
        infer_model = convert_model(training_model, anchor_params = parse_anchor_parameters(read_config_file('/usr/local/bin/config/config_hand.ini')))
    
    return infer_model

# ===================== Function for loading images in correct orientation =================
def exif_transpose(img):
    if not img:
        return img

    exif_orientation_tag = 274

    # Check for EXIF data (only present on some files)
    if hasattr(img, "_getexif") and isinstance(img._getexif(), dict) and exif_orientation_tag in img._getexif():
        exif_data = img._getexif()
        orientation = exif_data[exif_orientation_tag]

        # Handle EXIF Orientation
        if orientation == 1:
            # Normal image - nothing to do!
            pass
        elif orientation == 2:
            # Mirrored left to right
            img = img.transpose(PIL.Image.FLIP_LEFT_RIGHT)
        elif orientation == 3:
            # Rotated 180 degrees
            img = img.rotate(180)
        elif orientation == 4:
            # Mirrored top to bottom
            img = img.rotate(180).transpose(PIL.Image.FLIP_LEFT_RIGHT)
        elif orientation == 5:
            # Mirrored along top-left diagonal
            img = img.rotate(-90, expand=True).transpose(PIL.Image.FLIP_LEFT_RIGHT)
        elif orientation == 6:
            # Rotated 90 degrees
            img = img.rotate(-90, expand=True)
        elif orientation == 7:
            # Mirrored along top-right diagonal
            img = img.rotate(90, expand=True).transpose(PIL.Image.FLIP_LEFT_RIGHT)
        elif orientation == 8:
            # Rotated 270 degrees
            img = img.rotate(90, expand=True)

    return img

# ======================= Function to load images using PIL ===================
def load_image_file(file, mode='RGB'):
    # Load the image with PIL
    img = PIL.Image.open(file)

    if hasattr(PIL.ImageOps, 'exif_transpose'):
        # Very recent versions of PIL can do exit transpose internally
        img = PIL.ImageOps.exif_transpose(img)
    else:
        # Otherwise, do the exif transpose ourselves
        img = exif_transpose(img)

    img = img.convert(mode)

    return np.array(img)


# =========================== Test generator for joint prediction ===================

def test_gen(image_ids, test_path , bs = 1, min_size = 1000, max_size = 1400, test = True):

    imgs = []
    scale = None
    idx = 0
    if test:
        path  = test_path
    else:
        path = 'training_data/images/'
    
    while idx < len(image_ids):
        if len(imgs) < bs:
            image_id_iter = image_ids[idx]
            imgs.append(resize_image(preprocess_image(load_image_file(path + "/" + image_ids[idx] + '.jpg')),min_side=min_size,max_side=max_size)[0])       
            scale = resize_image(preprocess_image(load_image_file(path + "/" +  image_ids[idx] + '.jpg')),min_side=min_size,max_side=max_size)[1]
            idx += 1

        else:
            yield image_id_iter, np.array(imgs), scale
            imgs = []

    if len(imgs) > 0:
        yield image_id_iter, np.array(imgs), scale      

# ===================== Loading image functionality ========================
def get_image_to_array(PATH, file, size):
    img = cv2.imread(os.path.join(PATH, file))
    img = cv2.resize(img, (size, size))
    if img.shape[2] ==1:
        img = np.dstack([img, img, img])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)/255.
    return img

def get_input_img_array(path, size):
    image = load_image_file(path)
    image = cv2.resize(image, (size, size))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32)/255.
    return image

def strong_aug(p=0.5):
    return Compose([
        RandomRotate90(),
        HorizontalFlip(p=1),
        # RandomGridShuffle(grid=(5, 5), p=0.1),
        Flip(),
        Transpose(),
        ChannelShuffle(p=1),
        ], p=p)

# ============================ Predicitons for joints =========================

def get_predictions_for_joints(test_path, image_ids_list, model_path_dict):
    np.random.seed(17)
    """
    Input -> Image ID list has to be UAB001-LF, .... format 
             Test_path->  Path where images are loacted for predcitions
             Model_path -> Dictionary with the path for Foot and Hand model detion
             Type -> Type of detection to be performed
    Output -> Prediction files saved in the desired folder
    
    """

    #All joints arrays 

    foot_finger = []
    foot_finger_id = []
    hand_finger = []
    hand_finger_id = []
    hand_ip = []
    hand_ip_id =[]
    wrist = []
    wrist_id = []

    for type in model_path_dict.keys():

        # Select image IDS
        # Model definition takes place here
        if type == 'Foot_detection':
            infer_model = get_joint_detection_model(model_path_dict[type], type)
            image_ids_filtered = [image for image in image_ids_list if image.split("-")[1] in ['LF','RF']]
            label_map = {1:'fin_ip',2:'fin_1',3:'fin_2',4:'fin_3',5:'fin_4'}
            # min_size = 800
            # max_size = 1100 
            # If this does'nt work use the upper sizes
            min_size = 1000
            max_size = 1400

        elif type == 'Hand_detection':
            infer_model = get_joint_detection_model(model_path_dict[type], type)
            image_ids_filtered = [image for image in image_ids_list if image.split("-")[1] in ['LH','RH']]
            label_map = {1:'fin_ip',2:'fin_1',3:'fin_2',4:'fin_3',5:'fin_4',6:'wrist'}
            min_size = 1000
            max_size = 1400 

        image_ids = sorted(image_ids_filtered)
        test_bs = 1

        # Joint prediction and extraction is done in this loop
        for image_id, imgs, scale in tqdm(test_gen(image_ids, test_path, bs = test_bs), total = math.ceil(len(image_ids)/test_bs)):
            
            # shapes_of_images.append(imgs.shape)
            boxes, scores, labels = infer_model.predict_on_batch(imgs)
            #Boxes set as per original image size
            # boxes /= scale

            temp = pd.DataFrame([list(labels[0]), list(scores[0])]).T
            temp.columns = ['labels', 'scores']
            temp['labels'] = temp['labels'] + 1

            temp_a = pd.DataFrame(boxes[0], dtype = 'int')
            temp_a.columns = ['xmin','ymin','xmax','ymax']
            final_temp = pd.concat([temp, temp_a], axis = 1)
            final_temp.columns = ['labels', 'scores', 'xmin','ymin','xmax','ymax']

            final_temp = final_temp[final_temp['scores'] > 0]
            final_temp['xmin'] = final_temp['xmin'].astype('int')
            final_temp['ymin'] = final_temp['ymin'].astype('int')
            final_temp['xmax'] = final_temp['xmax'].astype('int')
            final_temp['ymax'] = final_temp['ymax'].astype('int')
            print("Predictions read")

            idx = final_temp.groupby(['labels'])['scores'].transform(max) == final_temp['scores']
            ffl_df = final_temp[idx]
            abc = ffl_df.sort_values(by ='labels', ascending=True)
            abc['labels'] = abc['labels'].replace(label_map)
            print(image_id)
            print(abc)
            image_from_opencv = resize_image(load_image_file(os.path.join(test_path, image_id + ".jpg")), min_side=min_size,max_side=max_size)[0]

            ## Read the necessary image
            size = 224

            for labels in abc['labels'].unique().tolist():
                #Extracting box for the joint 
                boxes = np.array(abc[abc['labels'] == labels][['xmin','ymin','xmax','ymax']].values[0], dtype='int')

                #Saving the extracted joint in a associated list
                try:
                    if (image_id.split("-")[1] in ['LF', 'RF']) & (labels in ['fin_1' ,'fin_2','fin_3','fin_4']):
                        joint = image_from_opencv[boxes[1]: boxes[3], boxes[0]:boxes[2]]
                        joint = cv2.resize(joint, (size, size))
                        joint = cv2.cvtColor(joint, cv2.COLOR_BGR2RGB)
                        joint = joint.astype(np.float32)/255.
                        # print(joint.shape)
                        # plt.title(labels+ " " + image_id + "-" + str(labels) + '.jpg')
                        # plt.imshow(joint)
                        # plt.show()
                        foot_finger.append(joint)
                        foot_finger_id.append(image_id + "-" + str(labels) + '.jpg')

                    # Hand finger list creation 
                    elif (image_id.split("-")[1] in ['LF', 'RF']) & (labels in ['fin_ip']):
                        joint = image_from_opencv[boxes[1]: boxes[3], boxes[0]:boxes[2]]
                        joint = cv2.resize(joint, (size, size))
                        joint = cv2.cvtColor(joint, cv2.COLOR_BGR2RGB)
                        joint = joint.astype(np.float32)/255.
                        # print(joint.shape)
                        # plt.title(labels+ " " + image_id + "-" + str(labels) + '.jpg')
                        # plt.imshow(joint)
                        # plt.show()
                        hand_finger.append(joint)
                        hand_finger_id.append(image_id + "-" + str(labels) + '.jpg')

                    elif (image_id.split("-")[1] in ['LH', 'RH']) & (labels in ['fin_1' ,'fin_2','fin_3','fin_4']):
                        joint = image_from_opencv[boxes[1]: boxes[3], boxes[0]:boxes[2]]
                        joint = cv2.resize(joint, (size, size))
                        joint = cv2.cvtColor(joint, cv2.COLOR_BGR2RGB)
                        joint = joint.astype(np.float32)/255.
                        # print(joint.shape)
                        # plt.title(labels+ " " + image_id + "-" + str(labels) + '.jpg')
                        # plt.imshow(joint)
                        # plt.show()
                        hand_finger.append(joint)
                        hand_finger_id.append(image_id + "-" + str(labels) + '.jpg')

                    #Wrist joints creation
                    elif  (image_id.split("-")[1] in ['LH', 'RH']) & (labels in ['wrist']):
                        joint = image_from_opencv[boxes[1]: boxes[3], boxes[0]:boxes[2]]
                        joint = cv2.resize(joint, (size, size))
                        joint = cv2.cvtColor(joint, cv2.COLOR_BGR2RGB)
                        joint = joint.astype(np.float32)/255.
                        # print(joint.shape)
                        # plt.title(labels + " " + image_id + "-" + str(labels) + '.jpg')
                        # plt.imshow(joint)
                        # plt.show()
                        wrist.append(joint)
                        wrist_id.append(image_id + "-" + str(labels) + '.jpg')

                    #Hand IP joint creation
                    elif (image_id.split("-")[1] in ['LH', 'RH']) & (labels in ['fin_ip']):
                        joint = image_from_opencv[boxes[1]: boxes[3], boxes[0]:boxes[2]]
                        joint = cv2.resize(joint, (size, size))
                        joint = cv2.cvtColor(joint, cv2.COLOR_BGR2RGB)
                        joint = joint.astype(np.float32)/255.
                        # print(joint.shape)
                        # plt.title(labels + " " + image_id + "-" + str(labels) + '.jpg')
                        # plt.imshow(joint)
                        # plt.show()
                        hand_ip.append(joint)
                        hand_ip_id.append(image_id + "-" + str(labels) + '.jpg')
                except:
                    pass

        image_array_dict = {
                            'Foot_finger': [np.array(foot_finger), foot_finger_id],
                            'Hand_finger':[np.array(hand_finger), hand_finger_id],
                            'hand_ip': [np.array(hand_ip),hand_ip_id],
                            'wrist': [np.array(wrist), wrist_id]

                            }
    return image_array_dict

# ========================= Generates the prediction file for the Patients ==============================

def get_final_submission_file(check, joint_type, mapping_file, PATIENT_ID_LIST):
    """Psuedo code -- 
    Input -> Prediction file for Hand fingers, Foot fingers, Wrists and Hand IP joints, including the joint IDS and Labels for the Joint patches
             Specify type of prediction file associated
             Mapping file with the relevent column names to be given to the Prediction files
             Patient ID list in the test set without JPG extension
    Output -> Prediction file / joint
    """
    if joint_type in ['wrist', 'hand_ip']:
        check['Patient_ID'] = check['Joint_image_ID'].str.split("-", expand=True)[0]
        check['limb_name']  = check['Joint_image_ID'].str.split("-", expand=True)[1]
        check['joint_name'] = check['Joint_image_ID'].str.split("-", expand=True)[2]
        final_df = pd.DataFrame()
        for val in check['limb_name'].unique().tolist():
            if val =='LH':
                columns = mapping_file[val+"-"+check['joint_name'].values[0]] + ['Joint_image_ID', 'Patient_ID', 'limb_name','joint_name']
            elif val == 'RH':
                columns = mapping_file[val+"-"+check['joint_name'].values[0]]+ ['Joint_image_ID', 'Patient_ID', 'limb_name','joint_name']

            # print(columns)
            # print("Got the column names ", columns)
            temp = check[check['limb_name'] == val].reset_index()
            del temp['index']
            temp.columns = columns 

            final_df = pd.concat([final_df, temp[['Patient_ID'] + mapping_file[val+"-"+check['joint_name'].values[0]]] ], axis = 1)
            # print(final_df)

        final_df  = final_df.loc[:,~final_df.columns.duplicated()]

    elif joint_type in ['Hand_finger','Foot_finger']:

        check['Patient_ID'] = check['Joint_image_ID'].str.split("-", expand=True)[0]
        check['limb_joint_name']  = check['Joint_image_ID'].str.split("-", expand = True)[1] + "-" + check['Joint_image_ID'].str.split("-", expand=True)[2]

        final_df = pd.DataFrame()
        for limb in check['limb_joint_name'].unique().tolist():
            columns = mapping_file[limb] + ['Joint_image_ID', 'Patient_ID', 'limb_joint_name']

            temp=check[check['limb_joint_name'] == limb].reset_index()
            del temp['index']
            temp.columns = columns
            final_df = pd.concat([final_df, temp[['Patient_ID'] + mapping_file[limb]]], axis = 1)

        # final_df['Final_ptid'] = PATIENT_ID_LIST
        final_df = final_df.loc[:,~final_df.columns.duplicated()]
        all_cols = final_df.columns[~final_df.columns.str.contains('Patient_ID')].to_list()
        final_df = final_df[all_cols + ['Patient_ID']]


    return final_df
    
# ============================== Loads model's and makes prediction =============================
def get_prediction_files(test_path,  joint_model_path_mapping, 
                         mapping_file, column_mapping, erosion_cols, narrowing_cols,
                         image_batch_array):
    """ 
    --------------------------------------------------

    Input -> Joints_path list, test_images_path, joint_image_mapping_dict, 
             Joint_models_mapping_dict, column_mapping_file, all_columns_in_prediction_file, 
             erosion_cols, narrowing_cols 
             
    Output -> Final prediction file 
    
    -------------------------------------------------
    """
    final_prediction_file = pd.DataFrame()
    for joint_type in joint_model_path_mapping.keys():
        #Dependency
        def rmse (y_true, y_pred):
            return K.sqrt(K.mean(K.square(y_pred -y_true), axis=-1))

        # from keras_radam import RAdam
        model_weights = joint_model_path_mapping[joint_type]

        # depend = {'rmse': rmse}
        loaded_model = load_model(model_weights, custom_objects={'rmse': rmse})
        print("Model loaded succesfully -> " + joint_type)

        # #Test data generators initiation
        # img = next(RA_test_image_generator(joints_path, sorted(joint_image_id_mapping[joint_type]), augment = False, size = 224))
        # print("All test images loaded ")

        #Predictions from model
        joint_type_df = pd.DataFrame(loaded_model.predict(image_batch_array[joint_type][0]))
        joint_type_df.columns = ['labels_'+ str(i) for i in range(joint_type_df.shape[1])]

        joint_type_df['Joint_image_ID'] = [img.split(".")[0] for img in image_batch_array[joint_type][1]]
        print("Predictions generated ")

        # Unique patients in test set
        patient_id_list = sorted(list(np.unique([img.split(".")[0].split("-")[0] for img in os.listdir(test_path) if img != 'desktop'])))
        #Calling the prediciton file generator function
        temp = get_final_submission_file(joint_type_df, joint_type, mapping_file, patient_id_list)

        final_prediction_file = pd.concat([final_prediction_file, temp], axis = 1)
        print("All steps for {} completed succesfully ".format(joint_type))

    #Final prediction file
    final_prediction_file = final_prediction_file .loc[:,~final_prediction_file .columns.duplicated()]
    #final_prediction_file.to_csv("error_file.csv")
    final_prediction_file['Overall_Tol'] = np.sum(final_prediction_file[erosion_cols + narrowing_cols], axis = 1)
    final_prediction_file['Overall_erosion'] = np.sum(final_prediction_file[erosion_cols],              axis = 1)
    final_prediction_file['Overall_narrowing'] = np.sum(final_prediction_file[narrowing_cols],          axis = 1)

    print(final_prediction_file.shape)
    return final_prediction_file[['Patient_ID', 'Overall_Tol', 'Overall_erosion', 'Overall_narrowing'] + column_mapping]


if __name__ == '__main__':
    # Defining the Object detection models for Foot and Hands 
    hand_model_path = "/usr/local/bin/object_detection_model/resnet50_csv_25_hand_classloss_0.0632.h5"
    foot_model_path = "/usr/local/bin/object_detection_model/resnet50_csv_21_foot_classloss_0.0848.h5"

    obj_detection_model_dict = {'Foot_detection': foot_model_path,
                                'Hand_detection' : hand_model_path 
                                }
    # Test images path
    test_path = '/test'
    image_id_list = [img.split(".")[0] for img in os.listdir(test_path)]
    image_id_list = [image for image in image_id_list if image not in  ['template']]
    print(image_id_list)
    #Predictions for joints bounding boxes
    image_array_dict = get_predictions_for_joints(test_path, image_id_list, obj_detection_model_dict)

    print("Bounding boxes predicted")
    print("Joints extraction complete")

    final_mapping_file = {'LH-wrist': ['LH_wrist_E__lunate',
      'LH_wrist_E__mc1',
      'LH_wrist_E__mul',
      'LH_wrist_E__nav',
      'LH_wrist_E__radius',
      'LH_wrist_E__ulna',
      'LH_wrist_J__capnlun',
      'LH_wrist_J__cmc3',
      'LH_wrist_J__cmc4',
      'LH_wrist_J__cmc5',
      'LH_wrist_J__mna',
      'LH_wrist_J__radcar'],
     'RH-wrist': ['RH_wrist_E__lunate',
      'RH_wrist_E__mc1',
      'RH_wrist_E__mul',
      'RH_wrist_E__nav',
      'RH_wrist_E__radius',
      'RH_wrist_E__ulna',
      'RH_wrist_J__capnlun',
      'RH_wrist_J__cmc3',
      'RH_wrist_J__cmc4',
      'RH_wrist_J__cmc5',
      'RH_wrist_J__mna',
      'RH_wrist_J__radcar'],
     'LH-fin_1': ['LH_mcp_E__2', 'LH_mcp_J__2', 'LH_pip_E__2', 'LH_pip_J__2'],
     'LH-fin_2': ['LH_mcp_E__3', 'LH_mcp_J__3', 'LH_pip_E__3', 'LH_pip_J__3'],
     'LH-fin_3': ['LH_mcp_E__4', 'LH_mcp_J__4', 'LH_pip_E__4', 'LH_pip_J__4'],
     'LH-fin_4': ['LH_mcp_E__5', 'LH_mcp_J__5', 'LH_pip_E__5', 'LH_pip_J__5'],
     'LF-fin_ip': ['LF_mtp_E__1', 'LF_mtp_E__ip', 'LF_mtp_J__1', 'LF_mtp_J__ip'],
     'RH-fin_1': ['RH_mcp_E__2', 'RH_mcp_J__2', 'RH_pip_E__2', 'RH_pip_J__2'],
     'RH-fin_2': ['RH_mcp_E__3', 'RH_mcp_J__3', 'RH_pip_E__3', 'RH_pip_J__3'],
     'RH-fin_3': ['RH_mcp_E__4', 'RH_mcp_J__4', 'RH_pip_E__4', 'RH_pip_J__4'],
     'RH-fin_4': ['RH_mcp_E__5', 'RH_mcp_J__5', 'RH_pip_E__5', 'RH_pip_J__5'],
     'RF-fin_ip': ['RF_mtp_E__1', 'RF_mtp_E__ip', 'RF_mtp_J__1', 'RF_mtp_J__ip'],

     'LH-fin_ip': ['LH_mcp_E__1', 'LH_mcp_E__ip', 'LH_mcp_J__1'],
     'RH-fin_ip': ['RH_mcp_E__1', 'RH_mcp_E__ip', 'RH_mcp_J__1'],

     'LF-fin_1': ['LF_mtp_E__2', 'LF_mtp_J__2'],
    'LF-fin_2' : ['LF_mtp_E__3', 'LF_mtp_J__3'],
    'LF-fin_3' : ['LF_mtp_E__4', 'LF_mtp_J__4'],
    'LF-fin_4' : ['LF_mtp_E__5', 'LF_mtp_J__5'],
    'RF-fin_1': ['RF_mtp_E__2', 'RF_mtp_J__2'],
    'RF-fin_2' : ['RF_mtp_E__3', 'RF_mtp_J__3'],
    'RF-fin_3' : ['RF_mtp_E__4', 'RF_mtp_J__4'],
    'RF-fin_4' : ['RF_mtp_E__5', 'RF_mtp_J__5']
     }

    all_columns = ['LH_mcp_E__ip', 'LH_pip_E__2', 'LH_pip_E__3', 'LH_pip_E__4',
           'LH_pip_E__5', 'LH_mcp_E__1', 'LH_mcp_E__2', 'LH_mcp_E__3',
           'LH_mcp_E__4', 'LH_mcp_E__5', 'LH_wrist_E__mc1', 'LH_wrist_E__mul',
           'LH_wrist_E__nav', 'LH_wrist_E__lunate', 'LH_wrist_E__radius',
           'LH_wrist_E__ulna', 'RH_mcp_E__ip', 'RH_pip_E__2', 'RH_pip_E__3',
           'RH_pip_E__4', 'RH_pip_E__5', 'RH_mcp_E__1', 'RH_mcp_E__2',
           'RH_mcp_E__3', 'RH_mcp_E__4', 'RH_mcp_E__5', 'RH_wrist_E__mc1',
           'RH_wrist_E__mul', 'RH_wrist_E__nav', 'RH_wrist_E__lunate',
           'RH_wrist_E__radius', 'RH_wrist_E__ulna', 'LF_mtp_E__ip', 'LF_mtp_E__1',
           'LF_mtp_E__2', 'LF_mtp_E__3', 'LF_mtp_E__4', 'LF_mtp_E__5',
           'RF_mtp_E__ip', 'RF_mtp_E__1', 'RF_mtp_E__2', 'RF_mtp_E__3',
           'RF_mtp_E__4', 'RF_mtp_E__5', 'LH_pip_J__2', 'LH_pip_J__3',
           'LH_pip_J__4', 'LH_pip_J__5', 'LH_mcp_J__1', 'LH_mcp_J__2',
           'LH_mcp_J__3', 'LH_mcp_J__4', 'LH_mcp_J__5', 'LH_wrist_J__cmc3',
           'LH_wrist_J__cmc4', 'LH_wrist_J__cmc5', 'LH_wrist_J__mna',
           'LH_wrist_J__capnlun', 'LH_wrist_J__radcar', 'RH_pip_J__2',
           'RH_pip_J__3', 'RH_pip_J__4', 'RH_pip_J__5', 'RH_mcp_J__1',
           'RH_mcp_J__2', 'RH_mcp_J__3', 'RH_mcp_J__4', 'RH_mcp_J__5',
           'RH_wrist_J__cmc3', 'RH_wrist_J__cmc4', 'RH_wrist_J__cmc5',
           'RH_wrist_J__mna', 'RH_wrist_J__capnlun', 'RH_wrist_J__radcar',
           'RF_mtp_J__ip', 'LF_mtp_J__1', 'LF_mtp_J__2', 'LF_mtp_J__3',
           'LF_mtp_J__4', 'LF_mtp_J__5', 'LF_mtp_J__ip', 'RF_mtp_J__1',
           'RF_mtp_J__2', 'RF_mtp_J__3', 'RF_mtp_J__4', 'RF_mtp_J__5']
    erosion_cols = ['LH_mcp_E__ip', 'LH_pip_E__2', 'LH_pip_E__3', 'LH_pip_E__4',
           'LH_pip_E__5', 'LH_mcp_E__1', 'LH_mcp_E__2', 'LH_mcp_E__3',
           'LH_mcp_E__4', 'LH_mcp_E__5', 'LH_wrist_E__mc1', 'LH_wrist_E__mul',
           'LH_wrist_E__nav', 'LH_wrist_E__lunate', 'LH_wrist_E__radius',
           'LH_wrist_E__ulna', 'RH_mcp_E__ip', 'RH_pip_E__2', 'RH_pip_E__3',
           'RH_pip_E__4', 'RH_pip_E__5', 'RH_mcp_E__1', 'RH_mcp_E__2',
           'RH_mcp_E__3', 'RH_mcp_E__4', 'RH_mcp_E__5', 'RH_wrist_E__mc1',
           'RH_wrist_E__mul', 'RH_wrist_E__nav', 'RH_wrist_E__lunate',
           'RH_wrist_E__radius', 'RH_wrist_E__ulna', 'LF_mtp_E__ip', 'LF_mtp_E__1',
           'LF_mtp_E__2', 'LF_mtp_E__3', 'LF_mtp_E__4', 'LF_mtp_E__5',
           'RF_mtp_E__ip', 'RF_mtp_E__1', 'RF_mtp_E__2', 'RF_mtp_E__3',
           'RF_mtp_E__4', 'RF_mtp_E__5']
    narrowing_cols = ['LH_pip_J__2', 'LH_pip_J__3', 'LH_pip_J__4', 'LH_pip_J__5',
           'LH_mcp_J__1', 'LH_mcp_J__2', 'LH_mcp_J__3', 'LH_mcp_J__4',
           'LH_mcp_J__5', 'LH_wrist_J__cmc3', 'LH_wrist_J__cmc4',
           'LH_wrist_J__cmc5', 'LH_wrist_J__mna', 'LH_wrist_J__capnlun',
           'LH_wrist_J__radcar', 'RH_pip_J__2', 'RH_pip_J__3', 'RH_pip_J__4',
           'RH_pip_J__5', 'RH_mcp_J__1', 'RH_mcp_J__2', 'RH_mcp_J__3',
           'RH_mcp_J__4', 'RH_mcp_J__5', 'RH_wrist_J__cmc3', 'RH_wrist_J__cmc4',
           'RH_wrist_J__cmc5', 'RH_wrist_J__mna', 'RH_wrist_J__capnlun',
           'RH_wrist_J__radcar', 'RF_mtp_J__ip', 'LF_mtp_J__1', 'LF_mtp_J__2',
           'LF_mtp_J__3', 'LF_mtp_J__4', 'LF_mtp_J__5', 'LF_mtp_J__ip',
           'RF_mtp_J__1', 'RF_mtp_J__2', 'RF_mtp_J__3', 'RF_mtp_J__4',
           'RF_mtp_J__5']

    joint_model_path_mapping = {
                        'Foot_finger' : '/usr/local/bin/joint_prediction_model/foot_finger_model_best_val_rmse_0.21833.h5',
                        'Hand_finger' : '/usr/local/bin/joint_prediction_model/hand_finger_model_EffnetB4_best.h5',
                        'wrist': '/usr/local/bin/joint_prediction_model/wrist_model_EffnetB4_best.h5',
                        'hand_ip':'/usr/local/bin/joint_prediction_model/hand_ip_model_best.h5'
                        }

    #  =================================== Final prediction_function ===============
    prediction_df = get_prediction_files(test_path, joint_model_path_mapping, 
                                        final_mapping_file, all_columns, erosion_cols, 
                                        narrowing_cols, image_array_dict)
    print(prediction_df)
    prediction_df.to_csv('/output/predictions.csv', index=False)


