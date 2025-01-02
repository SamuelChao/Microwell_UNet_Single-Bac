import pandas as pd
import numpy as np
import tensorflow as tf
import cv2
from time import time
from  matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay





# Function to calculate mask over image
def weighted_img(img, initial_img, α=1., β=0.5, γ=0.):
    return cv2.addWeighted(initial_img, α, img, β, γ)

# Function to process an individual image and it's mask
def process_image_mask(image, mask):
    # Round to closest
    mask = tf.math.round(mask)

    # Convert to mask image
    zero_image = np.zeros_like(mask)
    mask = np.dstack((mask, zero_image, zero_image))
    mask = np.asarray(mask, np.float32)

    # Convert to image image
    image = np.asarray(image, np.float32)

    # Get the final image
    final_image = weighted_img(mask, image)

    return final_image


### Function to calculate bacteria number inside microwell
def bacteria_count(mask):
    mask = mask = tf.math.round(mask)
    mask_img = tf.keras.preprocessing.image.array_to_img(mask)
    img2 = cv2.cvtColor(np.asarray(mask_img), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    cnts = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    bacNum = len(cnts)
    return bacNum

# Function to save the images as a plot
def save_predict_sample(display_list, index, result_folder):
    plt.figure(figsize=(18, 18))

    title = ['Input Image', 'True Mask', 'Predicted Mask']

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.title(title[i])
        plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
        plt.axis('off')

    plt.savefig(result_folder + f'outputs/{index}.png')
    plt.show()


    image_array = display_list[0]
    true_mask_array = display_list[1]
    pred_mask_array = display_list[2]

    img = tf.keras.preprocessing.image.array_to_img(image_array)
    img.save(result_folder + f'outputs/{index}_well.png')
    true_mask_img = tf.keras.preprocessing.image.array_to_img(true_mask_array)
    true_mask_img.save(result_folder + f'outputs/{index}_true.png')
    pred_mask_img = tf.keras.preprocessing.image.array_to_img(pred_mask_array)
    pred_mask_img.save(result_folder + f'outputs/{index}_predict.png')



# Function to save predictions
def get_predictions(dataset, model, result_folder):
    # Predict and save image the from input dataset
    True_Counts = []
    Pred_Counts = []
    index = 0
    for batch_image, batch_mask in dataset:
        for image, mask in zip(batch_image, batch_mask):
            print(f"Processing image : {index}")
            pred_mask = model.predict(tf.expand_dims(image, axis = 0))

            true_bacNum = bacteria_count(mask)
            True_Counts.append(true_bacNum)
            print("True Bac-Num:    ", true_bacNum, "\n")

            pred_bacNum = bacteria_count(pred_mask[0])
            Pred_Counts.append(pred_bacNum)
            print("Predict Bac-Num: ", pred_bacNum, "\n")

            display_list = [image, 
                            process_image_mask(image, mask), 
                            process_image_mask(image, pred_mask[0])
                            ]
            
            save_predict_sample(display_list, 
                                index, 
                                result_folder)

            index += 1


    num_result = pd.DataFrame({'True_Number': True_Counts, 
                               'Predict_Number': Pred_Counts})
    num_result.to_csv(
                result_folder + 'Number_Result.csv', 
                index= False , 
                header = True
                )

    return True_Counts, Pred_Counts
    

## Bacteria-counts accuracy
def acc_calculator(predict, label):
    total = len(predict)
    correct = 0
    for i in range(total):
        if (predict[i] == label[i]):
            correct += 1


    return round(100 * correct / total, 2)

########################################################
#### Bacteria-counts confusion matrix

## Generate Confusion Matrix
def get_confusion_matrix(True_Counts, Pred_Counts, result_folder):

    conf = confusion_matrix(True_Counts, Pred_Counts, normalize='false')
    nor_conf = confusion_matrix(True_Counts, Pred_Counts,normalize='true')

    conf_df = pd.DataFrame(conf)
    conf_df.to_csv(result_folder + 'Number_Conf.csv', 
                  index= False , 
                  header = False)

    nor_conf_df = pd.DataFrame(nor_conf)  
    nor_conf_df.to_csv(result_folder + 'Number_Conf_Nor.csv', 
                       index= False , 
                       header = False)

    return conf, nor_conf


def plot_confusion_matrix(conf, nor_conf, result_folder):
    plt.rcParams['font.size'] = 8
    plt.rcParams['figure.dpi'] = 300

    # Number confusion matrix
    conf = np.around(conf, 2)
    conf_disp = ConfusionMatrixDisplay(confusion_matrix=conf)
    conf_disp.plot(cmap ='gist_yarg', colorbar=False)
    plt.xlabel('Predicted Number', fontsize=12)
    plt.ylabel('True Number', fontsize=12)

    plt.savefig(result_folder + 'Number_Conf.png',
                 dpi=300, 
                 transparent=True, 
                 bbox_inches='tight')
    
    # Normalizaed confusion matrix
        
    nor_conf = np.around(nor_conf, 2)
    nor_disp = ConfusionMatrixDisplay(confusion_matrix=nor_conf)
    nor_disp.plot(cmap ='gist_yarg', colorbar=False)
    plt.xlabel('Predicted Number', fontsize=12)
    plt.ylabel('True Number', fontsize=12)

    plt.savefig(result_folder + 'Number_Conf_Nor.png', 
                dpi=300, 
                transparent=True, 
                bbox_inches='tight')


########################################################
#### Microwell Single-bacterium inference

def convert_singlebac(True_Counts, Pred_Counts, result_folder):
    true_single = np.array(True_Counts)
    true_single[true_single != 1] = 0

    pred_single = np.array(Pred_Counts)
    pred_single[pred_single != 1] = 0

    single_result = pd.DataFrame(
        {'True_Single': true_single, 'Pred_Single': pred_single})
    
    single_result.to_csv(result_folder + 'Single_Result.csv', 
                         index= False , 
                         header = True)
    
    return true_single, pred_single


## Generate Single-Bacterium Confusion Matrix
def get_singlebac_confusion_matrix(true_single, pred_single, result_folder):

    conf = confusion_matrix(true_single, pred_single, normalize='false')
    nor_conf = confusion_matrix(true_single, pred_single, normalize='true')

    conf_df = pd.DataFrame(conf)
    conf_df.to_csv(result_folder + 'Single_Conf.csv', 
                  index= False , 
                  header = False)

    nor_conf_df = pd.DataFrame(nor_conf)  
    nor_conf_df.to_csv(result_folder + 'Single_Conf_Nor.csv', 
                       index= False , 
                       header = False)

    return conf, nor_conf


def plot_singlebac_confusion_matrix(conf, nor_conf, result_folder):
    plt.rcParams['font.size'] = 8
    plt.rcParams['figure.dpi'] = 300
    singleBac_labels = np.array(['Non-Single', 'Single-Bacterium'])

    ## Plot Single-Bac confusion matrix 
    disp = ConfusionMatrixDisplay(confusion_matrix=conf, display_labels = singleBac_labels)
    disp.plot(cmap ='gist_yarg', colorbar=False)
    plt.xticks(fontsize = 10)
    plt.yticks(rotation=90, ha='right', fontsize = 10, 
               rotation_mode='default', va="center")
    plt.xlabel('Prediction', fontsize=12)
    plt.ylabel('True', fontsize=12)
    plt.savefig(result_folder + 'Single_Conf.png',
                 dpi=300, transparent=True, bbox_inches='tight')

    ## Plot Normalized Single-Bac confusion matrix 
    nor_conf = np.around(nor_conf, 2)
    nor_disp = ConfusionMatrixDisplay(confusion_matrix=nor_conf, display_labels = singleBac_labels)
    nor_disp.plot(cmap ='gist_yarg', colorbar=False)
    plt.xticks(fontsize = 10)
    plt.yticks(rotation=90, ha='right', fontsize = 10, 
               rotation_mode='default', va="center")
    plt.xlabel('Prediction', fontsize=12)
    plt.ylabel('True', fontsize=12)
    plt.savefig(result_folder + 'Single_Conf_Nor.png', 
                dpi=300, transparent=True, bbox_inches='tight')