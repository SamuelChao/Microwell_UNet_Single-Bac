import os
from os.path import join
import numpy as np
import tensorflow as tf
from tensorflow import keras
from  matplotlib import pyplot as plt
import random
import pandas as pd

from dataloader import get_dataset_all, all_dataset_size
from unetpp_model import build_unetPP_model

#### Repeat for safety and reabiility #####################################################

def display_segmentation(display_list, num):
    plt.figure(figsize=(18, 18))

    title = ['Input Image', 'True Mask', 'Predict Map', 'Predict Mask']

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.title(title[i])
        plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
        plt.axis('off')

    plt.savefig(unetPP_folder + f'outputs/validate_{num}_segmentation.png',
                dpi=300, transparent=True, bbox_inches='tight')
    
                # img.save(unetPP_folder + f'outputs/validate_{num}_well.png')
#     plt.show()

# Function to create a mask out of network prediction
def create_mask(pred_mask: tf.Tensor) -> tf.Tensor:
    # Round to closest
    pred_mask = tf.math.round(pred_mask)

    # [IMG_SIZE, IMG_SIZE] -> [IMG_SIZE, IMG_SIZE, 1]
    pred_mask = tf.expand_dims(pred_mask, axis=-1)
    return pred_mask



# Function to show predictions
def show_validation(dataset=None, num=1, model = build_unetPP_model()):
    if dataset:
        # Predict and show image from input dataset
        for image, mask in dataset.take(num):
            pred_mask = model.predict(image)
            display_segmentation([image, mask, pred_mask, create_mask(pred_mask)], num), 

            # img = tf.keras.preprocessing.image.array_to_img(image[0])
            # img.save(unetPP_folder + f'outputs/validate_{num}_well.png')

            # true_mask_img = tf.keras.preprocessing.image.array_to_img(mask)
            # true_mask_img.save(unetPP_folder + f'outputs/validate_{num}_true_mask.png')

            # pred_map_img = tf.keras.preprocessing.image.array_to_img(pred_mask)
            # pred_map_img.save(unetPP_folder + f'outputs/validate_{num}_predict_map.png')

            # pred_mask_img = tf.keras.preprocessing.image.array_to_img(create_mask(pred_mask))
            # pred_mask_img.save(unetPP_folder + f'outputs/validate_{num}_predict_mask.png')


    else:
        print("No dataset available")

    
    return



if __name__ == '__main__':

    print(tf.config.list_physical_devices('GPU'))

    dataset = get_dataset_all(dataset_dir = "./dataset")
    unetPP_folder = 'UNetPP_result/'

    model = build_unetPP_model()
    model.load_weights(unetPP_folder + 'best_UNetPP_model.keras')
    # model.load_weights(unetPP_folder + 'epoch300_UnetPP_model.keras')
    model.summary()


    ## Show validation image, true mask & predict mask
    show_validation(dataset['test'], 20, model = model)