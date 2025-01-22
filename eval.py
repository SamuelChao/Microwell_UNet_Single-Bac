import pandas as pd
import numpy as np
from os.path import join
import tensorflow as tf
from time import time
from tensorflow import keras
from keras.losses import BinaryCrossentropy


from dataloader import get_dataset_all, all_dataset_size
from utils.unetpp_model import build_unetPP_model
from utils.unet_model import build_unet_model
from utils.eval_utils import acc_calculator
from utils.eval_utils import get_predictions, get_confusion_matrix, plot_confusion_matrix
from utils.eval_utils import convert_singlebac, get_singlebac_confusion_matrix, plot_singlebac_confusion_matrix



## Setup Dataset
dataset = get_dataset_all(dataset_dir = "./dataset")

## Select model type
## Two options:
## Option 1: 'UNetPP' 
## Option 2: 'UNet'
model_name = 'UNetPP' 


## eval model setup 
if model_name == 'UNetPP':
    model = build_unetPP_model()
    result_folder = 'UNetPP_result/'
elif model_name == 'UNet':
    model = build_unet_model()
    result_folder = 'UNet_result/'
else:
    raise Exception("Sorry, you do not choose the available DL model type")


## compile model
opt = keras.optimizers.Adam(learning_rate=1e-4, clipvalue=0.5)
m_iou = tf.keras.metrics.BinaryIoU(
    target_class_ids=[1], threshold=0.5, name=None, dtype=None
)
model.compile(loss=BinaryCrossentropy(), optimizer=opt,  metrics=[m_iou])

## load checkpoint model weight
model.load_weights(result_folder + f'epoch300_{model_name}_model.keras')
model.summary()


## Calculate Time of test-set prediction
initial_time = time()
for image, mask in dataset['test']:
    pred_mask = model.predict(image)

print(f'Finished {model_name} Testing')
print('Testing time', time() - initial_time)


True_Counts, Pred_Counts = get_predictions(
                            dataset['test'], 
                            model, 
                            result_folder)



## Bacteria-counts accuracy

num_acc = acc_calculator(Pred_Counts, True_Counts)
print("Counts Accuracy: ", num_acc)

num_acc_df = pd.DataFrame({'Number_Accuracy': [num_acc]})
num_acc_df.to_csv(result_folder + 'Number_Accuracy.csv',
                   index= False, 
                   header = True)


conf, nor_conf = get_confusion_matrix(
    True_Counts, Pred_Counts, result_folder)

plot_confusion_matrix(conf, nor_conf, result_folder)

############################################################
## Microwell Single-bacterium inference
true_single, pred_single = convert_singlebac(
                                    True_Counts, 
                                    Pred_Counts, 
                                    result_folder
                                    )


acc_single = acc_calculator(pred_single, true_single)
print('Single_Accuracy: ', acc_single)

acc_single_df = pd.DataFrame({'Single_Accuracy': [acc_single]})
acc_single_df.to_csv(result_folder + 'Single_Accuracy.csv', 
                     index= False , 
                     header = True)



conf_single, nor_conf_single = get_singlebac_confusion_matrix(
    true_single, pred_single, result_folder)

plot_singlebac_confusion_matrix(conf_single, nor_conf_single, result_folder)