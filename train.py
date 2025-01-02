from os.path import join
from  matplotlib import pyplot as plt
import pandas as pd
from time import time

import tensorflow as tf
from tensorflow import keras
from keras.losses import BinaryCrossentropy
from keras import callbacks
from keras.callbacks import ModelCheckpoint


from dataloader import get_dataset_all, all_dataset_size
from unetpp_model import build_unetPP_model
from unet_model import build_unet_model
from custom_callback import CustomHistory


## Set Epochs 
EPOCHS = 300

## Set other configs
BATCH_SIZE = 15
BUFFER_SIZE = 1000
IMG_SIZE = 128
N_CHANNELS = 3
N_CLASSES = 1
SEED = 123

## Setup Dataset
dataset = get_dataset_all(dataset_dir = "./dataset")

## Select model type
## Two options:
## Option 1: 'UNetPP' 
## Option 2: 'UNet'
model_name = 'UNetPP' 


## result folder & build model
if model_name == 'UNetPP':
    result_folder = 'UNetPP_result/'
    model = build_unetPP_model()
    model.summary()
elif model_name == 'UNet':
    result_folder = 'UNet_result/'
    model = build_unet_model()
    model.summary()
else:
    raise Exception("Sorry, you do not choose the available DL model type")


# Optimization
opt = keras.optimizers.Adam(learning_rate=1e-4, clipvalue=0.5)
m_iou = tf.keras.metrics.BinaryIoU(
    target_class_ids=[1], threshold=0.5, name=None, dtype=None
)

# Compile
model.compile(loss=BinaryCrossentropy(), optimizer=opt,  metrics=[m_iou])




## Set Variables
dataset_size = all_dataset_size(dataset_dir = "./dataset")
TRAINSET_SIZE = dataset_size['TRAINSET_SIZE']
VALIDSET_SIZE = dataset_size['VALIDSET_SIZE']
TESTSET_SIZE = dataset_size['TESTSET_SIZE']

STEPS_PER_EPOCH = TRAINSET_SIZE // BATCH_SIZE
VALIDATION_STEPS = VALIDSET_SIZE // BATCH_SIZE


custom_history = CustomHistory()

checkpoint = ModelCheckpoint(
    result_folder + 'best_' + model_name +'_model.keras', 
    save_weights_only=True, 
    monitor='val_loss', 
    verbose=1, 
    save_best_only=True, 
    mode='min')

callbacks_list = [
    custom_history, 
    checkpoint,
    callbacks.TensorBoard(result_folder + 'log/', histogram_freq = -1)  
    ]


#################################################################
## Model Training
initial_time = time()

model.fit(dataset['train'], epochs=EPOCHS,
          steps_per_epoch=STEPS_PER_EPOCH,
          validation_data=dataset['val'],
          validation_steps=VALIDATION_STEPS,
          callbacks=callbacks_list)


print('Finished Unet++ Training')
print('Training time', time() - initial_time)
model.save_weights(result_folder + 'epoch300_UNetPP_model.keras')


#################################################################
### Save Losses & IoU records in csv format

## Training Loss curve: Dataframe & Save CSV
training_loss = custom_history.losses
epochs_list = list(range(1, len(training_loss) + 1))
df_tloss = pd.DataFrame({"Epochs": epochs_list, "train_loss" : training_loss})
df_tloss.to_csv(result_folder +'training_loss.csv', index=False)

## Validate Loss curve: Dataframe & Save CSV

val_losses = custom_history.val_losses
df_vloss = pd.DataFrame({"Epochs": epochs_list, "validate_loss" : val_losses})
df_vloss.to_csv(result_folder + 'validate_loss.csv', index=False)


## Training binaryIOU curve: Dataframe & Save CSV
binary_io_u_values = custom_history.binary_io_u_values
df_tbiou = pd.DataFrame({"Epochs": epochs_list, "train_bIOU" : binary_io_u_values})
df_tbiou.to_csv(result_folder + 'training_bIOU.csv', index=False)


## Validation binaryIOU curve: Dataframe & Save CSV
val_binary_io_u_values = custom_history.val_binary_io_u_values
df_vbiou = pd.DataFrame({"Epochs": epochs_list, "val_bIOU" : val_binary_io_u_values})
df_vbiou.to_csv(result_folder + 'validate_bIOU.csv', index=False)

#############################################################

## Learning Curve Plot

plt.rcParams['font.size'] = 8
plt.rcParams['figure.dpi'] = 300


fig, ax1 = plt.subplots()
color = 'tab:red'
ax1.set_xlabel('Epochs', fontsize=12)
ax1.set_ylabel('Loss', color=color, fontsize=12)
ax1.plot(df_tloss["Epochs"], df_tloss["train_loss"], '-', label='Training Loss', color=color)
ax1.plot(df_vloss["Epochs"], df_vloss["validate_loss"], '-', alpha = 0.5, label='Validation Loss', color=color)
ax1.tick_params(axis='y', labelcolor=color)
ax1.set_ylim([-0.01, 0.13])
ax1.set_box_aspect(0.75)



ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
color = 'tab:blue'
ax2.set_ylabel('IoU', color=color, fontsize=12)  # we already handled the x-label with ax1
ax2.plot(df_tbiou["Epochs"], df_tbiou["train_bIOU"], '-', label='Training IoU', color=color)
ax2.plot(df_vbiou["Epochs"], df_vbiou["val_bIOU"], '-', alpha = 0.5, label='Validation IoU', color=color)
ax2.tick_params(axis='y', labelcolor=color)

ax2.set_ylim([-0.1, 0.9])
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc=9)


plt.show()

fig.savefig(result_folder + 'learning_curve.png')
