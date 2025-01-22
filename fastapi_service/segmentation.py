import pandas as pd
import numpy as np
import tensorflow as tf
import cv2
from time import time
from  matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from pathlib import Path
# Path("/my/directory").mkdir(parents=True, exist_ok=True)
# import pandas as pd
# import numpy as np
from os.path import join
# import tensorflow as tf
from time import time
from tensorflow import keras
from keras.losses import BinaryCrossentropy


# from dataloader import get_dataset_all, all_dataset_size
# from utils.unetpp_model import build_unetPP_model
# from utils.unet_model import build_unet_model
# from utils.eval_utils import acc_calculator
# from utils.eval_utils import get_predictions, get_confusion_matrix, plot_confusion_matrix
# from utils.eval_utils import convert_singlebac, get_singlebac_confusion_matrix, plot_singlebac_confusion_matrix

from DL_model.unetpp_model import build_unetPP_model

# model_name = 'UNetPP' 
# ## eval model setup 
# if model_name == 'UNetPP':
#     model = build_unetPP_model()
#     result_folder = 'UNetPP_result/'
# else:
#     raise Exception("Sorry, you do not choose the available DL model type")

model = build_unetPP_model()

## compile model
opt = keras.optimizers.Adam(learning_rate=1e-4, clipvalue=0.5)
m_iou = tf.keras.metrics.BinaryIoU(
    target_class_ids=[1], threshold=0.5, name=None, dtype=None
)
model.compile(loss=BinaryCrossentropy(), optimizer=opt,  metrics=[m_iou])

## load checkpoint model weight
model.load_weights('DL_model/epoch300_UNetPP_model.keras')
# model.summary()


# Function to save predictions
def predict_mask(image):

    pred = model.predict(tf.expand_dims(image, axis = 0))
    pred = pred[0]
    pred_mask = tf.math.round(pred)

    return pred_mask



