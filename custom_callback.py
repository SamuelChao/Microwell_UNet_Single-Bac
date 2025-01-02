from tensorflow import keras
from keras import callbacks
from keras.callbacks import Callback
from keras.callbacks import ModelCheckpoint


class CustomHistory(Callback):
    def __init__(self):
        self.losses = []
        self.binary_io_u_values = []
        self.val_losses = []
        self.val_binary_io_u_values = []

    def on_epoch_end(self, epoch, logs=None):
        self.losses.append(logs.get('loss'))
        self.binary_io_u_values.append(logs.get('binary_io_u'))
        self.val_losses.append(logs.get('val_loss'))
        self.val_binary_io_u_values.append(logs.get('val_binary_io_u'))




def get_callback(model_name: str = 'UNetPP',
                 record_dir: str = 'UNetPP_result/'):

    custom_history = CustomHistory()
    
    checkpoint = ModelCheckpoint(
        record_dir + 'best_' + model_name +'_model.keras', 
        save_weights_only=True, 
        monitor='val_loss', 
        verbose=1, 
        save_best_only=True, 
        mode='min')
    
    callbacks_list = [
        custom_history, 
        checkpoint,
        callbacks.TensorBoard(record_dir + 'log/', histogram_freq = -1)  
        ]

    return callbacks_list