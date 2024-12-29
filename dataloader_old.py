import os
from os.path import join
import numpy as np
import tensorflow as tf
from tensorflow import keras
from  matplotlib import pyplot as plt
# %matplotlib inline

BATCH_SIZE = 15
BUFFER_SIZE = 1000
IMG_SIZE = 128
N_CHANNELS = 3
N_CLASSES = 1
SEED = 123

# Function to load image and return a dictionary
def parse_image(img_path: str) -> dict:
    image = tf.io.read_file(img_path)
    image = tf.image.decode_png(image, channels=3)


    mask_path = tf.strings.regex_replace(img_path,"images", "masks")
    mask_path = tf.strings.regex_replace(mask_path, "well", "mask")
    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask, channels=3)


    bac_label = np.array([255, 255, 255])


    # Convert to mask to binary mask
    bac_label = np.array([255, 255, 255])
    mask = tf.experimental.numpy.all(mask == bac_label, axis = 2)
    mask = tf.cast(mask, tf.uint8)
    mask = tf.expand_dims(mask, axis=-1)

    return {'image': image, 'segmentation_mask': mask}



# Tensorflow function to rescale images to [0, 1]
@tf.function
def normalize(input_image: tf.Tensor, input_mask: tf.Tensor) -> tuple:
    input_image = tf.cast(input_image, tf.float32) / 255.0
    return input_image, input_mask

# Tensorflow function to apply preprocessing transformations of training images
@tf.function
def load_image_train(datapoint: dict) -> tuple:
    input_image = tf.image.resize(datapoint['image'], (IMG_SIZE, IMG_SIZE))
    input_mask = tf.image.resize(datapoint['segmentation_mask'], (IMG_SIZE, IMG_SIZE))
    input_mask = tf.math.round(input_mask)

    if tf.random.uniform(()) > 0.5:
        input_image = tf.image.flip_left_right(input_image)
        input_mask = tf.image.flip_left_right(input_mask)

    input_image, input_mask = normalize(input_image, input_mask)
    return input_image, input_mask

# Tensorflow function to preprocess validation images
@tf.function
def load_image_val(datapoint: dict) -> tuple:
    input_image = tf.image.resize(datapoint['image'], (IMG_SIZE, IMG_SIZE))
    input_mask = tf.image.resize(datapoint['segmentation_mask'], (IMG_SIZE, IMG_SIZE))
    input_mask = tf.math.round(input_mask)
    input_image, input_mask = normalize(input_image, input_mask)
    return input_image, input_mask

# Tensorflow function to preprocess testing images
@tf.function
def load_image_test(datapoint: dict) -> tuple:
# def load_image_test(datapoint: dict, IMG_SIZE: int = 128) -> tuple:
    input_image = tf.image.resize(datapoint['image'], (IMG_SIZE, IMG_SIZE))
    input_mask = tf.image.resize(datapoint['segmentation_mask'], (IMG_SIZE, IMG_SIZE))
    input_mask = tf.math.round(input_mask)
    input_image, input_mask = normalize(input_image, input_mask)
    return input_image, input_mask



# Function to view the images from the directory
def display_sample(display_list):
    plt.figure(figsize=(18, 18))
    # fig =  plt.subplots()

    title = ['Input Image', 'True Mask', 'Predicted Mask']

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.title(title[i])
        plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
        plt.axis('off')
        plt.savefig("debug_test/" + 'display_sample_test.png', dpi=300, transparent=True, bbox_inches='tight')

def get_dataset_all(dataset_dir: str =  "./dataset/"):

    pass







if __name__ == '__main__':

    print(tf.config.list_physical_devices('GPU'))

    # Load directories
    dataset_dir = "./dataset"
    # dataset_dir = "./dataset/"
    image_dir = join(dataset_dir, "images")
    # image_dir = dataset_dir + "images/"
    train_data_dir = image_dir
    # Number of training examples

    TRAINSET_SIZE = int(round(len(os.listdir(train_data_dir)) * 0.7))
    print(f"Number of Training Examples: {TRAINSET_SIZE}")

    VALIDSET_SIZE = int(len(os.listdir(train_data_dir)) * 0.1)
    print(f"Number of Validation Examples: {VALIDSET_SIZE}")

    TESTSET_SIZE = int(len(os.listdir(train_data_dir)) - TRAINSET_SIZE - VALIDSET_SIZE)
    print(f"Number of Testing Examples: {TESTSET_SIZE}")



    BATCH_SIZE = 15
    BUFFER_SIZE = 1000
    IMG_SIZE = 128
    N_CHANNELS = 3
    N_CLASSES = 1
    SEED = 123


    print("HAHA 01")

    ### Generate dataset variables
    all_dataset = tf.data.Dataset.list_files(train_data_dir + "/*.png",  shuffle = False)
    print(all_dataset)
    all_dataset = all_dataset.shuffle(BUFFER_SIZE, seed=SEED, reshuffle_each_iteration=False)
    all_dataset = all_dataset.map(parse_image)

    print("HAHA 02")

    train_dataset = all_dataset.take(TRAINSET_SIZE + VALIDSET_SIZE)
    val_dataset = train_dataset.skip(TRAINSET_SIZE)
    train_dataset = train_dataset.take(TRAINSET_SIZE)
    test_dataset = all_dataset.skip(TRAINSET_SIZE + VALIDSET_SIZE)

    print("HAHA 03")

    dataset = {"train": train_dataset, "val": val_dataset, "test": test_dataset}

    print(dataset)

    print("HAHA 04")

    # -- Train Dataset --#
    dataset['train'] = dataset['train'].map(load_image_train, num_parallel_calls=tf.data.AUTOTUNE)
    dataset['train'] = dataset['train'].shuffle(buffer_size=BUFFER_SIZE, seed=SEED)
    dataset['train'] = dataset['train'].repeat()
    dataset['train'] = dataset['train'].batch(BATCH_SIZE)
    dataset['train'] = dataset['train'].prefetch(buffer_size=tf.data.AUTOTUNE)

    #-- Validation Dataset --#
    dataset['val'] = dataset['val'].map(load_image_test)
    dataset['val'] = dataset['val'].repeat()
    dataset['val'] = dataset['val'].batch(BATCH_SIZE)
    dataset['val'] = dataset['val'].prefetch(buffer_size=tf.data.AUTOTUNE)

    #-- Testing Dataset --#
    dataset['test'] = dataset['test'].map(load_image_test)
    dataset['test'] = dataset['test'].batch(BATCH_SIZE)
    dataset['test'] = dataset['test'].prefetch(buffer_size=tf.data.AUTOTUNE)

    print(dataset['train'])
    print(dataset['val'])
    print(dataset['test'])

    print("HAHA 05")

    # View the training images from the directory
    for image, mask in dataset['train'].take(2):
        sample_image, sample_mask = image, mask

    print("HAHA 06")

    display_sample([sample_image[0], sample_mask[0]])

    print("HAHA 07")
