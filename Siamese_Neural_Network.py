
from os.path import isfile, join
from os import listdir
import re
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from functools import partial
from kaggle_datasets import KaggleDatasets
from sklearn.model_selection import train_test_split
import tempfile
import matplotlib.pyplot as plt
from keras import layers
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate
from keras.models import Model

from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import Concatenate
from keras.layers.core import Lambda, Flatten, Dense
from keras.initializers import glorot_uniform

from keras.engine.topology import Layer
from keras.regularizers import l2
from keras import backend as K
from keras.layers import Input


try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    print('Device:', tpu.master())
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
except:
    strategy = tf.distribute.get_strategy()
print('Number of replicas:', strategy.num_replicas_in_sync)
    
AUTOTUNE = tf.data.experimental.AUTOTUNE
GCS_PATH = KaggleDatasets().get_gcs_path()
BATCH_SIZE = 16 * strategy.num_replicas_in_sync
IMAGE_SIZE = [1024, 1024]
IMAGE_RESIZE = [128, 128]


# LOAD DATA

TRAINING_FILENAMES, VALID_FILENAMES = train_test_split(
    tf.io.gfile.glob(GCS_PATH + '/tfrecords/train*.tfrec'),
    test_size=0.1, random_state=5
)
TEST_FILENAMES = tf.io.gfile.glob(GCS_PATH + '/tfrecords/test*.tfrec')
print('Train TFRecord Files:', len(TRAINING_FILENAMES))
print('Validation TFRecord Files:', len(VALID_FILENAMES))
print('Test TFRecord Files:', len(TEST_FILENAMES))

def decode_image(image):
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.reshape(image, [*IMAGE_SIZE, 3])
    return image
    
 def read_tfrecord(example, labeled):
    tfrecord_format = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "target": tf.io.FixedLenFeature([], tf.int64)
    } if labeled else {
        "image": tf.io.FixedLenFeature([], tf.string),
        "image_name": tf.io.FixedLenFeature([], tf.string)
    }
    example = tf.io.parse_single_example(example, tfrecord_format)
    image = decode_image(example['image'])
    if labeled:
        label = tf.cast(example['target'], tf.int32)
        return image, label
    idnum = example['image_name']
    return image, idnum
    
def load_dataset(filenames, labeled=True, ordered=False):
    ignore_order = tf.data.Options()
    if not ordered:
        ignore_order.experimental_deterministic = False # disable order, increase speed
    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTOTUNE) # automatically interleaves reads from multiple files
    dataset = dataset.with_options(ignore_order) # uses data as soon as it streams in, rather than in its original order
    dataset = dataset.map(partial(read_tfrecord, labeled=labeled), num_parallel_calls=AUTOTUNE)
    # returns a dataset of (image, label) pairs if labeled=True or (image, id) pairs if labeled=False
    return dataset
    
    
# DATA AUGMENTATION

def augmentation_pipeline(image, label):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.resize(image, IMAGE_RESIZE)
    return image, label
    
    
def get_training_dataset():
    dataset = load_dataset(TRAINING_FILENAMES, labeled=True)
    dataset = dataset.map(augmentation_pipeline, num_parallel_calls=AUTOTUNE)
    dataset = dataset.repeat()
    dataset = dataset.shuffle(2048)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(AUTOTUNE)
    return dataset
    
def get_validation_dataset(ordered=False):
    dataset = load_dataset(VALID_FILENAMES, labeled=True, ordered=ordered)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.cache()
    dataset = dataset.prefetch(AUTOTUNE)
    return dataset
    
    
def get_test_dataset(ordered=False):
    dataset = load_dataset(TEST_FILENAMES, labeled=False, ordered=ordered)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(AUTOTUNE)
    return dataset
    
def count_data_items(filenames):
    n = [int(re.compile(r"-([0-9]*)\.").search(filename).group(1)) for filename in filenames]
    return np.sum(n)
    
NUM_TRAINING_IMAGES = count_data_items(TRAINING_FILENAMES)
NUM_VALIDATION_IMAGES = count_data_items(VALID_FILENAMES)
NUM_TEST_IMAGES = count_data_items(TEST_FILENAMES)
STEPS_PER_EPOCH = NUM_TRAINING_IMAGES // BATCH_SIZE
print(
    'Dataset: {} training images, {} validation images, {} unlabeled test images'.format(
        NUM_TRAINING_IMAGES, NUM_VALIDATION_IMAGES, NUM_TEST_IMAGES
    )
)

# BUILD MODEL


def exponential_decay(lr0, s):
    def exponential_decay_fn(epoch):
        return lr0 * 0.1 **(epoch / s)
    return exponential_decay_fn

exponential_decay_fn = exponential_decay(0.01, 20)

lr_scheduler = tf.keras.callbacks.LearningRateScheduler(exponential_decay_fn)


train_csv = pd.read_csv('../input/siim-isic-melanoma-classification/train.csv')
test_csv = pd.read_csv('../input/siim-isic-melanoma-classification/test.csv')
    
total_img = train_csv['target'].size

malignant = np.count_nonzero(train_csv['target'])
benign = total_img - malignant

print('Examples:\n    Total: {}\n    Positive: {} ({:.2f}% of total)\n'.format(
    total_img, malignant, 100 * malignant / total_img))    
    
    
train_dataset = get_training_dataset()
valid_dataset = get_validation_dataset()    
    
image_batch, label_batch = next(iter(train_dataset))    
    
benign_imgs = []
benign_lbl = []
malignant_imgs = []
malignant_lbl = []
for i in range(1000):
    image_batch, label_batch = next(iter(valid_dataset))
    index = -1
    for lbl in label_batch:
        index += 1
        if lbl.numpy() == 0:
            benign_imgs.append(image_batch[index])
            benign_lbl.append(label_batch[index])
        elif lbl.numpy() == 1:
            malignant_imgs.append(image_batch[index])
            malignant_lbl.append(label_batch[index])
        #print(lbl.numpy())
benign_lbl, malignant_lbl, tf.shape(benign_imgs), tf.shape(malignant_imgs)

pair_img1 = []
pair_img2 = []
lbl = []
for index in range(0,16, 2):
    if(label_batch[index].numpy() == label_batch[index+1].numpy()):
        lbl.append(tf.convert_to_tensor(1))
    else:
        lbl.append(tf.convert_to_tensor(0))
    pair_img1.append((tf.convert_to_tensor(image_batch[index])))
    pair_img2.append((tf.convert_to_tensor(image_batch[index+1])))



pair_img1 = tf.convert_to_tensor(pair_img1)
pair_img2 = tf.convert_to_tensor(pair_img2)
lbl = tf.convert_to_tensor(lbl)
pair_img = (pair_img1, pair_img2)

len(pair_img[0]), type(lbl)

for lbl in label_batch:
    print(lbl.numpy())
image_batch.shape, label_batch[0]

# specify the shape of the inputs for our network
IMG_SHAPE = (128, 128, 1)
# specify the batch size and number of epochs
#BATCH_SIZE = 64
EPOCHS = 100



import numpy.random as rng
def initialize_weights(shape, dtype=None):
    """
        The paper, http://www.cs.utoronto.ca/~gkoch/files/msc-thesis.pdf
        suggests to initialize CNN layer weights with mean as 0.0 and standard deviation of 0.01
    """
    return np.random.normal(loc = 0.0, scale = 1e-2, size = shape)
def initialize_bias(shape, dtype=None):
    """
        The paper, http://www.cs.utoronto.ca/~gkoch/files/msc-thesis.pdf
        suggests to initialize CNN layer bias with mean as 0.5 and standard deviation of 0.01
    """
    return np.random.normal(loc = 0.5, scale = 1e-2, size = shape)

def get_siamese_model(input_shape):
    """
        Model architecture
    """
    
    # Define the tensors for the two input images
    left_input = Input(input_shape)
    right_input = Input(input_shape)
    
    # Convolutional Neural Network
    model = Sequential()
    model.add(Conv2D(64, (10,10), activation='relu', input_shape=input_shape,
                   kernel_initializer=initialize_weights, kernel_regularizer=l2(2e-4)))
    model.add(MaxPooling2D())
    model.add(Conv2D(128, (7,7), activation='relu',
                     kernel_initializer=initialize_weights,
                     bias_initializer=initialize_bias, kernel_regularizer=l2(2e-4)))
    model.add(MaxPooling2D())
    model.add(Conv2D(128, (4,4), activation='relu', kernel_initializer=initialize_weights,
                     bias_initializer=initialize_bias, kernel_regularizer=l2(2e-4)))
    model.add(MaxPooling2D())
    model.add(Conv2D(256, (4,4), activation='relu', kernel_initializer=initialize_weights,
                     bias_initializer=initialize_bias, kernel_regularizer=l2(2e-4)))
    model.add(Flatten())
    model.add(Dense(4096, activation='sigmoid',
                   kernel_regularizer=l2(1e-3),
                   kernel_initializer=initialize_weights,bias_initializer=initialize_bias))
    
    # Generate the encodings (feature vectors) for the two images
    encoded_l = model(left_input)
    encoded_r = model(right_input)
    
    # Add a customized layer to compute the absolute difference between the encodings
    L1_layer = Lambda(lambda tensors:K.abs(tensors[0] - tensors[1]))
    L1_distance = L1_layer([encoded_l, encoded_r])
    
    # Add a dense layer with a sigmoid unit to generate the similarity score
    prediction = Dense(1,activation='sigmoid',bias_initializer=initialize_bias)(L1_distance)
    
    # Connect the inputs with the outputs
    siamese_net = Model(inputs=[left_input,right_input],outputs=prediction)
    
    # return the model
    return siamese_net


Input((256, 256, 3))

model = get_siamese_model((128, 128, 3))
model.summary()

optimizer = Adam(lr = 0.00006)
model.compile(loss="binary_crossentropy",optimizer=optimizer)

type(label_batch[0:8]), (image_batch[0:8], image_batch[8:16])[1].shape


def get_batch(batch_size = 16):
    image_batch, label_batch = next(iter(train_dataset))
    pair_img1 = []
    pair_img2 = []
    lbl = []
    for index in range(0,batch_size, 2):
        if(label_batch[index].numpy() == label_batch[index+1].numpy()):
            lbl.append(tf.convert_to_tensor(1))
        else:
            lbl.append(tf.convert_to_tensor(0))
        pair_img1.append((tf.convert_to_tensor(image_batch[index])))
        pair_img2.append((tf.convert_to_tensor(image_batch[index+1])))
    pair_img1 = tf.convert_to_tensor(pair_img1)
    pair_img2 = tf.convert_to_tensor(pair_img2)
    lbl = tf.convert_to_tensor(lbl)
    pair_img = (pair_img1, pair_img2)
    return pair_img, lbl

pair_img, lbl = get_batch()
#model.train_on_batch()

tf.shape(pair_img[0])

for img, l in lbl:
    print(l.numpy())

tf.shape(pair_img), tf.shape(lbl)

# Hyper parameters
evaluate_every = 10 # interval for evaluating on one-shot tasks
batch_size = 16
n_iter = 1000 # No. of training iterations
N_way = 20 # how many classes for testing one-shot tasks
n_val = 250 # how many one-shot tasks to validate on
best = -1

for i in range(1000):
    

losses = []
for i in range(1, n_iter+1):
    (inputs,targets) = get_batch(16)
    loss = (model.train_on_batch(inputs, targets))
    if i % evaluate_every == 0:
        print("\n ------------- \n")
        #print("Time for {0} iterations: {1} mins".format(i, (time.time()-t_start)/60.0))
        print("Train Loss: {0}".format(loss)) 
        losses.append(loss)

plt.plot(losses), len(losses), plt.savefig('foo.png')

model.save("my_h5_model1.h5")

model_path = './'

model.save_weights(os.path.join(model_path, 'weights.{}.h5'.format(0)))

import pickle


filename = 'finalized_model.sav'
pickle.dump(model, open(filename, 'wb'))

def show_batch(image_batch, label_batch):
    fig = plt.figure(figsize=(10,10))
    for n in range(16):
        ax = plt.subplot(4,4,n+1)
        plt.imshow(image_batch[n])
        if label_batch[n]:
            plt.title("MALIGNANT")
        else:
            plt.title("BENIGN")
        plt.axis("off")
    fig.savefig('preprocessed_images.png')

image_batch, label_batch = next(iter(train_dataset))
show_batch(image_batch.numpy(), label_batch.numpy())


def make_model(output_bias = None, metrics = None):    
    if output_bias is not None:
        output_bias = tf.keras.initializers.Constant(output_bias)
        
    base_model = tf.keras.applications.Xception(input_shape=(*IMAGE_RESIZE, 3),
                                                include_top=False,
                                                weights='imagenet')
    
    base_model.trainable = False
    
    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(8, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid',
                              bias_initializer=output_bias)
    ])
    
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=metrics)
    
    return model

STEPS_PER_EPOCH = NUM_TRAINING_IMAGES // BATCH_SIZE
VALID_STEPS = NUM_VALIDATION_IMAGES // BATCH_SIZE

initial_bias = np.log([malignant/benign])
initial_bias

weight_for_0 = (1 / benign)*(total_img)/2.0 
weight_for_1 = (1 / malignant)*(total_img)/2.0

class_weight = {0: weight_for_0, 1: weight_for_1}

print('Weight for class 0: {:.2f}'.format(weight_for_0))
print('Weight for class 1: {:.2f}'.format(weight_for_1))

with strategy.scope():
    model = make_model(output_bias = initial_bias, metrics=tf.keras.metrics.AUC(name='auc'))

We can use callbacks to stop training when there are no improvements in our validation set predictions, and this stops overfitting. Additionally, we can save the best weights for our model so that it doesn't have to be retrained. At the end of the training process, the model will restore the weights of its best iteration.

checkpoint_cb = tf.keras.callbacks.ModelCheckpoint("melanoma_model.h5",
                                                    save_best_only=True)

early_stopping_cb = tf.keras.callbacks.EarlyStopping(patience=10,
                                                     restore_best_weights=True)

history = model.fit(
    train_dataset, epochs=100,
    steps_per_epoch=STEPS_PER_EPOCH,
    validation_data=valid_dataset,
    validation_steps=VALID_STEPS,
    callbacks=[checkpoint_cb, early_stopping_cb, lr_scheduler],
    class_weight=class_weight
)

# PREDICTING RESULTS


test_ds = get_test_dataset(ordered=True)

print('Computing predictions...')
test_images_ds = test_ds.map(lambda image, idnum: image)
probabilities = model.predict(test_images_ds)


sub = pd.read_csv('/kaggle/input/siim-isic-melanoma-classification/sample_submission.csv')
sub.head()


print('Generating submission.csv file...')
test_ids_ds = test_ds.map(lambda image, idnum: idnum).unbatch()
test_ids = next(iter(test_ids_ds.batch(NUM_TEST_IMAGES))).numpy().astype('U')

pred_df = pd.DataFrame({'image_name': test_ids, 'target': np.concatenate(probabilities)})
pred_df.head()

del sub['target']
sub = sub.merge(pred_df, on='image_name')
sub.to_csv('submission.csv', index=False)
sub.head()


    
    
