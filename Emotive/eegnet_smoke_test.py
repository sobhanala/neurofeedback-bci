import numpy as np
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, DepthwiseConv2D, SeparableConv2D
from tensorflow.keras.layers import BatchNormalization, Activation, AveragePooling2D, Dropout, Flatten, Dense, SpatialDropout2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import LabelBinarizer

def EEGNet(nb_classes, Chans=14, Samples=512, 
           dropoutRate=0.5, kernLength=64, F1=8, D=2, F2=16):
    input1 = Input(shape=(Chans, Samples, 1))

    # Block 1
    block1 = Conv2D(F1, (1, kernLength), padding='same', use_bias=False)(input1)
    block1 = BatchNormalization()(block1)
    block1 = DepthwiseConv2D((Chans,1), depth_multiplier=D, use_bias=False, 
                             depthwise_constraint=None, padding='valid')(block1)
    block1 = BatchNormalization()(block1)
    block1 = Activation('elu')(block1)
    block1 = AveragePooling2D((1,4))(block1)
    block1 = Dropout(dropoutRate)(block1)

    # Block 2
    block2 = SeparableConv2D(F2, (1,16), use_bias=False, padding='same')(block1)
    block2 = BatchNormalization()(block2)
    block2 = Activation('elu')(block2)
    block2 = AveragePooling2D((1,8))(block2)
    block2 = Dropout(dropoutRate)(block2)

    flatten = Flatten()(block2)
    dense = Dense(1, activation='sigmoid')(flatten)

    return Model(inputs=input1, outputs=dense)

import numpy as np

X = np.load("dataset/subject_P100/X.npy")
y = np.load("dataset/subject_P100/y.npy")

# CNN format
X = X[..., np.newaxis].astype(np.float32)

# simple train/test split
split = int(len(X) * 0.8)

X_train = X[:split]
X_test = X[split:]

y_train = y[:split].astype(np.float32)
y_test = y[split:].astype(np.float32)

model = EEGNet(
    nb_classes=2,
    Chans=14,
    Samples=512,
    dropoutRate=0.5
)
model.summary()
model.compile(
    loss="binary_crossentropy",
    optimizer="adam",
    metrics=["accuracy"]
)

print("Starting training...")

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

history = model.fit(
    X_train,
    y_train,
    validation_split=0.2,
    epochs=50,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1
)
print(
    "Best training accuracy:",
    max(history.history["accuracy"])
)

print(
    "Best validation accuracy:",
    max(history.history["val_accuracy"])
)

print("Evaluating...")

loss, acc = model.evaluate(
    X_test,
    y_test,
    verbose=1
)

print("Final accuracy:", acc)