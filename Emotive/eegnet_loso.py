import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, DepthwiseConv2D, SeparableConv2D
from tensorflow.keras.layers import BatchNormalization, Activation, AveragePooling2D, Dropout, Flatten, Dense, SpatialDropout2D
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import LabelBinarizer

DATASET_ROOT = "dataset"
SUBJECTS = ["P100", "P101"]
EPOCH_LENGTH = 512  # samples
N_CHANNELS = 14

BATCH_SIZE = 16
EPOCHS = 50
DROPOUT_RATE = 0.5
LEARNING_RATE = 0.001

# EEGNet model definition
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

# =====================================
# MAIN LOOP
# =====================================
print("\n" + "="*60)
print("EEGNet LOSO TEST")
print("="*60)

for sub in SUBJECTS:
    print(f"\n===== SUBJECT {sub} =====")

    X = np.load(f"{DATASET_ROOT}/subject_{sub}/X.npy")  # shape: (n_epochs, 14, 512)
    y = np.load(f"{DATASET_ROOT}/subject_{sub}/y.npy")
    groups = np.load(f"{DATASET_ROOT}/subject_{sub}/groups.npy")

    # reshape for CNN input: (epochs, channels, samples, 1)
    X_cnn = X[..., np.newaxis].astype(np.float32)

    # Encode labels
    y_cnn = y.astype(np.float32)

    logo = LeaveOneGroupOut()
    scores = []

    for train_idx, test_idx in logo.split(X_cnn, y, groups):
        X_train, X_test = X_cnn[train_idx], X_cnn[test_idx]
        y_train, y_test = y_cnn[train_idx], y_cnn[test_idx]

        model = EEGNet(nb_classes=2, Chans=N_CHANNELS, Samples=EPOCH_LENGTH,
                       dropoutRate=DROPOUT_RATE)
        model.compile(
            loss='binary_crossentropy',
            optimizer=Adam(LEARNING_RATE),
            metrics=['accuracy']
        )

        model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=0)
        loss, acc = model.evaluate(X_test, y_test, verbose=0)
        scores.append(acc)

    print(f"LOSO-CV Accuracy: {np.mean(scores):.3f} ± {np.std(scores):.3f}")

print("\n" + "="*60)
print("EEGNet TEST COMPLETE")
print("="*60)