import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import setup
import parse
import utils
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.activations import relu
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import TerminateOnNaN, EarlyStopping
from sklearn.model_selection import train_test_split, KFold

'''
TODOs
* Try normalizing and check if we are doing it properly
* Try elu and leakyrelu activations
* Try SGD
'''
def create_model():
    # normalizer = tf.keras.layers.Normalization(axis=-1)
    # normalizer.adapt(X)
    model = Sequential(
        [
            # normalizer,
            Input(shape=(setup.N_FEATURES,)),
            Dense(1024, activation=setup.HIDDEN_ACTIVATION, kernel_regularizer=tf.keras.regularizers.l2(setup.REGULARIZATION_RATE)),
            Dense(1024, activation=setup.HIDDEN_ACTIVATION, kernel_regularizer=tf.keras.regularizers.l2(setup.REGULARIZATION_RATE)),
            Dense(1024, activation=setup.HIDDEN_ACTIVATION, kernel_regularizer=tf.keras.regularizers.l2(setup.REGULARIZATION_RATE)),
            Dense(1, activation=setup.OUTPUT_ACTIVATION),
        ]
    )
    model.compile(
        loss=MeanSquaredError(),
        optimizer=Adam(learning_rate=setup.LEARNING_RATE),
        # optimizer=SGD(learning_rate=setup.LEARNING_RATE, nesterov=True, momentum=0.7),
    )

    return model

def train_evaluate_model(model, X_train, y_train, X_cv, y_cv):
    active_callbacks = [TerminateOnNaN()]
    if setup.EARLY_STOPPING:
        # Early stopping on training set loss
        active_callbacks.append(EarlyStopping(monitor="loss", patience=setup.PATIENCE))
        # Early stopping on cross-valiation set loss
        active_callbacks.append(EarlyStopping(monitor="val_loss", patience=setup.PATIENCE))

    history = model.fit(
        X_train, y_train, validation_data=(X_cv, y_cv),
        epochs=setup.EPOCHS, batch_size=setup.BATCH_SIZE, 
        callbacks=active_callbacks
    )

    train_error = model.evaluate(X_train, y_train)
    cv_error = model.evaluate(X_cv, y_cv)
    return history, train_error, cv_error

def create_model_and_train(df_vectorized):
    # Split in training set and cross-validation set (shuffles randomly and splits)
    features = [f"f_{str(x)}" for x in range(1, setup.N_FEATURES+1)]
    X_train, X_cv, y_train, y_cv = train_test_split(df_vectorized[features], df_vectorized["label"], train_size=setup.TRAINING_SET_SIZE)

    model = create_model()
    print(model.summary())
    return (model, train_evaluate_model(model, X_train, y_train, X_cv, y_cv))

#################################
# Train model, discard history
#################################

if __name__ == "__main__":
    df_vectorized = pd.read_csv(setup.DATASET_DIR+setup.DATASET_VECTORIZED, nrows=setup.N_ROWS)
    # Check for null values
    assert df_vectorized[df_vectorized.isnull().values].empty
    # Check shape (+1 is for the label)
    assert df_vectorized.shape == (setup.N_ROWS, setup.N_FEATURES + 1)

    model, _ = create_model_and_train(df_vectorized)
    model.save(setup.MODEL_NAME)