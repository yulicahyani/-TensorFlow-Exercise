# =============================================================================
# PROBLEM C2
#
# Create a classifier for the MNIST Handwritten digit dataset.
# The test will expect it to classify 10 classes.
#
# Don't use lambda layers in your model.
#
# Desired accuracy AND validation_accuracy > 91%
# =============================================================================

import tensorflow as tf
from tensorflow.keras.utils import to_categorical

def solution_C2():
    mnist = tf.keras.datasets.mnist

    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    print(X_train.shape)
    print(X_test.shape)

    X_train = X_train.reshape(60000, 28, 28, 1)
    X_test = X_test.reshape(10000, 28, 28, 1)

    y_train_one_hot = to_categorical(y_train)
    y_test_one_hot = to_categorical(y_test)

    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(64, kernel_size=3, activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.Conv2D(32, kernel_size=3, activation='relu'),
        tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy'])

    class myCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs={}):
            if (logs.get('accuracy') > 0.91 and logs.get('val_accuracy') > 0.91 and epoch > 50):
                self.model.stop_training = True

    callbacks = myCallback()

    model.fit(
        X_train,
        y_train_one_hot,
        validation_data=(X_test, y_test_one_hot),
        epochs=100,
        verbose=2,
        callbacks=[callbacks]
    )

    return model


# The code below is to save your model as a .h5 file.
# It will be saved automatically in your Submission folder.
if __name__ == '__main__':
    if __name__ == '__main__':
        model = solution_C2()
        model.save("model_C2.h5")