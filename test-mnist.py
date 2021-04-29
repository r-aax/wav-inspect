import keras
import keras.utils
import keras.utils.np_utils
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
import numpy as np

if __name__ == '__main__':
    print('test-mnist : begin')

    # Load.
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Process X.
    x_train = x_train.reshape(60000, 784)
    x_test = x_test.reshape(10000, 784)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    # Process Y.
    y_train = keras.utils.np_utils.to_categorical(y_train, 10)
    y_test = keras.utils.np_utils.to_categorical(y_test, 10)

    # Create model.
    model = Sequential()
    model.add(Dense(512, activation='relu', input_shape=(784,)))
    model.add(Dropout(0.2))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(10, activation='softmax'))

    # Compile model.
    model.compile(loss='categorical_crossentropy',
                  optimizer=RMSprop(),
                  metrics=['accuracy'])

    # Train the model.
    history = model.fit(x_train, y_train,
                        batch_size=128,
                        epochs=20,
                        verbose=1,
                        validation_data=(x_test, y_test))
