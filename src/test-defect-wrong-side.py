"""
Test defect Wrong Side of tape.
"""

import keras
import keras.utils
import keras.utils.np_utils
import wav_inspect_nnets as win
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from keras.optimizers import RMSprop

if __name__ == '__main__':
    print('test-defect-wrong-side : begin')
    (x_train, y_train), (x_test, y_test) = win.DefectWrongSide.load_data(defect_files=['wavs/origin/0150.wav'],
                                                                         no_defect_files=['wavs/origin/0151.wav'])

    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

    # Process X.
    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 40.0
    x_test /= 40.0

    # Process Y.
    y_train = keras.utils.np_utils.to_categorical(y_train, 2)
    y_test = keras.utils.np_utils.to_categorical(y_test, 2)

    print(x_train.shape, x_test.shape)

    # Create model.
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=(x_train.shape[1], x_train.shape[2], 1)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(8, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(2, activation='softmax'))

    # Compile model.
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    # Train the model.
    history = model.fit(x_train, y_train,
                        batch_size=8,
                        epochs=20,
                        verbose=1,
                        validation_data=(x_test, y_test))

    print('test-defect-wrong-side : end')
