from keras.preprocessing.image import ImageDataGenerator
from keras import layers
from tensorflow.keras.layers import Conv2D
from keras import models
from keras import optimizers
import tensorflow as tf

import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

train_dir = "../BazaDanych/train"
test_dir = '../BazaDanych/test'
validation_dir = '../BazaDanych/val'

model = models.Sequential()
model.add(layers.Conv2D(32, (3,3), activation='relu',
                    input_shape = (100,100,3)))
model.add(layers.MaxPooling2D((2,2)))
model.add(Conv2D(64, (3, 3), activation = 'relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(Conv2D(128, (3, 3), activation = 'relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(Conv2D(128, (3, 3), activation = 'relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Flatten())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(512, activation = 'relu'))
model.add(layers.Dense(24, activation='softmax'))#ilosc klas

model.summary()

model.compile(loss='categorical_crossentropy',
                optimizer=optimizers.RMSprop(learning_rate=1e-4),
                metrics=['acc'])

train_datagen = ImageDataGenerator(rescale=1./255,
                                    rotation_range=40,
                                    width_shift_range=0.2,
                                    height_shift_range=0.2,
                                    shear_range=0.2,
                                    zoom_range=0.2,
                                    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size = (100,100),
    batch_size = 64,
    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
    validation_dir,
    target_size = (100,100),
    batch_size = 64,
    class_mode = 'categorical')

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size = (100,100),
    batch_size = 64,
    class_mode = 'categorical')

history = model.fit(
    train_generator,
    #steps_per_epoch =64,
    epochs = 100,
    validation_data = validation_generator,
    #validation_steps = 10
    verbose=1,
    )


result = model.evaluate(test_generator)

print(result)
model.save('modelsaved')

#zapisanie do tflite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open('model.tflite', 'wb') as f:
  f.write(tflite_model)

#wykresy
import matplotlib.pyplot as plt

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))
 
plt.plot(epochs, acc, 'bo', label='Dokladnosc trenowania')
plt.plot(epochs, val_acc, 'b', label='Dokaldnosc walidacji')
plt.title('Dokladnosc trenowania i walidacji')

plt.legend()
plt.savefig('dokladnosc.png')
plt.figure()

plt.plot(epochs, loss, 'ro', label='Strata trenowania')
plt.plot(epochs, val_loss, 'r', label='Strata walidacji')
plt.title('Strata trenowania i walidacji')
plt.legend()
plt.savefig('strata.png')
plt.show()
