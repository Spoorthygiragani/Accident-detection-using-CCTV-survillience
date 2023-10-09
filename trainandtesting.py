import numpy as np
import cv2
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D
from keras.optimizers import Adam
from keras.layers import MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator

from keras.models import model_from_json
import pickle

train_dir = 'data/train'
val_dir = 'data/test'
train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(120,120),
        batch_size=64,
        color_mode="rgb",
        class_mode='categorical')
validation_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=(120,120),
        batch_size=64,
        color_mode="rgb",
        class_mode='categorical')
models = Sequential()

models.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(120,120,3)))
models.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
models.add(MaxPooling2D(pool_size=(2, 2)))
models.add(Dropout(0.25))

models.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
models.add(MaxPooling2D(pool_size=(2, 2)))
models.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
models.add(MaxPooling2D(pool_size=(2, 2)))
models.add(Dropout(0.25))

models.add(Flatten())
models.add(Dense(1024, activation='relu'))
models.add(Dropout(0.5))
models.add(Dense(2, activation='softmax'))

models.compile(loss='categorical_crossentropy',optimizer=Adam(lr=0.0001, decay=1e-6),metrics=['accuracy'])

models_info = models.fit_generator(
        train_generator,
        steps_per_epoch=1800 // 64,
        epochs=40,
        validation_data=validation_generator,
        validation_steps=1800 // 64)

models.save_weights('model/model_weights.h5')
model_json = models.to_json()
with open("model/model.json", "w") as json_file:
    json_file.write(model_json)
f = open('model/history.pckl', 'wb')
pickle.dump(models_info.history, f)
f.close()
f = open('model/history.pckl', 'rb')
data = pickle.load(f)
f.close()
acc = data['accuracy']
accuracy = acc[39] * 100
print("Training Model Accuracy = "+str(accuracy))



plt.plot(models_info.history['loss'])
plt.plot(models_info.history['val_loss'])
plt.title('Models Training and Validation loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper right')
plt.show()

plt.plot(models_info.history['accuracy'])
plt.plot(models_info.history['val_accuracy'])
plt.title('Models Training and Validation loss')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='lower right')
plt.show()
