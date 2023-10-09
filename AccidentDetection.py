from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
from tkinter import filedialog
from tkinter.filedialog import askopenfilename
import cv2
import random
import numpy as np
from keras.layers import Activation
import os
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D
from keras.optimizers import Adam
from keras.layers import MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
import winsound
from keras.models import model_from_json
import pickle

main = tkinter.Tk()
main.title("Accident Detection")
main.geometry("1300x1200")

global filename
global classifier

def beep():
    frequency = 2500  # Set Frequency To 2500 Hertz
    duration = 30 # Set Duration To 1000 ms == 1 second
    winsound.Beep(frequency, duration)

names = ['Accident Occured','No Accident Occured']

def uploadDataset():
    global filename
    global labels
    labels = []
    filename = filedialog.askdirectory(initialdir=".")
    pathlabel.config(text=filename)
    text.delete('1.0', END)
    text.insert(END,filename+" loaded\n\n");
    


def trainCNN():
    global classifier
    text.delete('1.0', END)
    
    if os.path.exists('model/model1.json'):
        with open('model/model1.json', "r") as json_file:
            loaded_model_json = json_file.read()
            classifier = model_from_json(loaded_model_json)
        classifier.load_weights("model/model_weights1.h5")
        classifier._make_predict_function()   
        print(classifier.summary())
        f = open('model/history.pckl', 'rb')
        data = pickle.load(f)
        f.close()
        acc = data['accuracy']
        accuracy = acc[39] * 100
        text.insert(END,"CNN Accident Detection Model Prediction Accuracy = "+str(accuracy))
    else:
        classifier = Sequential()
        classifier.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(120,120,3)))
        classifier.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
        classifier.add(MaxPooling2D(pool_size=(2, 2)))
        classifier.add(Dropout(0.25))
        classifier.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
        classifier.add(MaxPooling2D(pool_size=(2, 2)))
        classifier.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
        classifier.add(MaxPooling2D(pool_size=(2, 2)))
        classifier.add(Dropout(0.25))
        classifier.add(Flatten())
        classifier.add(Dense(1024, activation='relu'))
        classifier.add(Dropout(0.5))
        classifier.add(Dense(2, activation='softmax'))
        classifier.compile(loss='categorical_crossentropy',optimizer=Adam(lr=0.0001, decay=1e-6),metrics=['accuracy'])
        models_info = classifier.fit_generator(
        train_generator,
        steps_per_epoch=1800 // 64,
        epochs=40,
        validation_data=validation_generator,
        validation_steps=1800 // 64)

        classifier.save_weights('model/model_weights.h5')
        model_json = classifier.to_json()
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
        text.insert(END,"CNN Accident Detection Model Prediction Accuracy = "+str(accuracy))


    

def webcamPredict():
    videofile = askopenfilename(initialdir = "videos")
    video = cv2.VideoCapture(videofile)
    while(video.isOpened()):
        ret, frame = video.read()
        if ret == True:
            img = frame
            img = cv2.resize(img, (120,120))
            img = img.reshape(1, 120,120, 3)
            #img = np.array(img, dtype='float32')
            #img /= 255
            predict = classifier.predict(img)
            print(np.argmax(predict))
            result = names[np.argmax(predict)]
            cv2.putText(frame, result, (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 0, 255), 2)
            cv2.imshow("video frame", frame)
            if np.argmax(predict) == 0:
                beep()
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break                
        else:
            break
    video.release()
    cv2.destroyAllWindows()

def graph():
    f = open('model/history.pckl', 'rb')
    data = pickle.load(f)
    f.close()

    plt.plot(data['loss'])
    plt.plot(data['val_loss'])
    plt.title('Models Training and Validation loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper right')
    plt.show()

    plt.plot(data['accuracy'])
    plt.plot(data['val_accuracy'])
    plt.title('Models Training and Validation loss')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='lower right')
    plt.show()
        
    
font = ('times', 16, 'bold')
title = Label(main, text='Accident Detection',anchor=W, justify=CENTER)
title.config(bg='yellow4', fg='white')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)


font1 = ('times', 13, 'bold')
upload = Button(main, text="Upload Accident Train & Test Dataset", command=uploadDataset)
upload.place(x=50,y=100)
upload.config(font=font1)  

pathlabel = Label(main)
pathlabel.config(bg='yellow4', fg='white')  
pathlabel.config(font=font1)           
pathlabel.place(x=50,y=150)

markovButton = Button(main, text="Train CNN with Dataset Images", command=trainCNN)
markovButton.place(x=50,y=200)
markovButton.config(font=font1)

predictButton = Button(main, text="Upload Vide & Detect Accident", command=webcamPredict)
predictButton.place(x=50,y=250)
predictButton.config(font=font1)

graphButton = Button(main, text="Loss & Accuracy Graph", command=graph)
graphButton.place(x=50,y=300)
graphButton.config(font=font1)


font1 = ('times', 12, 'bold')
text=Text(main,height=15,width=78)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=450,y=100)
text.config(font=font1)


main.config(bg='magenta3')
main.mainloop()
