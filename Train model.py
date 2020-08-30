
train_path = './img_features_label/'
input_path = './'
label_file = './labels.csv'

casc_file = "haarcascade_frontalface_default.xml"
# casc_file = 'lbpcascade_frontalface.xml'

import cv2
import numpy as np
import os, sys, time
import pandas as pd

img_pixel = (96, 96)

def save_labels(people):
    df = pd.DataFrame(people, columns=['name'])
    df.to_csv(label_file)
    print('Saved labels..')
    print(people)

def get_images(path, size):
    class_id = 0  # target or class of the face
    images, labels= [], []
    people= []

    for subdir in os.listdir(path):
        for image in os.listdir(path + subdir):

            img= cv2.imread(path+os.path.sep+subdir+os.path.sep+image, cv2.IMREAD_GRAYSCALE)
            img= cv2.resize(img, size)

            images.append(np.asarray(img, dtype= np.uint8))
            labels.append(class_id)


        people.append(subdir)
        class_id += 1

    return [images, labels, people]


def train_model(path):
    [images, labels, people] = get_images(train_path, img_pixel) # ksb


    labels = np.asarray(labels, dtype= np.int32)
    print('Total trained images: {}'.format(len(labels)))
    print('Total classes : {}'.format(len(people)))

    # initializing eigen_model and training
    print("\nInitializing FaceRecognizer and training...")
    sttime= time.time()
#     eigen_model= cv2.face.EigenFaceRecognizer_create()
    face_model= cv2.face.LBPHFaceRecognizer_create()
    face_model.train(images, labels)

    print("\nCompleted training in {:.2f} s.\n" .format(time.time()- sttime))

    return [face_model, people]

if __name__== "__main__":
    face_model, people = train_model(train_path)
#     face_model.write('facemodel.xml')
    face_model.write('facemodel.yml')
    print('saved model..')
    save_labels(people)

pd.DataFrame(people)