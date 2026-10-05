import os.path

from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import numpy
import pickle
from config import directory

class FaceRecognizer:
    def __init__(self, model_dir=None, encoder_dir=None):
        if model_dir and encoder_dir:
            self.label_encoder = self.__load_pickle_object(encoder_dir)
            self.face_recognizer = self.__load_pickle_object(model_dir)
        else:
            self.label_encoder = LabelEncoder()
            self.face_recognizer = SVC(C=1.0, kernel='linear', probability=True)

    def __load_pickle_object(self, object_dir):
        object_file = open(os.path.join(directory, object_dir), 'rb')
        pickle_object = pickle.load(object_file)
        object_file.close()
        return pickle_object

    def __save_pickle_object(self, object_dir, pickle_object):
        object_file = open(os.path.join(directory, object_dir), 'wb')
        pickle.dump(pickle_object, object_file)
        object_file.close()

    def fit(self, vectors, labels):
        labels = self.label_encoder.fit_transform(labels)
        print('Training SVM...')
        self.face_recognizer.fit(vectors, labels)
        print('Finished training SVM...')

    def classify(self, vector):
        predictions = self.face_recognizer.predict_proba([vector])[0]
        max_prediction = numpy.argmax(predictions)
        name = self.label_encoder.classes_[max_prediction]
        return name

    def save(self, model_dir, encoder_dir):
        self.__save_pickle_object(model_dir, self.face_recognizer)
        self.__save_pickle_object(encoder_dir, self.label_encoder)