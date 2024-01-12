import os
import cv2
import face_recognition
import imutils


class FaceDetector:

    def __init__(self, model_dir):
        self.directory = os.path.dirname(__file__)
        self.weights_dir = os.path.join(self.directory, model_dir)
        self.face_detector = cv2.FaceDetectorYN_create(self.weights_dir, "", (0, 0))

    def detect_faces(self, image):
        height, width, _ = image.shape
        self.face_detector.setInputSize((width, height))
        _, faces = self.face_detector.detect(image)
        if faces is not None and len(faces) > 0:
            return [[int(f) for f in face[:4]] for face in faces]
        return []

    def get_face_features(self, face_image):
        try:
            face_image = imutils.resize(face_image, width=600)
            face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
            face_features = face_recognition.face_encodings(face_image)
            if face_features:
                return face_features[0]
        except:
            return []
        return []
