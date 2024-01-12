import os

from config import directory
from config import faces_img_dir
from config import yunet_file
from config import face_svm_encoder
from config import face_svm_file
from face_detection import FaceDetector
from image_utils import ImageUtils
from face_recognizer import FaceRecognizer


def fit_face_recognizer():
    all_features = []
    all_names = []
    faces_dir = os.path.join(directory, faces_img_dir)
    face_detector = FaceDetector(yunet_file)
    face_recognizer = FaceRecognizer()
    face_names = os.listdir(faces_dir)
    if '.DS_Store' in face_names:
        face_names.remove('.DS_Store')
    for face_name in face_names:
        face_dir = os.path.join(faces_dir, face_name)
        for image_file in ImageUtils.get_image_list(face_dir):
            try:
                image = ImageUtils.get_image(os.path.join(face_dir, image_file))
                features = face_detector.get_face_features(image)
                if len(features) > 0:
                    all_features.append(features)
                    all_names.append(face_name)
            except:
                pass
    face_recognizer.fit(all_features, all_names)
    face_recognizer.save(face_svm_file, face_svm_encoder)
    return face_recognizer
