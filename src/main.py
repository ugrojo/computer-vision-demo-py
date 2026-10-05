import cv2
from config import yunet_file
from config import face_svm_file
from config import face_svm_encoder
from config import media_dir
from face_detection import FaceDetector
from face_recognizer import FaceRecognizer
from image_utils import ImageUtils
from training import fit_face_recognizer

def detect_faces(face_detector):
    video = cv2.VideoCapture(0)
    while True:
        result, image = video.read()
        if result is False:
            cv2.waitKey(0)
            break
        faces = face_detector.detect_faces(image)
        for face in faces:
            # <Recognition>
            name = 'Unknown'
            face_image = ImageUtils.crop_image(image, face)
            face_features = face_detector.get_face_features(face_image)
            if len(face_features) > 0:
                name = face_recognizer.classify(face_features)
            # </Recognition>
            ImageUtils.draw_rect(image, face, name)
        cv2.imshow('J4G Faces', image)
        # If 'q' key is pressed then video capture stops
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    video.release()
    cv2.destroyAllWindows()

def recognize_faces(face_detector, face_recognizer):
    i = 0
    for image_file in ImageUtils.get_image_list(media_dir):
        image = ImageUtils.get_image(media_dir + image_file)
        faces = face_detector.detect_faces(image)
        for face in faces:
            face_image = ImageUtils.crop_image(image, face)
            # ImageUtils.save_image(face_image, faces_img_dir + 'face%d.jpg' % i)
            face_features = face_detector.get_face_features(face_image)
            name = 'Unknown'
            if len(face_features) > 0:
                name = face_recognizer.classify(face_features)
            ImageUtils.draw_rect(image, face, name)
            i += 1
        cv2.imshow('J4G faces', image)
        cv2.waitKey(0)


# face_recognizer = fit_face_recognizer()
face_recognizer = FaceRecognizer(face_svm_file, face_svm_encoder)
face_detector = FaceDetector(yunet_file)

detect_faces(face_detector)
# recognize_faces(face_detector, face_recognizer)
