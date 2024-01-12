import os
import cv2


class ImageUtils:

    directory = os.path.dirname(__file__)

    def __init__(self):
        pass

    @classmethod
    def get_image(cls, image_dir):
        video = cv2.VideoCapture(os.path.join(ImageUtils.directory, image_dir))
        result, image = video.read()
        return image

    @classmethod
    def save_image(cls, image, image_dir):
        faces_dir = os.path.join(ImageUtils.directory, image_dir)
        try:
            cv2.imwrite(faces_dir, image)
        except:
            pass

    @classmethod
    def draw_rect(cls, image, rect, label=None):
        color = (255, 255, 0)
        thickness = 2
        cv2.rectangle(image, rect, color, thickness, cv2.LINE_AA)
        cv2.putText(image, label, (rect[0] + 2, rect[1] + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    @classmethod
    def crop_image(cls, image, rect):
        start_col, start_row = rect[:2]
        end_col, end_row = start_col + rect[2], start_row + rect[3]
        return image[start_row:end_row, start_col:end_col]

    @classmethod
    def get_image_list(cls, images_dir):
        image_list = os.listdir(os.path.join(ImageUtils.directory, images_dir))
        if '.DS_Store' in image_list:
            image_list.remove('.DS_Store')
        return image_list
