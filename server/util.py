import numpy as np
import json
import joblib
import cv2
import base64
import pywt

__class_to_number = None
__number_to_class = {}
__model = None


def get_b64_test_image_for_virat():
    with open("b64.txt") as f:
        return f.read()


def load_artifacts():
    global __model
    with open("artifacts/saved_model.pkl", "rb") as f:
        __model = joblib.load(f)

    global __class_to_number
    global __number_to_class
    with open("artifacts/class_dictionary.json", "r") as f:
        __class_to_number = json.load(f)
        __number_to_class = {v: k for k, v in __class_to_number.items()}


def get_cv2_image_from_base64(base_64_data):
    encoded_data = base_64_data.split(",")[1]
    nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img


def get_cropped_image(img_path, base_64_data):
    face_cascade = cv2.CascadeClassifier(
        "artifacts/opencv/haarcascade_frontalface_default.xml"
    )
    eyes_cascade = cv2.CascadeClassifier("artifacts/opencv/haarcascade_eye.xml")
    if img_path:
        img = cv2.imread(img_path)
    else:
        img = get_cv2_image_from_base64(base_64_data)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for x, y, w, h in faces:
        roi_color = img[y : y + h, x : x + w]
        roi_gray = gray[y : y + h, x : x + w]

        eyes = eyes_cascade.detectMultiScale(roi_gray)
        if len(eyes) >= 2:
            return roi_color


def wavelet_transform(img, mode="haar", level=1):
    imArray = img
    imArray = cv2.cvtColor(imArray, cv2.COLOR_BGR2GRAY)
    imArray = np.float32(imArray)
    imArray /= 255

    coeffs = pywt.wavedec2(imArray, wavelet=mode, level=level)
    coeffs_H = list(coeffs)
    coeffs_H[0] *= 0

    imArray_H = pywt.waverec2(coeffs_H, wavelet=mode)
    imArray_H *= 255
    imArray_H = np.uint8(imArray_H)

    return imArray_H


def classify_image(image_base64_data, file_path=None):
    img = get_cropped_image(file_path, image_base64_data)
    scaled_img_raw = cv2.resize(img, (32, 32))
    img_har = wavelet_transform(img, "haar", level=5)
    scaled_img_har = cv2.resize(img_har, (32, 32))
    combined_img = np.vstack(
        (scaled_img_raw.reshape(32 * 32 * 3, 1), scaled_img_har.reshape(32 * 32 * 1, 1))
    )
    X = combined_img.reshape(1, 32 * 32 * 3 + 32 * 32).astype(float)
    pred = __model.predict(X)[0]
    return __number_to_class[pred]


if __name__ == "__main__":
    load_artifacts()
    print(classify_image(get_b64_test_image_for_virat()))
