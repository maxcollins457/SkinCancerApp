import numpy as np
from PIL import Image
from werkzeug.utils import secure_filename
import os

# from tensorflow.keras.models import load_model

ALLOWED_EXTENSIONS = ['png', 'jpg', 'jpeg']


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def upload_file(file, return_filepath = False):
    # Create 'uploads' directory if it doesn't exist
    if not os.path.exists('uploads'):
        os.makedirs('uploads')

    filename = secure_filename(file.filename)
    file.save(os.path.join('uploads', filename))
    if return_filepath:
        return os.path.join('uploads', filename)


# model = load_model('app/networks/model.h5')


# def myScaler(x: list, m= 159.88411714650246, s = 46.45448942251337):
#     return (x-m)/s


# def img_to_input(path: str):
#     img = Image.open(path)
#     img = img.resize((100, 75), Image.LANCZOS)
#     return list(img.getdata())


# def make_prediciton(input: list, model = model):
#     scaled_input = myScaler(input)
#     x = (scaled_input).reshape(1, *(75,100,3))
#     prediction = model.predict(x)
#     predicted_class  = np.argmax(prediction)
#     return predicted_class

# def implement_ML(path):
#     input = img_to_input(path)
#     return make_prediciton(input)