import numpy as np
from PIL import Image
from werkzeug.utils import secure_filename
import os

from tensorflow.keras.models import load_model

ALLOWED_EXTENSIONS = ['png', 'jpg', 'jpeg']

ML_MODELS = {
    ''
}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def resize_img(image,  dims = (100,75)):
    img = Image.open(image)
    return img.resize(dims, Image.LANCZOS)

def upload_file(file, return_filename=False):
    filename = secure_filename(file.filename)
    file_path = os.path.join('app/static/img', filename)

    # Resize the image to 100x75
    resized_image = resize_img(file)

    # Save the resized image
    resized_image.save(file_path)

    if return_filename:
        return filename



model = load_model('app/networks/model.h5')


def myScaler(x: list, m= 159.88411714650246, s = 46.45448942251337):
    return (np.asarray(x)-m)/s


def img_to_input(path: str):
    img = resize_img(path)
    return list(img.getdata())


def make_prediciton(input: list, model = model):
    scaled_input = myScaler(input)
    x = (scaled_input).reshape(1, *(75,100,3))
    prediction = model.predict(x)
    predicted_class  = np.argmax(prediction)
    prob = 100*np.max(prediction)
    prob_str = f'Probability: {prob:.2f}%'
    return predicted_class, prob_str

def implement_ML(path):
    input = img_to_input(path)
    return make_prediciton(input)