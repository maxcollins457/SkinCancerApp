import numpy as np
from PIL import Image
from werkzeug.utils import secure_filename
import os

from tensorflow.keras.models import load_model

ALLOWED_EXTENSIONS = ['png', 'jpg', 'jpeg']


CNN_model = load_model('app/networks/model.h5')
# Bin_model = load_model('app/networks/bin_model.h5')
Multi_input_model = load_model('app/networks/multi_input_model.h5')
ML_MODELS = {
    'CNN': {
        'model': CNN_model,
        'inputs': ['img']
    },
    # 'Binary': {
    #     'model': Bin_model,
    #     'inputs': ['img']
    # },
    'Multi-input': {
        'model': Multi_input_model,
        'inputs': ['img', 'age', 'localization', 'sex']
    }
}

Code_to_cell = {0: 'Actinic keratoses',
 1: 'Basal cell carcinoma',
 2: 'Benign keratosis-like lesions ',
 3: 'Dermatofibroma',
 4: 'Melanocytic nevi',
 5: 'Melanoma',
 6: 'Vascular lesions'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def resize_img(image,  dims = (100,75)):
    img = Image.open(image)
    return img.resize(dims, Image.LANCZOS)

def upload_file(file, return_filename=False):
    filename = secure_filename(file.filename)
    file_path = os.path.join('app/static/img/temp', filename)

    # Resize the image to 100x75
    resized_image = resize_img(file)

    # Save the resized image
    resized_image.save(file_path)

    if return_filename:
        return filename


def myScaler(x: list, m= 159.88411714650246, s = 46.45448942251337):
    return (np.asarray(x)-m)/s


def img_to_input(path: str):
    img = resize_img(path)
    return list(img.getdata())


def make_prediciton(input: list, model = CNN_model):
    scaled_input = myScaler(input)
    x = (scaled_input).reshape(1, *(75,100,3))
    prediction = model.predict(x)

    prediction_list = [f'{Code_to_cell.get(i)}: {(100*pred):.2f}% ' for i , pred in enumerate(prediction[0])]

    predicted_class  = Code_to_cell.get(np.argmax(prediction))
    prob = 100*np.max(prediction)
    prob_str = f'Probability: {prob:.2f}%'
    return predicted_class, prob_str, prediction_list

def implement_ML(path):
    input = img_to_input(path)
    return make_prediciton(input)