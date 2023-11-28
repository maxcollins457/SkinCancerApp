import pandas as pd
import seaborn as sns
import numpy as np
from PIL import Image
from werkzeug.utils import secure_filename
import os
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from io import BytesIO
import matplotlib.pyplot as plt
plt.switch_backend('Agg')
import tensorflow as tf

ALLOWED_EXTENSIONS = ['png', 'jpg', 'jpeg']

CNN_model = tf.keras.models.load_model('app/networks/model.h5')
Bin_model = tf.keras.models.load_model('app/networks/binary_model.h5')
Multi_input_model = tf.keras.models.load_model('app/networks/multi_input_model.h5')
VGG_model = tf.keras.models.load_model('app/networks/VGG_model.h5')
MobileNet_model = tf.keras.models.load_model('app/networks/MobileNet.h5')

ML_MODELS = {
    'CNN': {
        'model': CNN_model,
        'inputs': ['img'],
        'codes_dict': {0: 'Actinic keratoses',
                       1: 'Basal cell carcinoma',
                       2: 'Benign keratosis-like lesions ',
                       3: 'Dermatofibroma',
                       4: 'Melanocytic nevi',
                       5: 'Melanoma',
                       6: 'Vascular lesions'}
    },
    'Binary': {
        'model': Bin_model,
        'inputs': ['img'],
        'codes_dict': {0: 'Non-cancerous',
                       1: 'Cancerous'}
    },
    'Multi-input': {
        'model': Multi_input_model,
        'inputs': ['img', 'age', 'localization', 'sex'],
        'cat_dummies': ['localization_acral',
                        'localization_back',
                        'localization_chest',
                        'localization_ear',
                        'localization_face',
                        'localization_foot',
                        'localization_genital',
                        'localization_hand',
                        'localization_lower extremity',
                        'localization_neck',
                        'localization_scalp',
                        'localization_trunk',
                        'localization_unknown',
                        'localization_upper extremity',
                        'sex_male',
                        'sex_unknown'],
        'codes_dict': {0: 'Actinic keratoses',
                       1: 'Basal cell carcinoma',
                       2: 'Benign keratosis-like lesions ',
                       3: 'Dermatofibroma',
                       4: 'Melanocytic nevi',
                       5: 'Melanoma',
                       6: 'Vascular lesions'}
    },
    'VGG': {
        'model': VGG_model,
        'inputs': ['img'],
        'codes_dict': {0: 'Melanocytic nevi',
                       1: 'Melanoma',
                       2: 'Benign keratosis-like lesions ',
                       3: 'Basal cell carcinoma',
                       4: 'Actinic keratoses',
                       5: 'Vascular lesions',
                       6: 'Dermatofibroma'}
    },
    'MobileNet': {
        'model': MobileNet_model,
        'inputs': ['img'],
        'codes_dict': {0: 'Melanocytic nevi',
                       1: 'Melanoma',
                       2: 'Benign keratosis-like lesions ',
                       3: 'Basal cell carcinoma',
                       4: 'Actinic keratoses',
                       5: 'Vascular lesions',
                       6: 'Dermatofibroma'}
    }
}


def clear_temp_directory(app):
    temp_directory = os.path.join(app.root_path, 'static', 'img', 'temp')

    # Check if the directory exists
    if os.path.exists(temp_directory):
        # Loop through files in the directory and remove them
        for filename in os.listdir(temp_directory):
            file_path = os.path.join(temp_directory, filename)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(f"Error removing file {file_path}: {e}")


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def resize_img(image,  dims=(100, 75)):
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


def myScaler(x: list, m=159.88411714650246, s=46.45448942251337):
    return (np.asarray(x)-m)/s


def img_to_input(path: str):
    img = resize_img(path)
    return list(img.getdata())


def multi_input_preprocess(age, sex, local):
    mean_age = 51.863828077927295

    scaled_age = np.asarray(age)/mean_age

    feature_list = ML_MODELS['multi-input']['cat_dummies']
    cat_df = pd.DataFrame(0, index=np.arange(1), columns=feature_list)

    local = 'localization_' + local
    sex = 'sex_' + sex
    for col in cat_df.columns:
        if col == local or col == sex:
            cat_df[col] = 1

    dummies = np.asarray(cat_df)
    scaled_age = scaled_age.reshape(-1, 1)

    x_num = np.concatenate((scaled_age, dummies), axis=1)
    return x_num


def make_prediciton(input: list, model_name='CNN', age=51.863828077927295, sex='male', local='back'):
    model = ML_MODELS[model_name]['model']
    Code_to_cell = ML_MODELS[model_name]['codes_dict']
    scaled_input = myScaler(input)
    x_img = (scaled_input).reshape(1, *(75, 100, 3))

    if model_name == 'Multi-input':
        x_num = multi_input_preprocess(age, sex, local)
        prediction = model.predict(x_num, x_img)
    else:
        prediction = model.predict(x_img)
    prediction_dict = {Code_to_cell.get(
        i): 100*pred for i, pred in enumerate(prediction[0])}
    return prediction_dict


def implement_ML(path, model_name='CNN', age=51.863828077927295, sex='male', local='back'):
    input = img_to_input(path)
    return make_prediciton(input, model_name, age, sex, local)


def generate_pdf_report(class_probabilities):
    # Create a BytesIO buffer to store the PDF
    pdf_buffer = BytesIO()

    # Create a PDF document
    pdf = canvas.Canvas(pdf_buffer, pagesize=letter)

    # Set font and font size
    pdf.setFont("Helvetica", 12)

    # Add content to the PDF
    pdf.drawString(100, 750, "Classification Report")
    pdf.drawString(100, 730, "-----------------------------------")

    # Save the bar chart as an image file
    pdf_chart_buffer = BytesIO()
    generate_seaborn_bar_chart(class_probabilities, pdf_chart_buffer)
    pdf_chart_buffer.seek(0)

    # Create a unique filename for the chart image
    chart_image_filename = 'chart_image.png'

    # Save the chart image to a file
    with open(chart_image_filename, 'wb') as chart_image_file:
        chart_image_file.write(pdf_chart_buffer.read())

    # Draw the chart image onto the PDF
    pdf.drawInlineImage(chart_image_filename, 100, 500, width=400, height=200)

    # Add classification information
    pdf.drawString(100, 480, "Classification Information:")
    y_position = 460

    # Save the PDF to the buffer
    pdf.save()

    # Move the buffer position to the beginning
    pdf_buffer.seek(0)

    return pdf_buffer


def generate_seaborn_bar_chart(class_probabilities, buffer):
    # Create a DataFrame from the class_probabilities dictionary
    data = {'Class': list(class_probabilities.keys()),
            'Probability': list(class_probabilities.values())}
    df = pd.DataFrame(data)

    # Order the DataFrame by probability in descending order
    df = df.sort_values(by='Probability', ascending=False)

    # Create a horizontal bar chart using Seaborn
    plt.switch_backend('Agg')
    sns.set(style="whitegrid")

    plt.figure(figsize=(8, 4))

    # Create the bar plot
    sns.barplot(x='Probability', y='Class', data=df, color='skyblue')

    # Save the Seaborn plot to a BytesIO buffer
    plt.savefig(buffer, format='png', bbox_inches='tight')
    plt.close()


def get_localizations():
    df = pd.read_csv('app/data/CancerData.csv')
    locs = list(df['Localization'].unique())
    return locs
