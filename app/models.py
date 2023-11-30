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


from config import ALLOWED_EXTENSIONS, ML_MODELS
from app.gradcam import GradCAM

plt.switch_backend('Agg')


def clear_temp_directory(app):
    temp_directory = os.path.join(app.root_path, 'static', 'img', 'temp')
    if os.path.exists(temp_directory):
        for filename in os.listdir(temp_directory):
            file_path = os.path.join(temp_directory, filename)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(f"Error removing file {file_path}: {e}")


def get_localizations():
    df = pd.read_csv('app/data/CancerData.csv')
    locs = list(df['Localization'].unique())
    return locs


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def resize_img(image,  dims=(100, 75)):
    img = Image.open(image)
    return img.resize(dims, Image.LANCZOS)


def upload_file(file, return_filename=False):
    filename = secure_filename(file.filename)
    file_path = os.path.join('app/static/img/temp', filename)
    resized_image = resize_img(file)
    resized_image.save(file_path)
    if return_filename:
        return filename


def myScaler(x: list, m=159.88411714650246, s=46.45448942251337):
    return (np.asarray(x)-m)/s


def img_to_input(path: str):
    img = resize_img(path)
    return list(img.getdata())


def multi_input_preprocess(age: int, sex: str, local:str):
    mean_age = 51.863828077927295
    try:
        age = int(age)
    except Exception as e:
        raise ValueError(f"Error converting age to integer: {e}")
    
    scaled_age = np.asarray(age)/mean_age
    feature_list = ML_MODELS['Multi-input']['cat_dummies']
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
        prediction = model.predict([x_num, x_img])
    else:
        prediction = model.predict(x_img)
    prediction_dict = {Code_to_cell.get(i): 100*pred for i, pred in enumerate(prediction[0])}
    return prediction_dict


def implement_ML(path, model_name='CNN', age=51.863828077927295, sex='male', local='back'):
    input = img_to_input(path)
    return make_prediciton(input, model_name, age, sex, local)


def classification_text(class_probabilities, threshold=30):
    classes = list(class_probabilities.keys())
    probs = list(class_probabilities.values())
    sorted_probs = sorted(zip(probs, classes), key=lambda pair: pair[0], reverse=True)

    max_prob, max_prob_class = sorted_probs[0]
    
    class_text = [f'Most likely class is {max_prob_class} with a certainty of {max_prob:.1f}%']
    
    if max_prob > 70:
        class_text.append('Prediction confidence: High (more than 70%)')
    elif max_prob > 50:
        class_text.append('Prediction confidence: Moderate (50% - 70%)')
    else:
        class_text.append('Prediction confidence: Low (less than 50%)')
        
    if len(classes) > 2:
        likely_classes = [class_label for prob, class_label in sorted_probs[1:] if prob > threshold]
        if len(likely_classes) > 0:
            class_text.append('')
            class_text.append(f'Suggested likely classes with probability > {threshold}% are:')
            class_text.append(f'{", ".join(likely_classes)}')
        class_text.append('')
        class_text.append('Top 3 Classes:')
        for prob, class_label in sorted_probs[:3]:
            class_text.append(f'----->  {class_label}: {prob:.1f}%')

    return class_text


def generate_seaborn_bar_chart(class_probabilities, buffer):
    data = {'Class': list(class_probabilities.keys()),
            'Probability': list(class_probabilities.values())}
    df = pd.DataFrame(data)
    df = df.sort_values(by='Probability', ascending=False)

    plt.switch_backend('Agg')
    sns.set(style="whitegrid")
    plt.figure(figsize=(8, 4))
    sns.barplot(x='Probability', y='Class', data=df, color='skyblue')
    plt.xlim([0,100])
    plt.savefig(buffer, format='png', bbox_inches='tight')
    plt.close()


def add_seaborn_chart_to_pdf(class_probabilities, pdf, pos):
    pdf_chart_buffer = BytesIO()
    generate_seaborn_bar_chart(class_probabilities, pdf_chart_buffer)
    pdf_chart_buffer.seek(0)

    chart_image_filename = 'app/static/img/temp/chart_image.png'

    with open(chart_image_filename, 'wb') as chart_image_file:
        chart_image_file.write(pdf_chart_buffer.read())
    posx, posy = pos
    pdf.drawInlineImage(chart_image_filename, posx, posy, width=440, height=188)


def add_gradcam_chart_to_pdf(model_name, img_path, pdf, pos):
    gradcam = GradCAM(model_name=model_name) 
    pdf_chart_buffer = BytesIO()
    gradcam.write_gradcam_to_buffer(buffer = pdf_chart_buffer, img_path = img_path) 
    pdf_chart_buffer.seek(0)

    chart_image_filename = 'app/static/img/temp/gradcam.png'

    with open(chart_image_filename, 'wb') as chart_image_file:
        chart_image_file.write(pdf_chart_buffer.read())
    posx, posy = pos
    pdf.drawInlineImage(chart_image_filename, posx, posy, width=492, height=398)


def generate_pdf_report(class_probabilities, img_path, model_name = 'VGG'):
    pdf_buffer = BytesIO()
    pdf = canvas.Canvas(pdf_buffer, pagesize=letter)
    pdf.setFont("Helvetica", 12)

    pdf.drawString(100, 750, "Classification Report")
    pdf.drawString(100, 730, "-----------------------------------")

    add_seaborn_chart_to_pdf(class_probabilities, pdf, (100, 500))

    pdf.drawString(100, 480, "Classification Information:")
    classification_lines = classification_text(class_probabilities)

    for i, line in enumerate(classification_lines):
        pdf.drawString(100, 480-20*(i+1), line)

    pdf.showPage()
    pdf.drawString(100, 750, "GradCam")
    pdf.drawString(100, 730, "-----------------------------------")
    add_gradcam_chart_to_pdf(model_name, img_path, pdf, (100, 300))    
    pdf.save()
    pdf_buffer.seek(0)
    return pdf_buffer