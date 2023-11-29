from flask import Blueprint, render_template, request, flash, redirect, url_for, session, make_response
from app.models import allowed_file, ALLOWED_EXTENSIONS, upload_file, implement_ML, generate_pdf_report, get_localizations, ML_MODELS
import os
from io import BytesIO

upload_bp = Blueprint('upload', __name__)

@upload_bp.route('/upload')
def show_upload_form():
    try:
        img_filename =  'img/temp/' + session['img_filename']
    except:
        session['img_filename'] = None
        img_filename = None
    localizations = get_localizations()
    model_names = ML_MODELS.keys()
    return render_template('upload.html', 
                           img_filename = img_filename,
                           localizations=localizations,
                           model_names = model_names)


@upload_bp.route('/uploader', methods=['POST'])
def upload_file_page():
    session['img_path'] = None
    if 'file' not in request.files:
        flash('No file part', 'error')
        return redirect(url_for('upload.show_upload_form'))

    file = request.files['file']

    if file.filename == '':
        flash('No selected file', 'error')
        return redirect(url_for('upload.show_upload_form'))

    if file and allowed_file(file.filename):
        filename = upload_file(file, return_filename = True)
        session['img_filename'] = filename
        model_name = request.form['model']
        age = None
        sex = None
        local = None
        if model_name == 'Multi-input':
            age = request.form['age']
            sex = request.form['sex']
            local = request.form['localization']

        session['ML_inputs'] = (model_name,age, sex, local)

        img_filename = 'img/temp/' + filename
        return render_template('results.html', 
                        img_filename = img_filename)
    else:
        flash(f"Invalid file format. Allowed formats are: {', '.join(ALLOWED_EXTENSIONS)}", 'danger')

    return redirect(url_for('upload.show_upload_form'))



@upload_bp.route('/results')
def results():
    model_name, age, sex, local = session['ML_inputs']
    filename =  session['img_filename'] 
    file_path = os.path.join('app/static/img/temp', filename)
    prediction =  implement_ML(
            file_path, 
            model_name = model_name,
            age = age,
            sex = sex,
            local= local) 
    # Generate the PDF report
    pdf_buffer = generate_pdf_report(prediction, file_path)

    # Move the buffer position to the beginning
    pdf_buffer.seek(0)

    # Send the PDF as a response
    response = make_response(pdf_buffer.getvalue())
    response.headers['Content-Type'] = 'application/pdf'
    response.headers['Content-Disposition'] = 'inline; filename=report.pdf'

    return response


