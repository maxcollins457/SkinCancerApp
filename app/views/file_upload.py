from flask import Blueprint, render_template, request, flash, redirect, url_for, session
from app.models import allowed_file, ALLOWED_EXTENSIONS, upload_file
# implement_ML

upload_bp = Blueprint('upload', __name__)

@upload_bp.route('/upload')
def show_upload_form():
    try:
        session['img_filename']
    except:
        session['img_filename'] = None
    img_filename =  'img/' + session['img_filename']
    return render_template('upload.html', 
                           img_filename = img_filename)

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
        # prediction = implement_ML(file_path)
        prediction = 'UNDEFINED'
        flash(f'File uploaded successfully. Prediction: {prediction}', 'alert alert-success')
    else:
        flash(f"Invalid file format. Allowed formats are: {', '.join(ALLOWED_EXTENSIONS)}", 'alert alert-danger')

    return redirect(url_for('upload.show_upload_form'))
