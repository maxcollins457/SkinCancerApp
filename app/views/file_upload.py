from flask import Blueprint, render_template, request, flash, redirect, url_for
from app.models import allowed_file, ALLOWED_EXTENSIONS, upload_file
# implement_ML

upload_bp = Blueprint('upload', __name__)

@upload_bp.route('/upload')
def show_upload_form():
    return render_template('upload.html')

@upload_bp.route('/uploader', methods=['POST'])
def upload_file_page():
    if 'file' not in request.files:
        flash('No file part', 'error')
        return redirect(url_for('upload.show_upload_form'))

    file = request.files['file']

    if file.filename == '':
        flash('No selected file', 'error')
        return redirect(url_for('upload.show_upload_form'))

    if file and allowed_file(file.filename):
        file_path = upload_file(file, return_filepath = True)

        # prediction = implement_ML(file_path)
        prediction = 'YOU HAVE CANCER'
        flash(f'File uploaded successfully. Prediction: {prediction}', 'alert alert-success')
    else:
        flash(f"Invalid file format. Allowed formats are: {', '.join(ALLOWED_EXTENSIONS)}", 'alert alert-danger')

    return redirect(url_for('upload.show_upload_form'))
