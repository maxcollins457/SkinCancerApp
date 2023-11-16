from flask import Blueprint, render_template

eda_bp = Blueprint('EDA', __name__)

@eda_bp.route('/EDA')
def EDA_page():
    return render_template('EDA.html')