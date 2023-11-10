from flask import Blueprint, render_template

comparison_bp = Blueprint('comparsion', __name__)

@comparison_bp.route('/model-comparison')
def model_comparison_page():
    return render_template('model-comparison.html')