from flask import Flask

from app.views.home import home_bp
from app.views.file_upload import upload_bp

# Set up the flask app
app = Flask(__name__)
app.secret_key = 'your_secret_key_here'

app.register_blueprint(home_bp)
app.register_blueprint(upload_bp)