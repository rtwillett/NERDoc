from flask import Flask, render_template, url_for, request, session, flash, redirect

# Importing all of the Blueprint objects into the application
from flask_wtf.csrf import CSRFProtect

# from models import User

class Config(object):
	SECRET_KEY = '78w0o5tuuGex5Ktk8VvVDF9Pw3jv1MVE'

app = Flask(__name__)
app.config.from_object(Config)
# app.secret_key = "mastadon"
app.config['UPLOAD_FOLDER'] = './uploads'
app.config['DATA_FOLDER'] = './application_data'

csrf = CSRFProtect(app)

# Routing
@app.route("/")
@app.route("/home")
def landing():
	return render_template('general_templates/landing_page.html', title = 'NERDoc')

# Routing
@app.route("/dashboard")
def dashboard():
	return render_template('general_templates/dashboard.html', title = 'NERDoc')


if __name__ == '__main__':
	app.run(debug = True, threaded = True)