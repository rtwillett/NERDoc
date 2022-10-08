from flask import Flask, render_template, url_for, request, session, flash, redirect
from ocr import ocr
from ner import ner
from summarization import summ

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

# Registering all Blueprints (makes them available application)
app.register_blueprint(ner, url_prefix="")
app.register_blueprint(ocr, url_prefix="")
app.register_blueprint(summ, url_prefix="")

# Routing
@app.route("/")
@app.route("/home")
def landing():
	return render_template('general_templates/landing_page.html', title = 'NERDoc')

# Routing
@app.route("/dashboard")
def dashboard():
	return render_template('general_templates/dashboard.html', title = 'NERDoc')

@app.route("/dataviz")
def dataviz():
	'''
	This page is not connected to the UI and the page is not pushed because currently it crashes Flask when loaded.
	'''

	import base64
	import seaborn as sns
	import matplotlib.pyplot as plt
	from io import StringIO
	
	img = StringIO()
	sns.set_style("dark")
	
	y = [1,2,3,4,5]
	x = [0,2,1,3,4]

	plt.plot(x,y)
	plt.savefig(img, format='png')
	plt.close()
	img.seek(0)

	plot_url = base64.b64encode(img.getvalue())

	return render_template('general_templates/dataviz.html', plot_url = plot_url, title = 'NERDoc')


if __name__ == '__main__':
	app.run(debug = True, threaded = True)