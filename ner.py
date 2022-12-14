from flask import Blueprint, render_template, url_for, request, session, flash, redirect, jsonify, current_app
from forms import InputURL
from modules.api import ScrapeWebpage
from modules.nlp import NERDocument

ner = Blueprint("ner", __name__, static_folder = "static", template_folder = "templates")

# Routing
@ner.route("/ner_dashboard")
def ner_dashboard():
	return render_template('ner/ner_dashboard_base.html', title = 'NERDoc')

@ner.route("/ner_website")
def ner_website():

	form = InputURL()

	return render_template('ner/ner_website.html', form = form, title = 'NERDoc')

@ner.route("/ner_doc")
def ner_doc():

	return 'Placeholder'

@ner.route("/ner_batch")
def ner_batch():

	return 'Placeholder'

@ner.route("/post_ner_website", methods=['POST'])
def post_ner_website():

	url = request.form.get('url')

	scraper = ScrapeWebpage(url)
	url_text = scraper.text

	doc = NERDocument(url_text)

	if 'PERSON' in doc.ner_extracts.keys():
		return doc.ner_extracts['PERSON'].head(20).to_dict(orient='records')
	else: 
		return 'Sorry, no PERSON data'

	

	# return web_data
	# return 'Successful'
	