from flask import Blueprint, render_template, url_for, request, session, flash, redirect, jsonify, current_app

ocr = Blueprint("ocr", __name__, static_folder = "static", template_folder = "templates")

from forms import SelectFile
from werkzeug.http import dump_header
from werkzeug.utils import secure_filename

# Routing
@ocr.route("/ocr_dashboard")
def ocr_dashboard():
	return render_template('ocr/ocr_dashboard_base.html', title = 'NERDoc')

@ocr.route("/ocr_doc")
def ocr_doc():

	form = SelectFile()

	return render_template('ocr/ocr_doc.html', form = form, title = 'OCR Document')

@ocr.route("/post_ocr_doc", methods = ['POST'])
def post_ocr_doc():
	
	# from modules.document_processing import PdfReader, LinkedIn
	from modules.ocr import TesseractOCR
	
	# profile = request.form.get('profile')
	f = request.files['pdf_file']
	import os
	f.save(os.path.join(current_app.config['UPLOAD_FOLDER'], secure_filename(f.filename)))

	t = os.path.join(current_app.config['UPLOAD_FOLDER'], f.filename)
	session['ocr_filename'] = f.filename.split('.')[0]

	ocr = TesseractOCR()
	ocr.read_pdf(t)

	with open(f'./ocr_output/{f.filename}', 'w', encoding='utf-8') as f: 
		f.write(ocr.fulltext)

	return redirect(url_for('ocr.ocr_complete'))

@ocr.route("/ocr_complete")
def ocr_complete():

	load_filepath = f'./ocr_output/{session["ocr_filename"]}.txt'
	with open(load_filepath, 'r') as f: 
		data = f.readlines()

	return render_template('ocr/ocr_complete.html', ocr_filepath = load_filepath, data = data, title = 'OCR Document')

@ocr.route("/ocr_batch")
def ocr_batch():
	return render_template('ocr/ocr_batch.html', title='NERDoc Batch OCR Processing')