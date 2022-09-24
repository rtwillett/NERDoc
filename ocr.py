from flask import Blueprint, render_template, url_for, request, session, flash, redirect, jsonify, current_app

ocr = Blueprint("ocr", __name__, static_folder = "static", template_folder = "templates")

# Routing
@ocr.route("/ocr_dashboard")
def ocr_dashboard():
	return render_template('ocr/ocr_dashboard_base.html', title = 'NERDoc')