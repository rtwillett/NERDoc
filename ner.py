from flask import Blueprint, render_template, url_for, request, session, flash, redirect, jsonify, current_app

ner = Blueprint("ner", __name__, static_folder = "static", template_folder = "templates")

# Routing
@ner.route("/ner_dashboard")
def ner_dashboard():
	return render_template('ner/ner_dashboard_base.html', title = 'NERDoc')