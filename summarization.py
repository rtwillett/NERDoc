from flask import Blueprint, render_template, url_for, request, session, flash, redirect, jsonify, current_app

summ = Blueprint("summ", __name__, static_folder = "static", template_folder = "templates")

# Routing
@summ.route("/summary_dashboard")
def summary_dashboard():
	return render_template('summary/summary_dashboard_base.html', title = 'NERDoc')