from flask import Blueprint, render_template, redirect, url_for, flash, request
from flask_login import login_user, logout_user, login_required, current_user
from app.extensions import db
from app.models.user import User
from datetime import datetime

auth_bp = Blueprint("auth", __name__)


def _register_context():
    experimenters = User.query.filter_by(is_experimenter=True).order_by(User.username.asc()).all()
    return {"experimenters": experimenters}


@auth_bp.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for("main.dashboard"))

    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")

        user = User.query.filter_by(username=username).first()
        if user and user.check_password(password):
            user.last_login = datetime.utcnow()
            db.session.commit()
            login_user(user)
            next_page = request.args.get("next")
            if user.is_staff:
                return redirect(next_page or url_for("admin.dashboard"))
            return redirect(next_page or url_for("main.dashboard"))
        flash("Invalid username or password.", "danger")

    return render_template("auth/login.html")


@auth_bp.route("/register", methods=["GET", "POST"])
def register():
    if current_user.is_authenticated:
        return redirect(url_for("main.dashboard"))

    if request.method == "POST":
        username = request.form.get("username", "").strip()
        email = request.form.get("email", "").strip()
        password = request.form.get("password", "")
        confirm = request.form.get("confirm_password", "")

        experimenter_id_raw = request.form.get("experimenter_id", "")
        experimenter_code = request.form.get("experimenter_code", "").strip()

        if password != confirm:
            flash("Passwords do not match.", "danger")
            return render_template("auth/register.html", **_register_context())

        if User.query.filter_by(username=username).first():
            flash("Username already exists.", "danger")
            return render_template("auth/register.html", **_register_context())

        if User.query.filter_by(email=email).first():
            flash("Email already registered.", "danger")
            return render_template("auth/register.html", **_register_context())

        experimenter = None

        if experimenter_id_raw:
            try:
                experimenter_id = int(experimenter_id_raw)
                experimenter = User.query.filter_by(id=experimenter_id, is_experimenter=True).first()
            except ValueError:
                experimenter = None

        if experimenter is None and experimenter_code:
            experimenter = User.query.filter_by(is_experimenter=True, experimenter_code=experimenter_code).first()

        if experimenter is None:
            flash("Please select a valid experimenter or enter a valid experimenter code.", "danger")
            return render_template("auth/register.html", **_register_context())

        user = User(username=username, email=email, experimenter_id=experimenter.id)
        user.set_password(password)
        db.session.add(user)
        db.session.commit()

        flash("Registration successful! Please log in.", "success")
        return redirect(url_for("auth.login"))

    return render_template("auth/register.html", **_register_context())


@auth_bp.route("/logout")
@login_required
def logout():
    logout_user()
    flash("You have been logged out.", "info")
    return redirect(url_for("auth.login"))
