from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import UserMixin
from app.extensions import db


class User(UserMixin, db.Model):
    __tablename__ = "users"

    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(64), unique=True, nullable=False, index=True)
    email = db.Column(db.String(120), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(256), nullable=False)

    is_admin = db.Column(db.Boolean, default=False, nullable=False)
    is_experimenter = db.Column(db.Boolean, default=False, nullable=False)
    experimenter_code = db.Column(db.String(32), unique=True, nullable=True, index=True)
    experimenter_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=True, index=True)

    is_active_user = db.Column(db.Boolean, default=True, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_login = db.Column(db.DateTime, nullable=True)

    # Profile fields
    age = db.Column(db.Integer, nullable=True)
    gender = db.Column(db.String(16), nullable=True)

    # Relationships
    sessions = db.relationship("GameSession", backref="user", lazy="dynamic")
    assigned_users = db.relationship(
        "User",
        backref=db.backref("experimenter", remote_side=[id]),
        lazy="dynamic",
        foreign_keys=[experimenter_id],
    )

    @property
    def is_staff(self):
        return self.is_admin or self.is_experimenter

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

    def __repr__(self):
        return f"<User {self.username}>"
