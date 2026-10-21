from datetime import datetime
from app.extensions import db


class Game(db.Model):
    """Represents a neurofeedback game type defined by an admin."""
    __tablename__ = "games"

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(128), unique=True, nullable=False)
    slug = db.Column(db.String(64), unique=True, nullable=False, index=True)
    description = db.Column(db.Text, nullable=True)
    game_type = db.Column(db.String(64), nullable=False)  # reaction, memory, stroop, nback, unity
    is_unity_game = db.Column(db.Boolean, default=False)
    unity_build_path = db.Column(db.String(256), nullable=True)
    is_active = db.Column(db.Boolean, default=True)
    min_difficulty = db.Column(db.Integer, default=1)
    max_difficulty = db.Column(db.Integer, default=10)
    default_difficulty = db.Column(db.Integer, default=5)
    dda_algorithm = db.Column(db.String(64), default="threshold")
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    created_by = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=True)

    # Relationships
    sessions = db.relationship("GameSession", backref="game", lazy="dynamic")

    def __repr__(self):
        return f"<Game {self.name}>"
