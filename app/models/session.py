from datetime import datetime
from app.extensions import db


class GameSession(db.Model):
    """A single play-session for a user in a specific game."""
    __tablename__ = "game_sessions"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False, index=True)
    game_id = db.Column(db.Integer, db.ForeignKey("games.id"), nullable=False, index=True)
    started_at = db.Column(db.DateTime, default=datetime.utcnow)
    ended_at = db.Column(db.DateTime, nullable=True)
    initial_difficulty = db.Column(db.Integer, nullable=False)
    final_difficulty = db.Column(db.Integer, nullable=True)
    total_trials = db.Column(db.Integer, default=0)
    correct_trials = db.Column(db.Integer, default=0)
    avg_reaction_time_ms = db.Column(db.Float, nullable=True)
    score = db.Column(db.Float, default=0.0)
    dda_algorithm_used = db.Column(db.String(64), nullable=True)
    
    # Maze game specific fields
    maze_data = db.Column(db.JSON, nullable=True)
    moves_count = db.Column(db.Integer, default=0)
    current_position_x = db.Column(db.Integer, nullable=True)
    current_position_y = db.Column(db.Integer, nullable=True)

    # Relationships
    feedback_logs = db.relationship("FeedbackLog", backref="session", lazy="dynamic",
                                    cascade="all, delete-orphan")
    neurofeedback_data = db.relationship("NeuroFeedbackData", backref="session", lazy="dynamic",
                                         cascade="all, delete-orphan")

    @property
    def accuracy(self):
        if self.total_trials == 0:
            return 0.0
        return self.correct_trials / self.total_trials

    @property
    def duration_seconds(self):
        if self.ended_at and self.started_at:
            return (self.ended_at - self.started_at).total_seconds()
        return None

    def __repr__(self):
        return f"<GameSession {self.id} user={self.user_id} game={self.game_id}>"


class FeedbackLog(db.Model):
    """Individual trial/event log within a game session."""
    __tablename__ = "feedback_logs"

    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.Integer, db.ForeignKey("game_sessions.id"), nullable=False, index=True)
    trial_number = db.Column(db.Integer, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    reaction_time_ms = db.Column(db.Float, nullable=True)
    is_correct = db.Column(db.Boolean, nullable=False)
    difficulty_at_trial = db.Column(db.Integer, nullable=False)
    difficulty_after_trial = db.Column(db.Integer, nullable=False)
    stimulus_data = db.Column(db.JSON, nullable=True)
    response_data = db.Column(db.JSON, nullable=True)
    # EEG metrics (nullable for backward compatibility)
    eeg_attention = db.Column(db.Float, nullable=True)
    eeg_relaxation = db.Column(db.Float, nullable=True)
    eeg_cognitive_load = db.Column(db.Float, nullable=True)

    def __repr__(self):
        return f"<FeedbackLog trial={self.trial_number} session={self.session_id}>"
