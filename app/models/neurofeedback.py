from datetime import datetime
from app.extensions import db
 
 
class NeuroFeedbackData(db.Model):
     """Stores aggregated EEG metrics for neurofeedback game sessions."""
     __tablename__ = "neurofeedback_data"
 
     id = db.Column(db.Integer, primary_key=True)
     session_id = db.Column(db.Integer, db.ForeignKey("game_sessions.id"), nullable=False, index=True)
     trial_number = db.Column(db.Integer, nullable=False)
     timestamp = db.Column(db.DateTime, default=datetime.utcnow)
     attention_score = db.Column(db.Float, nullable=False)  # 0.0 - 1.0
     relaxation_score = db.Column(db.Float, nullable=False)  # 0.0 - 1.0
     cognitive_load = db.Column(db.Float, nullable=False)  # 0.0 - 1.0
 
     def __repr__(self):
         return f"<NeuroFeedbackData trial={self.trial_number} session={self.session_id}>"
