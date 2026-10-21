from flask import Blueprint, render_template
from flask_login import login_required, current_user
from app.models.session import GameSession, FeedbackLog
from app.models.game import Game
from app.extensions import db
from sqlalchemy import func

main_bp = Blueprint("main", __name__)


@main_bp.route("/")
def index():
    if current_user.is_authenticated:
        return redirect(url_for("main.dashboard"))
    return render_template("index.html")


@main_bp.route("/dashboard")
@login_required
def dashboard():
    # User performance summary
    total_sessions = GameSession.query.filter_by(user_id=current_user.id).count()
    recent_sessions = GameSession.query.filter_by(user_id=current_user.id).order_by(
        GameSession.started_at.desc()
    ).limit(10).all()

    games = Game.query.filter_by(is_active=True).all()

    # Per-game stats for this user (dashboard is game-based, not an average of games)
    per_game_rows = db.session.query(
        Game.id.label("game_id"),
        Game.name,
        Game.slug,
        Game.game_type,
        Game.is_unity_game,
        func.count(GameSession.id).label("sessions"),
        func.avg(GameSession.score).label("avg_score"),
        func.max(GameSession.score).label("best_score"),
        func.avg(GameSession.avg_reaction_time_ms).label("avg_rt"),
        func.sum(GameSession.total_trials).label("total_trials"),
        func.sum(GameSession.correct_trials).label("total_correct"),
    ).outerjoin(
        GameSession,
        (GameSession.game_id == Game.id) & (GameSession.user_id == current_user.id),
    ).group_by(Game.id).order_by(Game.name).all()

    per_game_stats = []
    for row in per_game_rows:
        total_trials = row.total_trials or 0
        total_correct = row.total_correct or 0
        accuracy_pct = round((total_correct / total_trials) * 100, 1) if total_trials else 0.0
        per_game_stats.append({
            "game_id": row.game_id,
            "name": row.name,
            "slug": row.slug,
            "game_type": row.game_type,
            "is_unity_game": row.is_unity_game,
            "sessions": row.sessions or 0,
            "avg_score": round(row.avg_score or 0, 1),
            "best_score": round(row.best_score or 0, 1),
            "avg_rt": round(row.avg_rt or 0, 0),
            "accuracy_pct": accuracy_pct,
        })

    # Per-game advancement over time (sessions ordered chronologically).
    # Each game becomes one line showing score progression across plays.
    advancement_rows = db.session.query(
        Game.name,
        GameSession.started_at,
        GameSession.score,
        GameSession.final_difficulty,
        GameSession.id,
    ).join(Game, GameSession.game_id == Game.id).filter(
        GameSession.user_id == current_user.id
    ).order_by(Game.name, GameSession.started_at).all()

    per_game_advancement = {}
    for row in advancement_rows:
        series = per_game_advancement.setdefault(row.name, {
            "labels": [],
            "scores": [],
            "difficulties": [],
        })
        series["labels"].append(row.started_at.strftime("%m/%d %H:%M") if row.started_at else "")
        series["scores"].append(round(row.score or 0, 1))
        series["difficulties"].append(row.final_difficulty or 0)

    return render_template("dashboard.html",
                           total_sessions=total_sessions,
                           recent_sessions=recent_sessions,
                           games=games,
                           per_game_stats=per_game_stats,
                           per_game_advancement=per_game_advancement)


@main_bp.route("/profile")
@login_required
def profile():
    sessions = GameSession.query.filter_by(user_id=current_user.id).order_by(
        GameSession.started_at.desc()
    ).all()

    # Per-game stats
    game_stats = db.session.query(
        Game.name,
        func.count(GameSession.id).label("sessions"),
        func.avg(GameSession.score).label("avg_score"),
        func.avg(GameSession.avg_reaction_time_ms).label("avg_rt"),
    ).join(Game, GameSession.game_id == Game.id).filter(
        GameSession.user_id == current_user.id
    ).group_by(Game.name).all()

    return render_template("profile.html", sessions=sessions, game_stats=game_stats)


from flask import redirect, url_for
