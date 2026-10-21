from flask import Blueprint, flash, redirect, render_template, request, jsonify, session as flask_session, url_for
from flask_login import login_required, current_user
from app.extensions import db
from app.models.game import Game
from app.models.session import GameSession, FeedbackLog
from app.services.dda_engine import get_dda_algorithm
from datetime import datetime

games_bp = Blueprint("games", __name__)


@games_bp.route("/")
@login_required
def game_list():
    games = Game.query.filter_by(is_active=True).all()
    return render_template("games/list.html", games=games)


@games_bp.route("/play/<slug>")
@login_required
def play_game(slug):
    game = Game.query.filter_by(slug=slug, is_active=True).first_or_404()

    # Get the last difficulty for this user+game, or use default
    last_session = GameSession.query.filter_by(
        user_id=current_user.id, game_id=game.id
    ).order_by(GameSession.started_at.desc()).first()

    initial_difficulty = last_session.final_difficulty if last_session and last_session.final_difficulty else game.default_difficulty

    # Create a new session
    game_session = GameSession(
        user_id=current_user.id,
        game_id=game.id,
        initial_difficulty=initial_difficulty,
        dda_algorithm_used=game.dda_algorithm,
    )
    db.session.add(game_session)
    db.session.commit()

    if game.is_unity_game:
        return redirect(url_for('games.play_unity_game', slug=game.slug))

    return render_template(
        f"games/{game.game_type}.html",
        game=game,
        game_session=game_session,
        initial_difficulty=initial_difficulty,
    )


@games_bp.route("/submit_trial", methods=["POST"])
@login_required
def submit_trial():
    """Receive a single trial result, log it, and return new difficulty."""
    data = request.get_json()
    session_id = data.get("session_id")
    is_correct = data.get("is_correct", False)
    reaction_time_ms = data.get("reaction_time_ms")
    stimulus_data = data.get("stimulus_data")
    response_data = data.get("response_data")

    game_session = GameSession.query.get_or_404(session_id)
    if game_session.user_id != current_user.id:
        return jsonify({"error": "Unauthorized"}), 403

    game = game_session.game

    # Count existing trials
    trial_count = FeedbackLog.query.filter_by(session_id=session_id).count()
    trial_number = trial_count + 1

    # Get history for DDA
    history = []
    logs = FeedbackLog.query.filter_by(session_id=session_id).order_by(FeedbackLog.trial_number).all()
    for log in logs:
        history.append({
            "is_correct": log.is_correct,
            "reaction_time_ms": log.reaction_time_ms,
            "difficulty_at_trial": log.difficulty_at_trial,
        })
    # Add current trial to history
    current_difficulty = logs[-1].difficulty_after_trial if logs else game_session.initial_difficulty
    history.append({
        "is_correct": is_correct,
        "reaction_time_ms": reaction_time_ms,
        "difficulty_at_trial": current_difficulty,
    })

    # Compute new difficulty
    dda = get_dda_algorithm(
        game.dda_algorithm,
        min_difficulty=game.min_difficulty,
        max_difficulty=game.max_difficulty,
    )
    new_difficulty = dda.compute_new_difficulty(current_difficulty, history)

    # Log the trial
    feedback = FeedbackLog(
        session_id=session_id,
        trial_number=trial_number,
        reaction_time_ms=reaction_time_ms,
        is_correct=is_correct,
        difficulty_at_trial=current_difficulty,
        difficulty_after_trial=new_difficulty,
        stimulus_data=stimulus_data,
        response_data=response_data,
    )
    db.session.add(feedback)

    # Update session counters
    game_session.total_trials = trial_number
    if is_correct:
        game_session.correct_trials = (game_session.correct_trials or 0) + 1

    # Update avg reaction time
    if reaction_time_ms is not None:
        all_rts = [log.reaction_time_ms for log in logs if log.reaction_time_ms is not None]
        all_rts.append(reaction_time_ms)
        game_session.avg_reaction_time_ms = sum(all_rts) / len(all_rts)

    game_session.final_difficulty = new_difficulty
    db.session.commit()

    return jsonify({
        "new_difficulty": new_difficulty,
        "trial_number": trial_number,
        "accuracy": game_session.accuracy,
        "avg_rt": game_session.avg_reaction_time_ms,
    })


@games_bp.route("/end_session", methods=["POST"])
@login_required
def end_session():
    """End a game session."""
    data = request.get_json()
    session_id = data.get("session_id")

    game_session = GameSession.query.get_or_404(session_id)
    if game_session.user_id != current_user.id:
        return jsonify({"error": "Unauthorized"}), 403

    game_session.ended_at = datetime.utcnow()

    # Calculate final score
    accuracy = game_session.accuracy
    avg_rt = game_session.avg_reaction_time_ms or 1000
    difficulty_factor = (game_session.final_difficulty or game_session.initial_difficulty) / 10.0
    game_session.score = round(accuracy * 100 * difficulty_factor * (1000 / max(avg_rt, 100)), 2)

    db.session.commit()

    return jsonify({
        "score": game_session.score,
        "total_trials": game_session.total_trials,
        "correct_trials": game_session.correct_trials,
        "accuracy": round(accuracy * 100, 1),
        "avg_rt": round(avg_rt, 1) if avg_rt else None,
        "final_difficulty": game_session.final_difficulty,
    })
@games_bp.route('/play/unity/<slug>')
@login_required
def play_unity_game(slug):
      """Play a Unity neurofeedback game."""
      game = Game.query.filter_by(slug=slug, is_active=True).first_or_404()

      if not game.is_unity_game:
          flash('This is not a Unity game.', 'error')
          return redirect(url_for('games.game_list'))

      return render_template('games/unity_neurofeedback.html', game=game)

@games_bp.route('/play/maze/<slug>')
@login_required
def play_maze_game(slug):
    """Play a maze neurofeedback game."""
    game = Game.query.filter_by(slug=slug, is_active=True).first_or_404()
    
    if game.game_type != 'maze':
        return jsonify({"error": "Not a maze game"}), 400
    
    # Get last difficulty for this user+game, or use default
    last_session = GameSession.query.filter_by(
        user_id=current_user.id, game_id=game.id
    ).filter(GameSession.ended_at.isnot(None)).order_by(GameSession.started_at.desc()).first()
    
    initial_difficulty = last_session.final_difficulty if last_session and last_session.final_difficulty else game.default_difficulty
    
    return render_template('games/maze.html', game=game, initial_difficulty=initial_difficulty)
