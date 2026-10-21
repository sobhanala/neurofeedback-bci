from functools import wraps
from flask import Blueprint, render_template, request, redirect, url_for, flash
from flask_login import login_required, current_user
from app.extensions import db
from app.models.user import User
from app.models.game import Game
from app.models.session import GameSession
from app.services.dda_engine import DDA_ALGORITHMS
from sqlalchemy import func
import re
import random

admin_bp = Blueprint("admin", __name__)


def staff_required(func_view):
    """Decorator to restrict access to admin and experimenter users."""
    @wraps(func_view)
    def decorated(*args, **kwargs):
        if not current_user.is_authenticated or not current_user.is_staff:
            flash("Staff access required.", "danger")
            return redirect(url_for("main.dashboard"))
        return func_view(*args, **kwargs)
    return decorated


def admin_only(func_view):
    """Decorator to restrict access to admins only."""
    @wraps(func_view)
    def decorated(*args, **kwargs):
        if not current_user.is_authenticated or not current_user.is_admin:
            flash("Admin access required.", "danger")
            return redirect(url_for("admin.dashboard"))
        return func_view(*args, **kwargs)
    return decorated


def admin_required(func_view):
    """Decorator to restrict access to admins only."""
    @wraps(func_view)
    def decorated(*args, **kwargs):
        if not current_user.is_authenticated or not current_user.is_admin:
            flash("Admin access required.", "danger")
            return redirect(url_for("admin.dashboard"))
        return func_view(*args, **kwargs)
    return decorated


def _scope_user_query(base_query):
    if current_user.is_admin:
        return base_query
    if current_user.is_experimenter:
        return base_query.filter(User.experimenter_id == current_user.id)
    return base_query.filter(User.id == -1)


def _scope_session_query(base_query):
    if current_user.is_admin:
        return base_query
    if current_user.is_experimenter:
        return base_query.join(User, User.id == GameSession.user_id).filter(User.experimenter_id == current_user.id)
    return base_query.filter(GameSession.id == -1)


def _generate_experimenter_code(username):
    base = re.sub(r"[^A-Za-z0-9]", "", (username or "EXP").upper())[:6] or "EXP"
    for _ in range(50):
        candidate = f"EXP-{base}-{random.randint(100, 999)}"
        if not User.query.filter_by(experimenter_code=candidate).first():
            return candidate
    return f"EXP-{base}-{random.randint(1000, 9999)}"


@admin_bp.route("/")
@login_required
@staff_required
def dashboard():
    user_query = _scope_user_query(User.query.filter(User.is_experimenter.is_(False), User.is_admin.is_(False)))
    session_query = _scope_session_query(GameSession.query)

    total_users = user_query.count()
    total_sessions = session_query.count()
    total_games = Game.query.count()
    active_games = Game.query.filter_by(is_active=True).count()

    recent_sessions = session_query.order_by(GameSession.started_at.desc()).limit(20).all()

    # Game-based stats (per game, not an average across games)
    game_stats_query = db.session.query(
        Game.name,
        Game.slug,
        Game.game_type,
        func.count(GameSession.id).label("total_sessions"),
        func.count(func.distinct(GameSession.user_id)).label("unique_users"),
        func.avg(GameSession.score).label("avg_score"),
        func.max(GameSession.score).label("best_score"),
        func.avg(GameSession.avg_reaction_time_ms).label("avg_rt"),
        func.sum(GameSession.total_trials).label("total_trials"),
        func.sum(GameSession.correct_trials).label("total_correct"),
    ).outerjoin(GameSession, Game.id == GameSession.game_id)

    if current_user.is_experimenter and not current_user.is_admin:
        game_stats_query = game_stats_query.outerjoin(User, User.id == GameSession.user_id).filter(
            (User.experimenter_id == current_user.id) | (User.id.is_(None))
        )

    game_stats = []
    for row in game_stats_query.group_by(Game.id).order_by(Game.name).all():
        total_trials = row.total_trials or 0
        total_correct = row.total_correct or 0
        accuracy_pct = round((total_correct / total_trials) * 100, 1) if total_trials else 0.0
        game_stats.append({
            "name": row.name,
            "slug": row.slug,
            "game_type": row.game_type,
            "total_sessions": row.total_sessions or 0,
            "unique_users": row.unique_users or 0,
            "avg_score": round(row.avg_score or 0, 1),
            "best_score": round(row.best_score or 0, 1),
            "avg_rt": round(row.avg_rt or 0, 0),
            "accuracy_pct": accuracy_pct,
        })

    # Per-game advancement over time: average score per game per day,
    # across all users in scope. Shows whether the cohort improves per game.
    date_col = func.date(GameSession.started_at).label("day")
    adv_query = db.session.query(
        Game.name,
        date_col,
        func.avg(GameSession.score).label("avg_score"),
    ).join(Game, GameSession.game_id == Game.id)

    if current_user.is_experimenter and not current_user.is_admin:
        adv_query = adv_query.join(User, User.id == GameSession.user_id).filter(
            User.experimenter_id == current_user.id
        )

    adv_rows = adv_query.group_by(Game.name, date_col).order_by(Game.name, date_col).all()

    per_game_advancement = {}
    for row in adv_rows:
        series = per_game_advancement.setdefault(row.name, {"labels": [], "scores": []})
        series["labels"].append(str(row.day))
        series["scores"].append(round(row.avg_score or 0, 1))

    return render_template(
        "admin/dashboard.html",
        total_users=total_users,
        total_sessions=total_sessions,
        total_games=total_games,
        active_games=active_games,
        recent_sessions=recent_sessions,
        game_stats=game_stats,
        per_game_advancement=per_game_advancement,
    )


@admin_bp.route("/experimenters", methods=["GET", "POST"])
@login_required
@staff_required
@admin_only
def experimenters():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        email = request.form.get("email", "").strip()
        password = request.form.get("password", "")
        code_input = request.form.get("experimenter_code", "").strip().upper()

        if not username or not email or not password:
            flash("Username, email, and password are required.", "danger")
            return redirect(url_for("admin.experimenters"))

        if User.query.filter_by(username=username).first():
            flash("Username already exists.", "danger")
            return redirect(url_for("admin.experimenters"))

        if User.query.filter_by(email=email).first():
            flash("Email already exists.", "danger")
            return redirect(url_for("admin.experimenters"))

        if code_input:
            if User.query.filter_by(experimenter_code=code_input).first():
                flash("Experimenter code already exists.", "danger")
                return redirect(url_for("admin.experimenters"))
            experimenter_code = code_input
        else:
            experimenter_code = _generate_experimenter_code(username)

        experimenter = User(
            username=username,
            email=email,
            is_experimenter=True,
            experimenter_code=experimenter_code,
        )
        experimenter.set_password(password)

        db.session.add(experimenter)
        db.session.commit()
        flash(f"Experimenter '{username}' created with code '{experimenter_code}'.", "success")
        return redirect(url_for("admin.experimenters"))

    all_experimenters = User.query.filter_by(is_experimenter=True).order_by(User.created_at.desc()).all()
    return render_template("admin/experimenters.html", experimenters=all_experimenters)


@admin_bp.route("/experimenters/<int:exp_id>/edit", methods=["GET", "POST"])
@login_required
@staff_required
@admin_only
def edit_experimenter(exp_id):
    """Admin can edit an experimenter's username and/or password."""
    experimenter = User.query.filter_by(id=exp_id, is_experimenter=True).first_or_404()

    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")

        if not username:
            flash("Username is required.", "danger")
            return redirect(url_for("admin.edit_experimenter", exp_id=exp_id))

        existing = User.query.filter(User.username == username, User.id != experimenter.id).first()
        if existing:
            flash("Username already taken.", "danger")
            return redirect(url_for("admin.edit_experimenter", exp_id=exp_id))

        experimenter.username = username
        if password:
            if len(password) < 6:
                flash("Password must be at least 6 characters.", "danger")
                return redirect(url_for("admin.edit_experimenter", exp_id=exp_id))
            experimenter.set_password(password)

        db.session.commit()
        flash(f"Experimenter '{username}' updated.", "success")
        return redirect(url_for("admin.experimenters"))

    return render_template("admin/edit_experimenter.html", experimenter=experimenter)


@admin_bp.route("/users/<int:user_id>/assign", methods=["POST"])
@login_required
@staff_required
@admin_only
def assign_user(user_id):
    """Admin can assign a user to an experimenter, or unassign (empty selection)."""
    user = User.query.filter(
        User.id == user_id, User.is_admin.is_(False), User.is_experimenter.is_(False)
    ).first_or_404()

    exp_id = request.form.get("experimenter_id", "").strip()
    if exp_id:
        experimenter = User.query.filter_by(id=int(exp_id), is_experimenter=True).first()
        if experimenter is None:
            flash("Experimenter not found.", "danger")
            return redirect(url_for("admin.users"))
        user.experimenter_id = experimenter.id
        flash(f"User '{user.username}' assigned to '{experimenter.username}'.", "success")
    else:
        user.experimenter_id = None
        flash(f"User '{user.username}' unassigned.", "success")

    db.session.commit()
    return redirect(url_for("admin.users"))


@admin_bp.route("/users/<int:user_id>/edit", methods=["GET", "POST"])
@login_required
@staff_required
def edit_user(user_id):
    """Admin can edit any user; experimenter can edit users assigned to them."""
    user = _scope_user_query(
        User.query.filter(User.is_admin.is_(False), User.is_experimenter.is_(False))
    ).filter(User.id == user_id).first()
    if user is None:
        flash("User not found in your scope.", "danger")
        return redirect(url_for("admin.users"))

    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")

        if not username:
            flash("Username is required.", "danger")
            return redirect(url_for("admin.edit_user", user_id=user_id))

        existing = User.query.filter(User.username == username, User.id != user.id).first()
        if existing:
            flash("Username already taken.", "danger")
            return redirect(url_for("admin.edit_user", user_id=user_id))

        user.username = username
        if password:
            if len(password) < 6:
                flash("Password must be at least 6 characters.", "danger")
                return redirect(url_for("admin.edit_user", user_id=user_id))
            user.set_password(password)

        db.session.commit()
        flash(f"User '{username}' updated.", "success")
        return redirect(url_for("admin.users"))

    return render_template("admin/edit_user.html", user=user)


@admin_bp.route("/users")
@login_required
@staff_required
def users():
    all_users = _scope_user_query(
        User.query.filter(User.is_admin.is_(False), User.is_experimenter.is_(False))
    ).order_by(User.created_at.desc()).all()

    rows = _scope_session_query(
        db.session.query(
            GameSession.user_id,
            func.count(GameSession.id).label("sessions"),
            func.sum(GameSession.total_trials).label("total_trials"),
            func.sum(GameSession.correct_trials).label("total_correct"),
            func.avg(GameSession.score).label("avg_score"),
            func.avg(GameSession.avg_reaction_time_ms).label("avg_rt"),
        ).group_by(GameSession.user_id)
    ).all()

    user_metrics = {}
    for row in rows:
        total_trials = row.total_trials or 0
        total_correct = row.total_correct or 0
        accuracy_pct = round((total_correct / total_trials) * 100, 1) if total_trials else 0.0
        user_metrics[row.user_id] = {
            "sessions": row.sessions or 0,
            "accuracy_pct": accuracy_pct,
            "avg_score": round(row.avg_score or 0, 2),
            "avg_rt": round(row.avg_rt or 0, 1),
        }

    experimenters = []
    if current_user.is_admin:
        experimenters = User.query.filter_by(is_experimenter=True).order_by(User.username).all()

    return render_template("admin/users.html", users=all_users, user_metrics=user_metrics,
                           experimenters=experimenters)


@admin_bp.route("/users/<int:user_id>")
@login_required
@staff_required
def user_detail(user_id):
    user_query = _scope_user_query(User.query)
    user = user_query.filter(User.id == user_id).first()
    if user is None:
        flash("User not found in your scope.", "danger")
        return redirect(url_for("admin.users"))

    sessions = _scope_session_query(GameSession.query.filter_by(user_id=user_id)).order_by(
        GameSession.started_at.desc()
    ).all()

    game_stats_query = db.session.query(
        Game.name,
        Game.game_type,
        func.count(GameSession.id).label("sessions"),
        func.avg(GameSession.score).label("avg_score"),
        func.max(GameSession.score).label("best_score"),
        func.avg(GameSession.avg_reaction_time_ms).label("avg_rt"),
        func.sum(GameSession.total_trials).label("total_trials"),
        func.sum(GameSession.correct_trials).label("total_correct"),
    ).join(Game, GameSession.game_id == Game.id).filter(GameSession.user_id == user_id)

    if current_user.is_experimenter and not current_user.is_admin:
        game_stats_query = game_stats_query.join(User, User.id == GameSession.user_id).filter(User.experimenter_id == current_user.id)

    game_stats = []
    for row in game_stats_query.group_by(Game.id, Game.name, Game.game_type).order_by(Game.name).all():
        total_trials = row.total_trials or 0
        total_correct = row.total_correct or 0
        accuracy_pct = round((total_correct / total_trials) * 100, 1) if total_trials else 0.0
        game_stats.append({
            "name": row.name,
            "game_type": row.game_type,
            "sessions": row.sessions or 0,
            "avg_score": round(row.avg_score or 0, 1),
            "best_score": round(row.best_score or 0, 1),
            "avg_rt": round(row.avg_rt or 0, 0),
            "accuracy_pct": accuracy_pct,
        })

    # Per-game advancement over time for this specific user.
    advancement_rows = db.session.query(
        Game.name,
        GameSession.started_at,
        GameSession.score,
        GameSession.final_difficulty,
    ).join(Game, GameSession.game_id == Game.id).filter(
        GameSession.user_id == user_id
    ).order_by(Game.name, GameSession.started_at).all()

    if current_user.is_experimenter and not current_user.is_admin:
        advancement_rows = [r for r in advancement_rows]

    per_game_advancement = {}
    for row in advancement_rows:
        series = per_game_advancement.setdefault(row.name, {"labels": [], "scores": []})
        series["labels"].append(row.started_at.strftime("%m/%d %H:%M") if row.started_at else "")
        series["scores"].append(round(row.score or 0, 1))

    return render_template("admin/user_detail.html", user=user, sessions=sessions,
                           game_stats=game_stats, per_game_advancement=per_game_advancement)


@admin_bp.route("/games/<int:game_id>")
@login_required
@staff_required
def game_detail(game_id):
    """Per-user stats for a single game (admin sees all users, experimenter sees own users)."""
    game = Game.query.get_or_404(game_id)

    user_stats_query = db.session.query(
        User.id.label("user_id"),
        User.username,
        func.count(GameSession.id).label("sessions"),
        func.avg(GameSession.score).label("avg_score"),
        func.max(GameSession.score).label("best_score"),
        func.avg(GameSession.avg_reaction_time_ms).label("avg_rt"),
        func.sum(GameSession.total_trials).label("total_trials"),
        func.sum(GameSession.correct_trials).label("total_correct"),
        func.max(GameSession.started_at).label("last_played"),
    ).join(GameSession, GameSession.user_id == User.id).filter(
        GameSession.game_id == game_id
    )

    if current_user.is_experimenter and not current_user.is_admin:
        user_stats_query = user_stats_query.filter(User.experimenter_id == current_user.id)

    user_stats = []
    for row in user_stats_query.group_by(User.id, User.username).order_by(func.count(GameSession.id).desc()).all():
        total_trials = row.total_trials or 0
        total_correct = row.total_correct or 0
        accuracy_pct = round((total_correct / total_trials) * 100, 1) if total_trials else 0.0
        user_stats.append({
            "user_id": row.user_id,
            "username": row.username,
            "sessions": row.sessions or 0,
            "avg_score": round(row.avg_score or 0, 1),
            "best_score": round(row.best_score or 0, 1),
            "avg_rt": round(row.avg_rt or 0, 0),
            "accuracy_pct": accuracy_pct,
            "last_played": row.last_played,
        })

    recent_sessions = _scope_session_query(
        GameSession.query.filter_by(game_id=game_id)
    ).order_by(GameSession.started_at.desc()).limit(20).all()

    return render_template(
        "admin/game_detail.html",
        game=game,
        user_stats=user_stats,
        recent_sessions=recent_sessions,
    )


@admin_bp.route("/games")
@login_required
@staff_required
def games():
    all_games = Game.query.order_by(Game.created_at.desc()).all()
    return render_template("admin/games.html", games=all_games, algorithms=list(DDA_ALGORITHMS.keys()))


# @admin_bp.route("/games/create", methods=["GET", "POST"])
# @login_required
# @staff_required
# def create_game():
#     if request.method == "POST":
#         name = request.form.get("name", "").strip()
#         slug = request.form.get("slug", "").strip().lower().replace(" ", "-")
#         description = request.form.get("description", "")
#         game_type = request.form.get("game_type", "reaction")
#         min_diff = int(request.form.get("min_difficulty", 1))
#         max_diff = int(request.form.get("max_difficulty", 10))
#         default_diff = int(request.form.get("default_difficulty", 5))
#         dda_algo = request.form.get("dda_algorithm", "threshold")

#         if Game.query.filter_by(slug=slug).first():
#             flash("A game with this slug already exists.", "danger")
#             return render_template("admin/create_game.html", algorithms=list(DDA_ALGORITHMS.keys()))

#         game = Game(
#             name=name,
#             slug=slug,
#             description=description,
#             game_type=game_type,
#             min_difficulty=min_diff,
#             max_difficulty=max_diff,
#             default_difficulty=default_diff,
#             dda_algorithm=dda_algo,
#             created_by=current_user.id,
#         )
#         db.session.add(game)
#         db.session.commit()
#         flash(f"Game '{name}' created successfully.", "success")
#         return redirect(url_for("admin.games"))

#     return render_template("admin/create_game.html", algorithms=list(DDA_ALGORITHMS.keys()))


# @admin_bp.route("/games/<int:game_id>/edit", methods=["GET", "POST"])
# @login_required
# @staff_required
# def edit_game(game_id):
#     game = Game.query.get_or_404(game_id)

#     if request.method == "POST":
#         game.name = request.form.get("name", game.name).strip()
#         game.description = request.form.get("description", game.description)
#         game.is_active = request.form.get("is_active") == "on"
#         game.min_difficulty = int(request.form.get("min_difficulty", game.min_difficulty))
#         game.max_difficulty = int(request.form.get("max_difficulty", game.max_difficulty))
#         game.default_difficulty = int(request.form.get("default_difficulty", game.default_difficulty))
#         game.dda_algorithm = request.form.get("dda_algorithm", game.dda_algorithm)
#         db.session.commit()
#         flash("Game updated successfully.", "success")
#         return redirect(url_for("admin.games"))

#     return render_template("admin/edit_game.html", game=game, algorithms=list(DDA_ALGORITHMS.keys()))


@admin_bp.route("/analytics")
@login_required
@staff_required
def analytics():
    from datetime import datetime, timedelta

    thirty_days_ago = datetime.utcnow() - timedelta(days=30)

    daily_sessions = db.session.query(
        func.date(GameSession.started_at).label("date"),
        func.count(GameSession.id).label("count"),
    ).filter(GameSession.started_at >= thirty_days_ago)

    if current_user.is_experimenter and not current_user.is_admin:
        daily_sessions = daily_sessions.join(User, User.id == GameSession.user_id).filter(User.experimenter_id == current_user.id)

    daily_sessions = daily_sessions.group_by(func.date(GameSession.started_at)).order_by(func.date(GameSession.started_at)).all()

    algo_stats = db.session.query(
        GameSession.dda_algorithm_used,
        func.count(GameSession.id).label("sessions"),
        func.avg(GameSession.score).label("avg_score"),
        func.avg(GameSession.avg_reaction_time_ms).label("avg_rt"),
    )

    if current_user.is_experimenter and not current_user.is_admin:
        algo_stats = algo_stats.join(User, User.id == GameSession.user_id).filter(User.experimenter_id == current_user.id)

    algo_stats = algo_stats.group_by(GameSession.dda_algorithm_used).all()

    return render_template("admin/analytics.html", daily_sessions=daily_sessions, algo_stats=algo_stats)

@admin_bp.route('/games/create', methods=['GET', 'POST'])
@login_required
@staff_required
def create_game():
      if request.method == 'POST':
          name = request.form.get('name')
          slug = request.form.get('slug')
          description = request.form.get('description')
          game_type = request.form.get('game_type')
          is_unity_game = request.form.get('is_unity_game') == 'on'
          if game_type == 'shooter':
              is_unity_game = False
          unity_build_path = request.form.get('unity_build_path') if is_unity_game else None
          dda_algorithm = request.form.get('dda_algorithm', 'threshold')
          min_difficulty = int(request.form.get('min_difficulty', 1))
          max_difficulty = int(request.form.get('max_difficulty', 10))
          default_difficulty = int(request.form.get('default_difficulty', 5))

          # Validate
          if Game.query.filter_by(slug=slug).first():
              flash('A game with this slug already exists.', 'error')
              return render_template('admin/create_game.html', algorithms=list(DDA_ALGORITHMS.keys()))

          game = Game(
              name=name,
              slug=slug,
              description=description,
              game_type=game_type,
              is_unity_game=is_unity_game,
              unity_build_path=unity_build_path,
              dda_algorithm=dda_algorithm,
              min_difficulty=min_difficulty,
              max_difficulty=max_difficulty,
              default_difficulty=default_difficulty,
              created_by=current_user.id
          )

          db.session.add(game)
          db.session.commit()

          flash(f'Game "{name}" created successfully!', 'success')
          return redirect(url_for('admin.games'))

      return render_template('admin/create_game.html', algorithms=list(DDA_ALGORITHMS.keys()))


@admin_bp.route('/games/<int:game_id>/edit', methods=['GET', 'POST'])
@login_required
@staff_required
def edit_game(game_id):
    game = Game.query.get_or_404(game_id)

    if request.method == 'POST':
        game.name = request.form.get('name')
        game.description = request.form.get('description')
        game.game_type = request.form.get('game_type')
        game.is_unity_game = request.form.get('is_unity_game') == 'on'
        if game.game_type == 'shooter':
            game.is_unity_game = False
        game.unity_build_path = request.form.get('unity_build_path') if game.is_unity_game else None
        game.dda_algorithm = request.form.get('dda_algorithm')
        game.min_difficulty = int(request.form.get('min_difficulty'))
        game.max_difficulty = int(request.form.get('max_difficulty'))
        game.default_difficulty = int(request.form.get('default_difficulty'))
        game.is_active = request.form.get('is_active') == 'on'

        db.session.commit()

        flash(f'Game "{game.name}" updated successfully!', 'success')
        return redirect(url_for('admin.games'))

    return render_template('admin/edit_game.html', game=game, algorithms=list(DDA_ALGORITHMS.keys()))


@admin_bp.route('/unity/games/create', methods=['GET', 'POST'])
@login_required
@admin_required
def create_unity_game():
    if request.method == 'POST':
        name = request.form.get('name', '').strip()
        slug = request.form.get('slug', '').strip().lower().replace(' ', '-')
        description = request.form.get('description', '')
        unity_build_path = request.form.get('unity_build_path', '').strip()
        dda_algorithm = request.form.get('dda_algorithm', 'eeg_enhanced')
        min_difficulty = int(request.form.get('min_difficulty', 1))
        max_difficulty = int(request.form.get('max_difficulty', 10))
        default_difficulty = int(request.form.get('default_difficulty', 5))

        if not name or not slug:
            flash('Name and slug are required.', 'danger')
            return render_template('admin/create_game.html', algorithms=list(DDA_ALGORITHMS.keys()))

        if Game.query.filter_by(slug=slug).first():
            flash('A game with this slug already exists.', 'danger')
            return render_template('admin/create_game.html', algorithms=list(DDA_ALGORITHMS.keys()))

        game = Game(
            name=name,
            slug=slug,
            description=description,
            game_type='unity',
            is_unity_game=True,
            unity_build_path=unity_build_path,
            dda_algorithm=dda_algorithm,
            min_difficulty=min_difficulty,
            max_difficulty=max_difficulty,
            default_difficulty=default_difficulty,
            created_by=current_user.id,
        )

        db.session.add(game)
        db.session.commit()
        flash(f'Unity game "{name}" created successfully.', 'success')
        return redirect(url_for('admin.games'))

    return render_template('admin/create_game.html', algorithms=list(DDA_ALGORITHMS.keys()))
