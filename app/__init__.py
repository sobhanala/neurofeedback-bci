from pathlib import Path
import sys

if __package__ in {None, ""}:
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from flask import Flask, request
from sqlalchemy import text
from config.settings import get_config
from app.extensions import db, login_manager, migrate


def _ensure_user_schema_columns():
    """Best-effort lightweight migration for new experimenter fields."""
    dialect = db.engine.dialect.name
    if dialect == "sqlite":
        table_info = db.session.execute(text("PRAGMA table_info(users)")).fetchall()
        cols = {row[1] for row in table_info}

        if "is_experimenter" not in cols:
            db.session.execute(text("ALTER TABLE users ADD COLUMN is_experimenter BOOLEAN NOT NULL DEFAULT 0"))
        if "experimenter_code" not in cols:
            db.session.execute(text("ALTER TABLE users ADD COLUMN experimenter_code VARCHAR(32)"))
        if "experimenter_id" not in cols:
            db.session.execute(text("ALTER TABLE users ADD COLUMN experimenter_id INTEGER"))
        db.session.commit()

    # Create indexes if possible
    try:
        db.session.execute(text("CREATE INDEX IF NOT EXISTS ix_users_experimenter_code ON users (experimenter_code)"))
    except Exception:
        db.session.rollback()
    try:
        db.session.execute(text("CREATE INDEX IF NOT EXISTS ix_users_experimenter_id ON users (experimenter_id)"))
    except Exception:
        db.session.rollback()
    db.session.commit()


def create_app(config_class=None):
    app = Flask(__name__)

    if config_class is None:
        config_class = get_config()
    app.config.from_object(config_class)

    # Initialize extensions
    db.init_app(app)
    login_manager.init_app(app)
    migrate.init_app(app, db)

    # Register blueprints
    from app.routes.auth import auth_bp
    from app.routes.main import main_bp
    from app.routes.games import games_bp
    from app.routes.admin import admin_bp
    from app.routes.api import api_bp

    app.register_blueprint(auth_bp)
    app.register_blueprint(main_bp)
    app.register_blueprint(games_bp, url_prefix="/games")
    app.register_blueprint(admin_bp, url_prefix="/admin")
    app.register_blueprint(api_bp, url_prefix="/api")

    # Unity WebGL builds use brotli-compressed assets (.br). The browser/Unity
    # loader needs a Content-Encoding: br header to decompress them correctly.
    @app.after_request
    def add_unity_brotli_headers(response):
        if request.path.endswith('.br'):
            response.headers['Content-Encoding'] = 'br'
            if request.path.endswith('.js.br'):
                response.headers['Content-Type'] = 'application/javascript'
            elif request.path.endswith('.wasm.br'):
                response.headers['Content-Type'] = 'application/wasm'
            else:
                response.headers['Content-Type'] = 'application/octet-stream'
        return response

    # User loader
    from app.models.user import User

    @login_manager.user_loader
    def load_user(user_id):
        return User.query.get(int(user_id))

    # Create tables and apply lightweight schema updates
    with app.app_context():
        db.create_all()
        _ensure_user_schema_columns()

    return app
