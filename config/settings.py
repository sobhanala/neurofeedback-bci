import os


class BaseConfig:
    """Base configuration."""
    SECRET_KEY = os.environ.get("SECRET_KEY", "neurofeedback-dev-secret-key-change-in-prod")
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    JSON_AS_ASCII = False

    # DDA defaults
    DDA_WINDOW_SIZE = 5
    DDA_EASY_THRESHOLD = 0.8
    DDA_HARD_THRESHOLD = 0.4
    DDA_MIN_DIFFICULTY = 1
    DDA_MAX_DIFFICULTY = 10
    DDA_DEFAULT_DIFFICULTY = 5


class DevelopmentConfig(BaseConfig):
    """Development configuration - uses SQLite for easy local setup."""
    DEBUG = True
    SQLALCHEMY_DATABASE_URI = os.environ.get(
        "DATABASE_URL",
        "sqlite:///neurofeedback.db",
    )


class ProductionConfig(BaseConfig):
    """Production configuration - expects PostgreSQL."""
    DEBUG = False
    SQLALCHEMY_DATABASE_URI = os.environ.get(
        "DATABASE_URL",
        "postgresql://neurofeedback:neurofeedback@localhost:5432/neurofeedback",
    )


class TestingConfig(BaseConfig):
    """Testing configuration."""
    TESTING = True
    SQLALCHEMY_DATABASE_URI = "sqlite:///:memory:"


config_by_name = {
    "development": DevelopmentConfig,
    "production": ProductionConfig,
    "testing": TestingConfig,
}


def get_config():
    """Return the configuration class based on the FLASK_ENV env var."""
    env = os.environ.get("FLASK_ENV", "development")
    return config_by_name.get(env, DevelopmentConfig)
