"""Seed the database with admin/experimenter users and default cognitive games."""
from app import create_app
from app.extensions import db
from app.models.user import User
from app.models.game import Game

app = create_app()

with app.app_context():
    db.create_all()

    # Create admin user if not exists
    admin = User.query.filter_by(username="admin").first()
    if not admin:
        admin = User(username="admin", email="admin@neurofeedback.local", is_admin=True)
        admin.set_password("admin123")
        db.session.add(admin)
        print("Created admin user (username: admin, password: admin123)")

    # Create experimenters
    exp1 = User.query.filter_by(username="exp_ali").first()
    if not exp1:
        exp1 = User(
            username="exp_ali",
            email="exp_ali@neurofeedback.local",
            is_experimenter=True,
            experimenter_code="EXP-ALI-001",
        )
        exp1.set_password("exp123")
        db.session.add(exp1)
        print("Created experimenter: exp_ali (password: exp123, code: EXP-ALI-001)")

    exp2 = User.query.filter_by(username="exp_sara").first()
    if not exp2:
        exp2 = User(
            username="exp_sara",
            email="exp_sara@neurofeedback.local",
            is_experimenter=True,
            experimenter_code="EXP-SARA-001",
        )
        exp2.set_password("exp123")
        db.session.add(exp2)
        print("Created experimenter: exp_sara (password: exp123, code: EXP-SARA-001)")

    db.session.flush()

    # Create regular test users assigned to experimenters
    test_user = User.query.filter_by(username="testuser").first()
    if not test_user:
        test_user = User(username="testuser", email="test@neurofeedback.local", experimenter_id=exp1.id)
        test_user.set_password("test123")
        db.session.add(test_user)
        print("Created test user (username: testuser, password: test123, experimenter: exp_ali)")

    test_user2 = User.query.filter_by(username="testuser2").first()
    if not test_user2:
        test_user2 = User(username="testuser2", email="test2@neurofeedback.local", experimenter_id=exp2.id)
        test_user2.set_password("test123")
        db.session.add(test_user2)
        print("Created test user (username: testuser2, password: test123, experimenter: exp_sara)")

    default_games = [
        {
            "name": "Reaction Speed",
            "slug": "reaction-speed",
            "description": "Test and improve your reaction time. Click as fast as you can when the signal turns green!",
            "game_type": "reaction",
            "dda_algorithm": "threshold",
            "default_difficulty": 5,
        },
        {
            "name": "Pattern Memory",
            "slug": "pattern-memory",
            "description": "Memorize highlighted patterns on a grid and reproduce them from memory.",
            "game_type": "memory",
            "dda_algorithm": "moving_average",
            "default_difficulty": 3,
        },
        {
            "name": "Stroop Challenge",
            "slug": "stroop-challenge",
            "description": "The classic Stroop test - identify the COLOR of the text, ignoring the word itself.",
            "game_type": "stroop",
            "dda_algorithm": "performance_zone",
            "default_difficulty": 4,
        },
        {
            "name": "N-Back Training",
            "slug": "nback-training",
            "description": "Working memory training. Identify when the current letter matches the one N steps ago.",
            "game_type": "nback",
            "dda_algorithm": "elo",
            "default_difficulty": 3,
        },
        {
            "name": "Quick Reactions (Rule-Based)",
            "slug": "quick-reactions-rb",
            "description": "Reaction time game using rule-based DDA for difficulty adjustment.",
            "game_type": "reaction",
            "dda_algorithm": "rule_based",
            "default_difficulty": 5,
        },
    ]

    for game_data in default_games:
        existing = Game.query.filter_by(slug=game_data["slug"]).first()
        if not existing:
            game = Game(**game_data, created_by=1)
            db.session.add(game)
            print(f"Created game: {game_data['name']} (DDA: {game_data['dda_algorithm']})")

    db.session.commit()
    print("\nDatabase seeded successfully!")
    print(f"Total users: {User.query.count()}")
    print(f"Total games: {Game.query.count()}")
