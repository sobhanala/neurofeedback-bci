"""Seed script to add shooter game to database."""
from app import create_app, db
from app.models.game import Game
from app.models.user import User

app = create_app()

with app.app_context():
    # Check if shooter game already exists
    existing_shooter = Game.query.filter_by(slug='neurofeedback-shooter').first()
    
    if existing_shooter:
        print("Shooter game already exists. Updating...")
        existing_shooter.name = "NeuroFeedback Shooter"
        existing_shooter.description = "Defend the center from waves of enemies using EEG-controlled rotation and shooting. Difficulty increases over time."
        existing_shooter.game_type = "shooter"
        existing_shooter.dda_algorithm = "simple"
        existing_shooter.is_active = True
        existing_shooter.min_difficulty = 1
        existing_shooter.max_difficulty = 10
        existing_shooter.default_difficulty = 1
        db.session.commit()
        print(f"[OK] Updated shooter game: {existing_shooter.name}")
    else:
        # Get admin user
        admin = User.query.filter_by(username='admin').first()
        if not admin:
            print("Warning: Admin user not found. Creating shooter game without creator.")
            admin_id = None
        else:
            admin_id = admin.id
        
        # Create shooter game
        shooter_game = Game(
            name="NeuroFeedback Shooter",
            slug="neurofeedback-shooter",
            description="Defend the center from waves of enemies using EEG-controlled rotation and shooting. Difficulty increases over time.",
            game_type="shooter",
            is_active=True,
            min_difficulty=1,
            max_difficulty=10,
            default_difficulty=1,
            dda_algorithm="simple",
            created_by=admin_id
        )
        
        db.session.add(shooter_game)
        db.session.commit()
        
        print(f"[OK] Created shooter game: {shooter_game.name} (ID: {shooter_game.id})")

print("\nShooter game seed complete!")
print("\nTo play:")
print("1. Run: python main.py")
print("2. Login and navigate to Games")
print("3. Click 'Play Shooter Game'")
print("\nEEG Controls:")
print("- LEFT/RIGHT: Rotate player orientation")
print("- UP (focus on red dot): Shoot bullets")
print("- Commands sent every 5.5 seconds via /api/maze/eeg/command")
print("- Session ID stored in localStorage as 'shooter_session_id'")
