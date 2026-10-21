"""Seed script to add maze game to database."""
from app import create_app, db
from app.models.game import Game
from app.models.user import User

app = create_app()

with app.app_context():
    # Check if maze game already exists
    existing_maze = Game.query.filter_by(slug='neurofeedback-maze').first()
    
    if existing_maze:
        print("Maze game already exists. Updating...")
        existing_maze.name = "NeuroFeedback Maze"
        existing_maze.description = "Navigate through procedurally generated mazes using neurofeedback commands. Difficulty adjusts based on performance and brain metrics."
        existing_maze.game_type = "maze"
        existing_maze.dda_algorithm = "eeg_enhanced"
        existing_maze.is_active = True
        existing_maze.min_difficulty = 1
        existing_maze.max_difficulty = 10
        existing_maze.default_difficulty = 5
        db.session.commit()
        print(f"✓ Updated maze game: {existing_maze.name}")
    else:
        # Get admin user
        admin = User.query.filter_by(username='admin').first()
        if not admin:
            print("Warning: Admin user not found. Creating maze game without creator.")
            admin_id = None
        else:
            admin_id = admin.id
        
        # Create maze game
        maze_game = Game(
            name="NeuroFeedback Maze",
            slug="neurofeedback-maze",
            description="Navigate through procedurally generated mazes using neurofeedback commands. Difficulty adjusts based on performance and brain metrics.",
            game_type="maze",
            is_active=True,
            min_difficulty=1,
            max_difficulty=10,
            default_difficulty=5,
            dda_algorithm="eeg_enhanced",
            created_by=admin_id
        )
        
        db.session.add(maze_game)
        db.session.commit()
        
        print(f"✓ Created maze game: {maze_game.name} (ID: {maze_game.id})")

print("\nMaze game seed complete!")
print("\nTo play:")
print("1. Run: python main.py")
print("2. Login and navigate to Games")
print("3. Click 'Play Maze Game'")
print("\nFor neurofeedback integration:")
print("- Session ID stored in localStorage as 'maze_session_id'")
print("- Send commands to: POST /api/maze/command")
print("- Command format: {\"session_id\": <id>, \"action\": \"move_left\", \"confidence\": 0.85, \"eeg_attention\": 0.75, \"eeg_cognitive_load\": 0.5}")
