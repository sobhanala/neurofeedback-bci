"""Manual DDA testing script - simulates game sessions with different outcomes."""
from app import create_app, db
from app.models.game import Game
from app.models.session import GameSession
from app.models.user import User
from datetime import datetime, timedelta

app = create_app()

with app.app_context():
    # Find maze game
    maze_game = Game.query.filter_by(slug='neurofeedback-maze').first()
    if not maze_game:
        print("ERROR: Maze game not found. Run seed_maze_game.py first.")
        exit(1)
    
    # Find a test user (use admin or first user)
    user = User.query.filter_by(username='admin').first()
    if not user:
        user = User.query.first()
    
    if not user:
        print("ERROR: No users found. Please create a user first.")
        exit(1)
    
    print(f"\n=== DDA Manual Test for User: {user.username} ===\n")
    
    # Check existing sessions
    existing = GameSession.query.filter_by(
        user_id=user.id, game_id=maze_game.id
    ).filter(GameSession.ended_at.isnot(None)).order_by(GameSession.started_at.desc()).all()
    
    print(f"Found {len(existing)} completed sessions:")
    for s in existing[:5]:
        print(f"  - Session {s.id}: Difficulty {s.initial_difficulty} → {s.final_difficulty}")
    
    # Create test session with specific outcome
    print("\n=== Creating Test Session ===")
    print("Choose outcome:")
    print("1. GOOD performance (should increase difficulty)")
    print("2. BAD performance (should decrease difficulty)")
    print("3. MEDIUM performance (should stay same)")
    print("4. Delete all test sessions")
    
    choice = input("Enter choice (1-4): ").strip()
    
    if choice == "4":
        for s in existing:
            db.session.delete(s)
        db.session.commit()
        print(f"Deleted {len(existing)} sessions.")
        exit(0)
    
    if choice not in ["1", "2", "3"]:
        print("Invalid choice.")
        exit(1)
    
    # Get current difficulty
    last_session = GameSession.query.filter_by(
        user_id=user.id, game_id=maze_game.id
    ).filter(GameSession.ended_at.isnot(None)).order_by(GameSession.started_at.desc()).first()
    
    current_diff = last_session.final_difficulty if last_session and last_session.final_difficulty else maze_game.default_difficulty
    
    print(f"\nCurrent difficulty: {current_diff}")
    
    # Create session
    from app.services.maze_generator import generate_maze
    maze_data = generate_maze(current_diff)
    
    session = GameSession(
        user_id=user.id,
        game_id=maze_game.id,
        initial_difficulty=current_diff,
        dda_algorithm_used=maze_game.dda_algorithm,
        maze_data=maze_data,
        current_position_x=maze_data['start'][0],
        current_position_y=maze_data['start'][1],
        moves_count=0,
        started_at=datetime.utcnow() - timedelta(seconds=30)
    )
    db.session.add(session)
    db.session.commit()
    
    # Simulate different outcomes
    from app.services.maze_generator import MazeGenerator
    optimal_moves = MazeGenerator.calculate_optimal_moves(maze_data['start'], maze_data['goal'])
    
    if choice == "1":
        # Good performance
        session.current_position_x = maze_data['goal'][0]
        session.current_position_y = maze_data['goal'][1]
        session.moves_count = optimal_moves + 2  # Very efficient
        completion_time = 15000  # 15 seconds - fast
        goal_reached = True
        print("Simulating GOOD performance (should increase difficulty)")
    elif choice == "2":
        # Bad performance
        session.current_position_x = maze_data['goal'][0]
        session.current_position_y = maze_data['goal'][1]
        session.moves_count = optimal_moves * 3  # Inefficient
        completion_time = 60000  # 60 seconds - slow
        goal_reached = True
        print("Simulating BAD performance (should decrease difficulty)")
    else:
        # Medium performance
        session.current_position_x = maze_data['goal'][0]
        session.current_position_y = maze_data['goal'][1]
        session.moves_count = int(optimal_moves * 1.5)  # Medium efficiency
        completion_time = 30000  # 30 seconds
        goal_reached = True
        print("Simulating MEDIUM performance (should stay same)")
    
    session.ended_at = datetime.utcnow()
    
    efficiency = optimal_moves / max(session.moves_count, 1)
    
    avg_attention = 0.7 if choice == "1" else 0.4 if choice == "2" else 0.5
    avg_cognitive_load = 0.5
    
    if goal_reached and efficiency > 0.85 and avg_attention > 0.65:
        new_difficulty = min(current_diff + 1, maze_game.max_difficulty)
        dda_reason = "Harder: excellent performance and high attention."
    elif not goal_reached or efficiency < 0.5 or avg_attention < 0.4:
        new_difficulty = max(current_diff - 1, maze_game.min_difficulty)
        dda_reason = "Easier: struggled with completion or low efficiency/attention."
    else:
        new_difficulty = current_diff
        dda_reason = "Same: performance was in target zone."
    
    session.final_difficulty = new_difficulty
    session.total_trials = 1
    session.correct_trials = 1 if goal_reached else 0
    session.avg_reaction_time_ms = completion_time
    session.score = round(efficiency * 100 * (current_diff / 10.0) * (1 if goal_reached else 0.5), 2)
    
    db.session.commit()
    
    print(f"\n=== Results ===")
    print(f"Session ID: {session.id}")
    print(f"Goal Reached: {goal_reached}")
    print(f"Moves: {session.moves_count} / Optimal: {optimal_moves}")
    print(f"Efficiency: {efficiency * 100:.1f}%")
    print(f"Completion Time: {completion_time/1000:.1f}s")
    print(f"Score: {session.score}")
    print(f"Current Difficulty: {current_diff}")
    print(f"New Difficulty: {new_difficulty}")
    print(f"DDA Reason: {dda_reason}")
    print(f"\n✓ Test session created. Now refresh the game page and start a new game.")
    print(f"  The new game should start at difficulty {new_difficulty}")
