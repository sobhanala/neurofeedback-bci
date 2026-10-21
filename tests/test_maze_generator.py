"""Quick test script for maze generator."""
import sys
sys.path.insert(0, '/mnt/c/users/sobhan/desktop/project')

from app.services.maze_generator import MazeGenerator, generate_maze

def test_maze_generation():
    print("Testing Maze Generator...")
    print("=" * 50)
    
    # Test different difficulty levels
    for difficulty in [1, 5, 10]:
        print(f"\nDifficulty {difficulty}:")
        maze_data = generate_maze(difficulty)
        
        print(f"  Size: {maze_data['size']}x{maze_data['size']}")
        print(f"  Start: {maze_data['start']}")
        print(f"  Goal: {maze_data['goal']}")
        
        # Calculate optimal moves
        optimal = MazeGenerator.calculate_optimal_moves(
            maze_data['start'], 
            maze_data['goal']
        )
        print(f"  Optimal moves (Manhattan): {optimal}")
        
        # Verify start and goal are paths (not walls)
        start_x, start_y = maze_data['start']
        goal_x, goal_y = maze_data['goal']
        
        assert not maze_data['maze'][start_y][start_x], "Start position is a wall!"
        assert not maze_data['maze'][goal_y][goal_x], "Goal position is a wall!"
        print(f"  ✓ Start and goal are valid paths")
        
    print("\n" + "=" * 50)
    print("✓ All tests passed!")
    
    # Visual test - print a small maze
    print("\nVisual test (difficulty 3):")
    maze_data = generate_maze(3)
    maze = maze_data['maze']
    start = maze_data['start']
    goal = maze_data['goal']
    
    for y in range(len(maze)):
        for x in range(len(maze[0])):
            if [x, y] == start:
                print('S', end='')  # Start
            elif [x, y] == goal:
                print('G', end='')  # Goal
            elif maze[y][x]:
                print('█', end='')  # Wall
            else:
                print(' ', end='')  # Path
        print()

if __name__ == '__main__':
    test_maze_generation()
