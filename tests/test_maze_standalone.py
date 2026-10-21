"""Standalone test for no-down maze generator."""
import random
from collections import deque

class MazeGenerator:
    DIFFICULTY_SIZES = {
        1: 7, 2: 7, 3: 9,
        4: 11, 5: 11, 6: 13,
        7: 15, 8: 15,
        9: 19, 10: 21
    }
    
    def __init__(self, difficulty=5):
        self.difficulty = max(1, min(10, difficulty))
        self.size = self.DIFFICULTY_SIZES.get(self.difficulty, 11)
        self.maze = None
        self.start = None
        self.goal = None
    
    def generate(self):
        max_attempts = 50
        for _ in range(max_attempts):
            self.maze = [[True for _ in range(self.size)] for _ in range(self.size)]
            start_x, start_y = 1, 1
            self.maze[start_y][start_x] = False
            self._carve_path(start_x, start_y)
            self.start = [1, self.size - 2]
            self.goal = [self.size - 2, 1]
            self.maze[self.start[1]][self.start[0]] = False
            self.maze[self.goal[1]][self.goal[0]] = False
            if self._is_solvable_no_down():
                return {
                    'maze': self.maze, 'start': self.start,
                    'goal': self.goal, 'size': self.size
                }
        self._force_upward_path()
        return {
            'maze': self.maze, 'start': self.start,
            'goal': self.goal, 'size': self.size
        }
    
    def _carve_path(self, x, y):
        directions = [(0, -2), (2, 0), (0, 2), (-2, 0)]
        random.shuffle(directions)
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if (0 < nx < self.size - 1 and 
                0 < ny < self.size - 1 and self.maze[ny][nx]):
                self.maze[ny][nx] = False
                self.maze[y + dy // 2][x + dx // 2] = False
                self._carve_path(nx, ny)
    
    def _is_solvable_no_down(self):
        start = tuple(self.start)
        goal = tuple(self.goal)
        visited = set()
        queue = deque([start])
        visited.add(start)
        directions = [(0, -1), (-1, 0), (1, 0)]
        while queue:
            x, y = queue.popleft()
            if (x, y) == tuple(goal):
                return True
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if (0 <= nx < self.size and 0 <= ny < self.size and
                    not self.maze[ny][nx] and (nx, ny) not in visited):
                    visited.add((nx, ny))
                    queue.append((nx, ny))
        return False
    
    def _force_upward_path(self):
        sx, sy = self.start
        gx, gy = self.goal
        x, y = sx, sy
        while y > gy:
            self.maze[y][x] = False
            y -= 1
        self.maze[y][x] = False
        while x != gx:
            if x < gx:
                x += 1
            else:
                x -= 1
            self.maze[y][x] = False

# Run tests
print("Testing No-Down Maze Generator")
print("=" * 50)

for difficulty in [1, 3, 5, 7, 10]:
    gen = MazeGenerator(difficulty)
    data = gen.generate()
    
    # Verify solvable with up/left/right only
    solvable = gen._is_solvable_no_down()
    
    print(f"\nDifficulty {difficulty} ({data['size']}x{data['size']}):")
    print(f"  Start (bottom): {data['start']}")
    print(f"  Goal (top):     {data['goal']}")
    print(f"  Solvable (up/left/right only): {'YES' if solvable else 'NO'}")
    assert solvable, f"Maze at difficulty {difficulty} is NOT solvable!"

print("\n" + "=" * 50)
print("All tests passed!")

# Visual test
print("\nVisual (difficulty 2, 7x7):")
gen = MazeGenerator(2)
data = gen.generate()
maze = data['maze']
start = data['start']
goal = data['goal']

for y in range(len(maze)):
    for x in range(len(maze[0])):
        if [x, y] == start:
            print('S', end='')
        elif [x, y] == goal:
            print('G', end='')
        elif maze[y][x]:
            print('#', end='')
        else:
            print(' ', end='')
    print()
print(f"\nS=Start (bottom), G=Goal (top)")
print("Allowed moves: UP, LEFT, RIGHT (no down)")
