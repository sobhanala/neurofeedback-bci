"""
Maze generation service using Recursive Backtracker algorithm.

Generates procedural mazes with difficulty-based size and complexity scaling.
Mazes are designed to be solvable using only up, left, right movements (no down).
The goal is placed at the top row, and start at the bottom row.
"""
import random
from collections import deque


class MazeGenerator:
    """Generates mazes solvable with up/left/right only (no down movement)."""
    
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
        """
        Generate a maze solvable using only up/left/right movements.
        
        Start is placed at the bottom, goal at the top.
        The generation is validated to ensure a path exists using
        only up/left/right moves.
        """
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
                    'maze': self.maze,
                    'start': self.start,
                    'goal': self.goal,
                    'size': self.size
                }
        
        self._force_upward_path()
        return {
            'maze': self.maze,
            'start': self.start,
            'goal': self.goal,
            'size': self.size
        }
    
    def _carve_path(self, x, y):
        """Recursively carve paths using backtracker algorithm."""
        directions = [(0, -2), (2, 0), (0, 2), (-2, 0)]
        random.shuffle(directions)
        
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if (0 < nx < self.size - 1 and 
                0 < ny < self.size - 1 and 
                self.maze[ny][nx]):
                self.maze[ny][nx] = False
                self.maze[y + dy // 2][x + dx // 2] = False
                self._carve_path(nx, ny)
    
    def _is_solvable_no_down(self):
        """Check if maze is solvable using only up/left/right (no down)."""
        start = tuple(self.start)
        goal = tuple(self.goal)
        
        visited = set()
        queue = deque([start])
        visited.add(start)
        
        # Only allow up (0, -1), left (-1, 0), right (1, 0)
        directions = [(0, -1), (-1, 0), (1, 0)]
        
        while queue:
            x, y = queue.popleft()
            
            if (x, y) == tuple(goal):
                return True
            
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if (0 <= nx < self.size and 
                    0 <= ny < self.size and 
                    not self.maze[ny][nx] and
                    (nx, ny) not in visited):
                    visited.add((nx, ny))
                    queue.append((nx, ny))
        
        return False
    
    def _force_upward_path(self):
        """Fallback: carve a guaranteed upward-solvable path."""
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
    
    def to_json(self):
        if not self.maze:
            self.generate()
        return {
            'maze': self.maze,
            'start': self.start,
            'goal': self.goal,
            'size': self.size,
            'difficulty': self.difficulty
        }
    
    @staticmethod
    def calculate_optimal_moves(start, goal):
        return abs(goal[0] - start[0]) + abs(goal[1] - start[1])


def generate_maze(difficulty):
    """Factory function to generate a maze."""
    generator = MazeGenerator(difficulty)
    return generator.generate()
