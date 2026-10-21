# Script to show the changes needed in app/routes/api.py for the maze command endpoint

print("""
Update the maze_command function in app/routes/api.py:

Find this section (around line 70-80 of the maze endpoints):

    # Parse action and calculate new position
    action_map = {
        "move_up": (0, -1),
        "move_down": (0, 1),
        "move_left": (-1, 0),
        "move_right": (1, 0)
    }

CHANGE TO:

    # Parse action and calculate new position (no down movement supported)
    action_map = {
        "move_up": (0, -1),
        "move_left": (-1, 0),
        "move_right": (1, 0)
    }
    
This removes the move_down action from the allowed movements.
""")
