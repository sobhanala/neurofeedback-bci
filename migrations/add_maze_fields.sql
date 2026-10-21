-- Migration: Add maze game fields to game_sessions table
-- Run with: sqlite3 instance/neurofeedback.db < migrations/add_maze_fields.sql

ALTER TABLE game_sessions ADD COLUMN maze_data TEXT;
ALTER TABLE game_sessions ADD COLUMN moves_count INTEGER DEFAULT 0;
ALTER TABLE game_sessions ADD COLUMN current_position_x INTEGER;
ALTER TABLE game_sessions ADD COLUMN current_position_y INTEGER;
