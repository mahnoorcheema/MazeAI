import numpy as np
from random import randint as rand

WALL = 0
SPACE = 1

MIN_MAZE_SIZE = 9


def generate_prims_maze_matrix(width=11, height=11, complexity=0.75, density=0.75):
    """
    Generate a maze using a modified version of Prims algorithm.
    :param width:
    :param height:
    :param complexity:
    :param density:
    :return: A matrix of floats representing a maze, top left and bottom right corners will always be clear
    """
    # Set the minimum size for the maze
    height = max(height, MIN_MAZE_SIZE)
    width = max(width, MIN_MAZE_SIZE)

    # Create only odd sized mazes
    height = height + 3 if height % 2 == 0 else height + 2
    width = width + 3 if width % 2 == 0 else width + 2

    # Adjust complexity and density in accordance to maze size
    complexity = int(complexity * (5 * (height + width)))
    density = int(density * (height // 2) * (width // 2))

    maze_matrix = np.ones((height, width), dtype=float)

    # Set the borders to be walls
    maze_matrix[0, :] = maze_matrix[-1, :] = WALL
    maze_matrix[:, 0] = maze_matrix[:, -1] = WALL

    for i in range(density):
        x, y = rand(0, width // 2) * 2, rand(0, height // 2) * 2
        maze_matrix[y, x] = WALL
        for j in range(complexity):
            neighbours = []
            if x > 1:
                neighbours.append((y, x - 2))
            if x < width - 2:
                neighbours.append((y, x + 2))
            if y > 1:
                neighbours.append((y - 2, x))
            if y < height - 2:
                neighbours.append((y + 2, x))
            if len(neighbours):
                y_, x_ = neighbours[rand(0, len(neighbours) - 1)]
                if maze_matrix[y_, x_] == SPACE:
                    maze_matrix[y_, x_] = WALL
                    maze_matrix[y_ + (y - y_) // 2, x_ + (x - x_) // 2] = WALL
                    x, y = x_, y_
    # Return the maze with the borders stripped off
    return maze_matrix[1:-1, 1:-1]


def print_maze_matrix(matrix):
    """
    Prints a maze-matrix to the console.
    :param matrix:
    :return:
    """
    for row in matrix:
        for tile in row:
            print("_ " if tile else "[]", end="")
        print()


if __name__ == "__main__":
    for i in range(5):
        print_maze_matrix(generate_prims_maze_matrix())
