import numpy as np
from random import randint as rand

WALL = 0
SPACE = 1


def generate_prims_maze(width=11, height=11, complexity=0.75, density=.75):
    """
    Generate a maze using a modified version of Prims algorithm.
    :param width:
    :param height:
    :param complexity:
    :param density:
    :return: A matrix of floats representing a maze, top left and bottom right corners will always be clear
    """
    # Create only odd shaped islands
    height = (height // 2) * 2 + 1
    width = (width // 2) * 2 + 1
    height = min(height, 9)
    width = min(width, 9)

    # adjust complexity and density in accordance to maze size
    complexity = int(complexity * (5 * (height + width)))
    density = int(density * (height // 2) * (width // 2))
    print("Generating Maze with ajusted complexity and density", complexity, density)

    maze_matrix = np.ones((height, width), dtype=float)

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
    maze_matrix[0:3, 0:3] = maze_matrix[(width - 3):, (height - 3):] = SPACE

    return maze_matrix


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
    print_maze_matrix(generate_prims_maze())
