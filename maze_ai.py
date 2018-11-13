from __future__ import print_function
import os, sys, time, datetime, json, random
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers.advanced_activations import PReLU
import matplotlib.pyplot as plt
from prims_maze import generate_prims_maze_matrix
from environment import Maze, Experience

# Exploration factor
EPSILON = 0.1

# Possible actions a player can take
NUMBER_OF_ACTIONS = 4


def show(maze, title="", show=True, file_name=None):
    plt.interactive(False)
    plt.grid(True)
    plt.title(title)
    nrows, ncols = maze.maze.shape
    ax = plt.gca()
    ax.set_xticks(np.arange(0.5, nrows, 1))
    ax.set_yticks(np.arange(0.5, ncols, 1))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    canvas = np.copy(maze.maze)
    for row, col in maze.visited:
        canvas[row, col] = 0.6
    rat_row, rat_col, _ = maze.state
    canvas[rat_row, rat_col] = 0.3  # rat cell
    canvas[nrows - 1, ncols - 1] = 0.9  # cheese cell
    img = plt.imshow(canvas, interpolation='none', cmap='gray')
    if file_name:
        plt.savefig(file_name)
    if show:
        plt.show()
    return img


def play_game(model, maze, player_cell, display=False):
    maze.reset(player_cell)
    envstate = maze.observe()
    while True:
        prev_envstate = envstate
        # get next action
        q = model.predict(prev_envstate)
        action = np.argmax(q[0])
        if display:
            show(maze)

        # apply action, get rewards and new state
        envstate, reward, game_status = maze.act(action)
        if game_status == 'win':
            return True
        elif game_status == 'lose':
            return False


# for small mazes only
def completion_check(model, maze):
    for cell in maze.free_cells:
        if not maze.valid_actions(cell):
            return False
        if not play_game(model, maze, cell):
            return False
    return True


def format_time(seconds):
    if seconds < 400:
        s = float(seconds)
        return "%.1f seconds" % (s,)
    elif seconds < 4000:
        m = seconds / 60.0
        return "%.2f minutes" % (m,)
    else:
        h = seconds / 3600.0
        return "%.2f hours" % (h,)


def build_model(size):
    model = Sequential()
    model.add(Dense(size, input_shape=(size,)))
    model.add(PReLU())
    model.add(Dense(size))
    model.add(PReLU())
    model.add(Dense(NUMBER_OF_ACTIONS))
    model.compile(optimizer='adam', loss='mse')
    return model


def save_model(model, file_name="model"):
    h5file = file_name + ".h5"
    json_file = file_name + ".json"
    model.save_weights(h5file, overwrite=True)
    with open(json_file, "w") as output_file:
        json.dump(model.to_json(), output_file)


def train(model, maze, **opt):
    global EPSILON
    n_epoch = opt.get('n_epoch', 15000)
    max_memory = opt.get('max_memory', 1000)
    data_size = opt.get('data_size', 50)
    start_time = datetime.datetime.now()

    # Initialize experience replay object
    experience = Experience(model, max_memory=max_memory)

    win_history = []  # history of win/lose game
    history_size = maze.maze.size // 2  # history window size
    win_rate = 0.0

    for epoch in range(n_epoch):
        loss = 0.0
        initial_cell = random.choice(maze.free_cells)
        maze.reset(initial_cell)
        game_over = False

        # get initial environment_state (1d flattened canvas)
        environment_state = maze.observe()

        n_episodes = 0
        while not game_over:
            valid_actions = maze.valid_actions()
            if not valid_actions: break
            old_environment_state = environment_state
            # Get next action
            if np.random.rand() < EPSILON:
                action = random.choice(valid_actions)
            else:
                action = np.argmax(experience.predict(old_environment_state))

            # Apply action, get reward and new environment state
            environment_state, reward, game_status = maze.act(action)
            if game_status == 'win':
                win_history.append(1)
                game_over = True
            elif game_status == 'lose':
                win_history.append(0)
                game_over = True
            else:
                game_over = False

            # Store episode (experience)
            episode = [old_environment_state, action, reward, environment_state, game_over]
            experience.remember(episode)
            n_episodes += 1

            # Train neural network model
            inputs, targets = experience.get_data(data_size=data_size)
            model.fit(
                inputs,
                targets,
                epochs=8,
                batch_size=16,
                verbose=0,
            )
            loss = model.evaluate(inputs, targets, verbose=0)

        if len(win_history) > history_size:
            win_rate = sum(win_history[-history_size:]) / history_size

        dt = datetime.datetime.now() - start_time
        t = format_time(dt.total_seconds())
        template = "Epoch: {:03d}/{:d} | Loss: {:.4f} | Episodes: {:d} | Win count: {:d} | Win rate: {:.3f} | time: {}"
        print(template.format(epoch, n_epoch - 1, loss, n_episodes, sum(win_history), win_rate, t))
        # we simply check if training has exhausted all free cells and if in all
        # cases the agent won
        if win_rate > 0.9:
            EPSILON = 0.05
        if sum(win_history[-history_size:]) == history_size and completion_check(model, maze):
            print("Reached 100%% win rate at epoch: %d" % (epoch,))
            break

    dt = datetime.datetime.now() - start_time
    seconds = dt.total_seconds()
    t = format_time(seconds)

    print("n_epoch: %d, max_mem: %d, data: %d, time: %s" % (epoch, max_memory, data_size, t))
    return seconds


def train_interactive():
    regenerate = True
    maze_matrix = None

    while regenerate:
        maze_matrix = generate_prims_maze_matrix()
        environment_maze = Maze(maze_matrix)
        show(environment_maze)
        regenerate = input("Should regenerate?") == "y"
    if input("Should Train?") == "y":
        model = build_model(maze_matrix.size)
        weights_file = input("File to load from (h5)?")
        if weights_file:
            print("loading weights from file: %s" % (weights_file,))
            model.load_weights(weights_file)
        train(model, environment_maze, epochs=1000, max_memory=8 * maze_matrix.size, data_size=32)
        save_model(model, input("Output filename?"))


def train_recurring():
    name = "super_model"
    file_name = "super_model.h5"
    first = True
    index = 0
    path = "TestTwo/"
    name = "TT"
    while True:
        sample_maze = generate_prims_maze_matrix(11, 11)
        environment_maze = Maze(sample_maze)
        show(environment_maze, file_name="%s%s%d" % (path, name, index))
        model = build_model(sample_maze.size)
        if not first:
            model.load_weights(file_name)
            first = False
        train(model, environment_maze, epochs=1000, max_memory=8 * sample_maze.size, data_size=32)
        save_model(model, name)
        index += 1


def test_single(model, maze_width=11):
    maze_matrix = generate_prims_maze_matrix(maze_width)
    maze = Maze(maze_matrix)
    initial_cell = random.choice(maze.free_cells)
    maze.reset(initial_cell)
    return play_game(model, maze, initial_cell), maze


def test_recurring(file_name="TT.h5",
                   runs=100,
                   maze_size=11**2,
                   show_wins=False,
                   show_losses=False,
                   show_all=False):
    if show_all:
        show_losses = show_wins = True

    model = build_model(maze_size)
    model.load_weights(file_name)
    wins = 0
    for i in range(runs):
        print("Test: " + str(i + 1), end="\t")
        won, maze = test_single(model)
        if won:
            wins += 1
            print("🎉 Winner 🎉")
            if show_wins:
                show(maze, "🎉 Winner 🎉")
        else:
            print("💩 Loser 💩")
            if show_losses:
                show(maze, "💩 Loser 💩")

    percentage = ((100 * wins) // runs)
    print("Won: %d Out of: %d (%d%%)" % (wins, runs, percentage))


def test_interactive():
    test_recurring(
        input("File to test?"),
        runs=int(input("Number of runs?")),
        show_wins=True)


if __name__ == '__main__':
    test_interactive()
