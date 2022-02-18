"""
Othello is played on the 8x8 board.

Each game starts with 4 pieces already placed on the board:
    blacks: e4 and d5
    whites: d4 and e5

Blacks move first.

Black must place a black disc on the board, in such a way that there is at least one straight (horizontal, vertical,
or diagonal) occupied line between the new disc and another black disc, with one or more contiguous white pieces
between them.

After placing the disc, Black flips all white discs lying on a straight line between the new disc and any existing
black discs. All flipped discs are now black.

Players alternate taking turns. If a player does not have any valid moves, play passes back to the other player.
When neither player can move, the game ends. A game of Othello may end before the board is completely filled.

The player with the most discs on the board at the end of the game wins. If both players have the same number of discs,
then the game is a draw.
"""
# Imports
import numpy as np
import dill as pickle
import csv
import tqdm


class OthelloStatesGenerator:
    """
    Generates game states based on sequence of moves.
    """

    def __init__(self):
        """
        Some info:
            State is represented as a numpy array where 1 represents blacks, -1 whites and 0 empty field.

            The flattened array has contains the following:
                winner, n-th move, flattened states
        """
        self.board_size = 8  # board_size x board_size
        self.state = np.zeros(shape=(self.board_size, self.board_size), dtype=int)
        self.states_flattened = None

    def generate_states(self, winner: int, moves: str, game_id: str):
        """
        Generates game states based on the provided moves.
        The states are saved in a flattened state into a numpy array

        Args:
            winner: int, winner of the game, 1 for blacks, -1 for whites, 0 for draw
            moves: string, sequence of moves
            game_id: id of the game
        """
        # extract the exact coordinates from the string of moves
        coord_letters = ["a", "b", "c", "d", "e", "f", "g", "h"]  # for translation of coordinates
        x_coord = [int(moves[i]) - 1 for i in range(1, len(moves), 2)]
        y_coord = [coord_letters.index(moves[i]) for i in range(0, len(moves), 2)]

        # set the start positions
        self.state[3, 3], self.state[4, 4] = - 1, -1
        self.state[3, 4], self.state[4, 3] = 1, 1

        # create an empty numpy array for appending flattened states and insert the initial state
        self.states_flattened = np.zeros(shape=(len(x_coord) + 1, (self.board_size * self.board_size) + 3), dtype=int)
        self.states_flattened[:, 0] = game_id  # save the ID of the game
        self.states_flattened[:, 1] = winner
        self.states_flattened[0, 3:] = self.state.flatten()  # flattened by rows

        # loop over the moves
        for i in range(len(x_coord)):
            if i % 2 == 0:  # black moves
                self.state[x_coord[i], y_coord[i]] = 1
                # If no stone was flipped from this player, then this move is not legal and thus it was not his turn
                if not self._flip_stones(label=1, x=x_coord[i], y=y_coord[i]):
                    self.state[x_coord[i], y_coord[i]] = -1
                    _ = self._flip_stones(label=-1, x=x_coord[i], y=y_coord[i])
            else:  # white moves
                self.state[x_coord[i], y_coord[i]] = -1
                if not self._flip_stones(label=-1, x=x_coord[i], y=y_coord[i]):
                    self.state[x_coord[i], y_coord[i]] = 1
                    _ = self._flip_stones(label=1, x=x_coord[i], y=y_coord[i])

            # save the state
            self.states_flattened[i + 1, 2] = i + 1
            self.states_flattened[i + 1, 3:] = self.state.flatten()

    def _flip_stones(self, label: int, x: int, y: int):
        """
        Flips all stones of the opposite color lying on a straight line between the new stone and any existing
        stones of the same colo. All flipped stones have the same color as the placed stone.

        Args:
            label: int, label (color) of the stone that has been placed (1 represents blacks, -1 whites)
            x: int, x coordinate on the board
            y: int, y coordinate on the board
        """
        # get the neighborhood
        neighborhood = self._get_neighborhood(x, y, specific=None)

        # is there a different color in the stone's neighborhood?
        neighborhood_check = [
            False if len(position) == 0 else
            True if self.state[position] == (-1 * label) else False for position in neighborhood
        ]

        legal = False   # Checks legality of a move, if the move is not legal, it was not this player's turn
        # Look for the same color in the direction where the opposite color was found
        for i in range(len(neighborhood)):
            if not neighborhood_check[i]:
                continue

            flip_candidates = []  # contains candidates for flipping
            current = neighborhood[i]
            while True:
                flip_candidates.append(current)
                neighbor = self._get_neighborhood(x=current[0], y=current[1], specific=i)

                # If we reached the end of the board
                if len(neighbor) == 0:
                    break

                # If we find the same stone in this direction, flip all the stones of the opposite color
                if self.state[neighbor] == label:
                    for flip_candidate in flip_candidates:
                        self.state[flip_candidate] = label
                    legal = True
                    break
                # If we hit a dead end by finding an empty field
                if self.state[neighbor] == 0:
                    break

                # If there is another stone of the opposite color
                current = neighbor

        return legal

    def _get_neighborhood(self, x: int, y: int, specific: int = None):
        """
        Returns neighborhood of a specific position on the game board.

        Args:
            x: int, x coordinate on the board
            y: int, y coordinate on the board
            specific: int, if specified, returns only 1 specific position in the neighborhood

        Returns:
            All game board positions in surrounding of the given coordinates.
        """
        neighborhood = [
            (x + 1, y),  # south
            (x - 1, y),  # north
            (x, y + 1),  # east
            (x, y - 1),  # west
            (x - 1, y + 1),  # north east
            (x + 1, y - 1),  # south west
            (x - 1, y - 1),  # north west
            (x + 1, y + 1),  # south east
        ]

        # check whether the position are really inside the game board
        for i, coord in enumerate(neighborhood):
            if coord[0] < 0 or coord[0] > self.board_size - 1 or coord[1] < 0 or coord[1] > self.board_size - 1:
                neighborhood[i] = ()

        if specific is None:
            return neighborhood

        return neighborhood[specific]

    def clear_data(self):
        """
        "Deletes" the existing data.
        """
        self.state[:, :] = 0
        self.states_flattened = None

    def get_results(self):
        """
        Returns numpy array of states.
        """
        return self.states_flattened


def main(input_path: str, output_path: str, random_sampling: bool = True, n_samples: int = 200, seed: int = 66):
    """
    Generates new game states for multiple games and saves the output as a pickle file.

    Args:
        input_path: string, path to the dataset
        output_path: string, name and path to the output file (do NOT specify the format, ex: ".csv")
        random_sampling: True or False, whether to randomly sample games from the input file
        n_samples: integer, number of games to be sampled
        seed: random seed for reproducibility
    """
    # get the game data
    with open(input_path, "r") as f:
        data = list(csv.reader(f, delimiter=','))

    # randomly pick some rows
    if random_sampling:
        rng = np.random.default_rng(seed=seed)
        row_ids = rng.choice(np.arange(1, len(data) + 1), size=n_samples, replace=False)
    else:
        row_ids = range(1, len(data) + 1)

    results = []
    OSG = OthelloStatesGenerator()
    # loop over samples and generate the states
    for i, row_id in tqdm.tqdm(enumerate(row_ids), total=len(row_ids), desc="Generating the game states"):
        OSG.generate_states(winner=int(data[row_id][1]), moves=data[row_id][2], game_id=i)
        results.append(OSG.get_results())
        OSG.clear_data()

    # Save the files
    results = np.concatenate(results)

    output_path += ".pkl"
    with open(output_path, "wb") as f:
        pickle.dump(results, f)

    print("Done")


if __name__ == '__main__':
    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument('config_file', help='path to config file', type=str)
    args = parser.parse_args()
    config_file = args.config_file

    with open(config_file, 'r') as fh:
        config = json.load(fh)
    main(**config)
