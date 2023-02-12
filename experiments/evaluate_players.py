
import chess
import random
from tensorflow import keras
import sys
import os
import pathlib
import numpy as np
import matplotlib.pyplot as plt

PROJ_DIR = pathlib.Path().absolute().parent
ENGINE_DIR = os.path.join(PROJ_DIR, "engine")
sys.path.append(ENGINE_DIR)
print(sys.path)

import setup
import parse
MODEL_PATH = os.path.join(ENGINE_DIR, setup.MODEL_NAME)

#########################
# Players Logic
#########################

def random_player(board):
    moves = list(board.legal_moves)

    if len(moves) == 0:
        return None
    else:
        return random.choice(moves)

nn_model = keras.models.load_model(MODEL_PATH)
def nn_player(board):
    best_move = None
    best_score = -9999999
    
    for move in board.legal_moves:
        board.push(move)

        input = np.array([parse.fen_to_vector(board.fen())])
        print(input.shape)
        score = nn_model.predict(input)
        print(score)

        if score > best_score:
            best_score = score
            best_move = move

        board.pop()

    return best_move



#########################
# UI + Game Loop
#########################

def material_balance(board):
    white = board.occupied_co[chess.WHITE]
    black = board.occupied_co[chess.BLACK]
    return (
        chess.popcount(white & board.pawns) - chess.popcount(black & board.pawns) +
        3 * (chess.popcount(white & board.knights) - chess.popcount(black & board.knights)) +
        3 * (chess.popcount(white & board.bishops) - chess.popcount(black & board.bishops)) +
        5 * (chess.popcount(white & board.rooks) - chess.popcount(black & board.rooks)) +
        9 * (chess.popcount(white & board.queens) - chess.popcount(black & board.queens))
    )

def game(white_player, black_player):
    board = chess.Board()

    while not board.is_seventyfive_moves():

        # White
        move = white_player(board)
        if move == None: break
        board.push(move)

        # Black
        move = black_player(board)
        if move == None: break
        board.push(move)

    return material_balance(board)
    

white_player = random_player
black_player = random_player
NUM_GAMES = 100
res = [game(white_player, black_player) for _ in range(NUM_GAMES)]
print(res)

plt.hist(res)
plt.suptitle(f"Balance over {NUM_GAMES} games. (POS:white wins NEG:black wins)")
plt.show()