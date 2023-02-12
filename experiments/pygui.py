
import chess
from chessboard import display
from time import sleep
import random
import pygame
from tensorflow import keras
import sys
import os
import pathlib
import numpy as np

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
    best_score = 9999999
    
    for move in board.legal_moves:
        board.push(move)

        input = np.array([parse.fen_to_vector(board.fen())])
        score = nn_model.predict(input, verbose=0)
        print(score)

        if score < best_score:
            best_score = score
            best_move = move

        board.pop()

    return best_move


white_player = random_player
black_player = nn_player

#########################
# UI + Game Loop
#########################

board = chess.Board()
game_board_ui = display.start()
pygame.display.set_caption(f"WHITE: {white_player.__name__}   |   BLACK: {black_player.__name__}")

turn = chess.WHITE
while True:

    if turn == chess.WHITE:
        move = white_player(board)
        if move == None: break
        board.push(move)
        turn = chess.BLACK

    else: # turn == chess.BLACK
        move = black_player(board)
        if move == None: break
        board.push(move)
        turn = chess.WHITE


    # Checking GUI window for QUIT event. (Esc or GUI CANCEL)
    display.update(board.fen(), game_board_ui)
    display.check_for_quit()

    #sleep(0.1)

sleep(5)
# Close window
display.terminate()

