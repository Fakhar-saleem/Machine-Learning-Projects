"""
Tic Tac Toe Player
"""

import math

X = "X"
O = "O"
EMPTY = None


def initial_state():
    """
    Returns starting state of the board.
    """
    return [[EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY]]


def player(board):
    x_count = sum(row.count(X) for row in board)
    o_count = sum(row.count(O) for row in board)
    return X if x_count == o_count else O

    raise NotImplementedError


def actions(board):
    return {(i, j) for i in range(3) for j in range(3) if board[i][j] == EMPTY}

    raise NotImplementedError


import copy

def result(board, action):
    i, j = action

    # Check for out-of-bounds indices
    if not (0 <= i <= 2 and 0 <= j <= 2):
        raise Exception("Action out of bounds")

    if board[i][j] is not EMPTY:
        raise Exception("Invalid move: cell already taken")

    new_board = [row.copy() for row in board]
    new_board[i][j] = player(board)
    return new_board


    raise NotImplementedError


def winner(board):
    lines = []

    # Rows and columns
    lines.extend(board)  # rows
    lines.extend([[board[i][j] for i in range(3)] for j in range(3)])  # columns

    # Diagonals
    lines.append([board[i][i] for i in range(3)])
    lines.append([board[i][2 - i] for i in range(3)])

    for line in lines:
        if line == [X, X, X]:
            return X
        elif line == [O, O, O]:
            return O
    return None

    raise NotImplementedError


def terminal(board):
    return winner(board) is not None or all(cell is not EMPTY for row in board for cell in row)

    raise NotImplementedError


def utility(board):
    win = winner(board)
    if win == X:
        return 1
    elif win == O:
        return -1
    else:
        return 0

    raise NotImplementedError


def minimax(board):
    if terminal(board):
        return None

    turn = player(board)

    def max_value(board):
        if terminal(board):
            return utility(board), None
        v = -math.inf
        best_action = None
        for action in actions(board):
            min_result, _ = min_value(result(board, action))
            if min_result > v:
                v = min_result
                best_action = action
                if v == 1:
                    break  # pruning
        return v, best_action

    def min_value(board):
        if terminal(board):
            return utility(board), None
        v = math.inf
        best_action = None
        for action in actions(board):
            max_result, _ = max_value(result(board, action))
            if max_result < v:
                v = max_result
                best_action = action
                if v == -1:
                    break  # pruning
        return v, best_action

    if turn == X:
        _, action = max_value(board)
    else:
        _, action = min_value(board)
    return action

    raise NotImplementedError
