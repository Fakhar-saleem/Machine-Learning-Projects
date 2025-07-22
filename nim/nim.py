import math
import random
import time


class Nim():

    def __init__(self, initial=[1, 3, 5, 7]):
        self.piles = initial.copy()
        self.player = 0
        self.winner = None

    @classmethod
    def available_actions(cls, piles):
        actions = set()
        for i, pile in enumerate(piles):
            for j in range(1, pile + 1):
                actions.add((i, j))
        return actions

    @classmethod
    def other_player(cls, player):
        return 0 if player == 1 else 1

    def switch_player(self):
        self.player = Nim.other_player(self.player)

    def move(self, action):
        pile, count = action
        if self.winner is not None:
            raise Exception("Game already won")
        if pile < 0 or pile >= len(self.piles):
            raise Exception("Invalid pile")
        if count < 1 or count > self.piles[pile]:
            raise Exception("Invalid number of objects")
        self.piles[pile] -= count
        self.switch_player()
        if all(p == 0 for p in self.piles):
            self.winner = self.player


class NimAI():

    def __init__(self, alpha=0.5, epsilon=0.1):
        self.q = {}
        self.alpha = alpha
        self.epsilon = epsilon

    def update(self, old_state, action, new_state, reward):
        old_q = self.get_q_value(old_state, action)
        future = self.best_future_reward(new_state)
        self.update_q_value(old_state, action, old_q, reward, future)

    def get_q_value(self, state, action):
        key = (tuple(state), action)
        return self.q.get(key, 0)

    def update_q_value(self, state, action, old_q, reward, future_rewards):
        key = (tuple(state), action)
        new_estimate = reward + future_rewards
        self.q[key] = old_q + self.alpha * (new_estimate - old_q)

    def best_future_reward(self, state):
        actions = Nim.available_actions(state)
        if not actions:
            return 0
        return max(self.get_q_value(state, action) for action in actions)

    def choose_action(self, state, epsilon=True):
        actions = list(Nim.available_actions(state))
        if not actions:
            return None
        # Explore
        if epsilon and random.random() < self.epsilon:
            return random.choice(actions)
        # Exploit
        best = None
        best_q = -math.inf
        for action in actions:
            q = self.get_q_value(state, action)
            if q > best_q:
                best_q = q
                best = action
        return best


def train(n):
    player = NimAI()
    for i in range(n):
        game = Nim()
        last = {0: {"state": None, "action": None},
                1: {"state": None, "action": None}}
        while True:
            state = game.piles.copy()
            action = player.choose_action(game.piles)
            last[game.player]["state"] = state
            last[game.player]["action"] = action
            game.move(action)
            new = game.piles.copy()
            if game.winner is not None:
                player.update(state, action, new, -1)
                prev = Nim.other_player(game.player)
                ps = last[prev]
                player.update(ps["state"], ps["action"], new, 1)
                break
            elif last[game.player]["state"] is not None:
                ps = Nim.other_player(game.player)
                ls = last[ps]
                player.update(ls["state"], ls["action"], new, 0)
    return player


def play(ai, human_player=None):
    if human_player is None:
        human_player = random.randint(0, 1)
    game = Nim()
    while True:
        print("Piles:", game.piles)
        actions = Nim.available_actions(game.piles)
        if game.player == human_player:
            while True:
                i = int(input("Choose pile: "))
                j = int(input("Choose count: "))
                if (i, j) in actions:
                    action = (i, j)
                    break
                print("Invalid move")
        else:
            action = ai.choose_action(game.piles, epsilon=False)
            print(f"AI chose {action}")
        game.move(action)
        if game.winner is not None:
            winner = "Human" if game.winner == human_player else "AI"
            print(f"Winner: {winner}")
            return


if __name__ == '__main__':
    ai = train(10000)
    play(ai)
