from typing import List
import numpy as np
import pickle


class Agent:
    def __init__(self, alpha: float, gamma: float, epsilon: float,
                 n_actions: int) -> None:
        assert 0. < alpha <= 1.
        assert 0. < gamma < 1.
        assert 0. <= epsilon <= 1.
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.n_actions = n_actions
        self.state_actions_value = {}

    def choose_action(self, state: str, valid_actions: List[int]) -> int:
        if not state in self.state_actions_value:
            self.state_actions_value[state] = np.zeros(
                self.n_actions, dtype=np.float32)
        if np.random.uniform(0., 1.) <= self.epsilon:
            return np.random.choice(np.array(valid_actions, dtype=np.int8))
        else:
            max_ = float('-inf')
            for valid_action in valid_actions:
                if self.state_actions_value[state][valid_action] > max_:
                    max_ = self.state_actions_value[state][valid_action]
            best_actions = []
            for valid_action in valid_actions:
                if self.state_actions_value[state][valid_action] == max_:
                    best_actions.append(valid_action)
            return np.random.choice(np.array(best_actions, dtype=np.int8))

    def update(self, state: str, action: int, reward: float, next_state: str,
               next_state_action: int) -> None:
        self.state_actions_value[state][action] += self.alpha * (
            reward + self.gamma * self.state_actions_value[next_state][
                next_state_action] - self.state_actions_value[state][action])

    def update_for_terminal_state(self, state: str, action: int, reward: float
                                  ) -> None:
        self.state_actions_value[state][action] += self.alpha * \
            (reward - self.state_actions_value[state][action])

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(self.state_actions_value, f)

    def load(self, path: str) -> None:
        with open(path, "rb") as f:
            self.state_actions_value = pickle.load(f)
