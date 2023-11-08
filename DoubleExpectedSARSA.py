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
        self.state_actions_value_1 = {}
        self.state_actions_value_2 = {}

    def choose_action(self, state: str, valid_actions: List[int]) -> int:
        if (not state in self.state_actions_value_1) or \
                (not state in self.state_actions_value_2):
            self.state_actions_value_1[state] = np.zeros(
                self.n_actions, dtype=np.float32)
            self.state_actions_value_2[state] = np.zeros(
                self.n_actions, dtype=np.float32)
        if np.random.uniform(0., 1.) <= self.epsilon:
            return np.random.choice(np.array(valid_actions, dtype=np.int8))
        else:
            max_ = float('-inf')
            for valid_action in valid_actions:
                state_value_sum = self.state_actions_value_1[state][
                    valid_action] + self.state_actions_value_2[state][
                    valid_action]
                if state_value_sum > max_:
                    max_ = state_value_sum
            best_actions = []
            for valid_action in valid_actions:
                if self.state_actions_value_1[state][valid_action] + \
                        self.state_actions_value_2[state][valid_action] == max_:
                    best_actions.append(valid_action)
            return np.random.choice(np.array(best_actions, dtype=np.int8))

    def update(self, state: str, action: int, reward: float, next_state: str,
               next_state_valid_actions: List[int]) -> None:
        if np.random.uniform(0., 1.) >= 0.5:
            max_ = float('-inf')
            for valid_action in next_state_valid_actions:
                if self.state_actions_value_2[next_state][valid_action] > max_:
                    max_ = self.state_actions_value_2[next_state][valid_action]
            actions = []
            for valid_action in next_state_valid_actions:
                if self.state_actions_value_2[next_state][valid_action] == max_:
                    actions.append(valid_action)
            selected_action = np.random.choice(actions)
            self.state_actions_value_1[state][action] += self.alpha * (
                reward + self.gamma * self.state_actions_value_2[next_state][
                    selected_action] - self.state_actions_value_1[state][action]
            )
        else:
            max_ = float('-inf')
            for valid_action in next_state_valid_actions:
                if self.state_actions_value_1[next_state][valid_action] > max_:
                    max_ = self.state_actions_value_1[next_state][valid_action]
            actions = []
            for valid_action in next_state_valid_actions:
                if self.state_actions_value_1[next_state][valid_action] == max_:
                    actions.append(valid_action)
            selected_action = np.random.choice(actions)
            self.state_actions_value_2[state][action] += self.alpha * (
                reward + self.gamma * self.state_actions_value_1[next_state][
                    selected_action] - self.state_actions_value_2[state][action]
            )

    def update_for_terminal_state(self, state: str, action: int, reward: float
                                  ) -> None:
        self.state_actions_value_1[state][action] += self.alpha * \
            (reward - self.state_actions_value_1[state][action])
        self.state_actions_value_2[state][action] += self.alpha * \
            (reward - self.state_actions_value_2[state][action])

    def get_state_mean_value(self, state: str, valid_actions: List[int]
                             ) -> float:
        max_ = float('-inf')
        state_actions_value = np.empty(len(valid_actions), dtype=np.float32)
        for i, valid_action in enumerate(valid_actions):
            state_actions_value[i] = self.state_actions_value_1[state][
                valid_action] + self.state_actions_value_2[state][
                valid_action]
            if state_actions_value[i] > max_:
                max_ = state_actions_value[i]
        best_actions = []
        for valid_action in valid_actions:
            if self.state_actions_value_1[state][valid_action] + \
                    self.state_actions_value_2[state][valid_action] == max_:
                best_actions.append(valid_action)
        mean = 0.
        for valid_action in valid_actions:
            weight = 0.
            if valid_action in best_actions:
                weight += (1. - self.epsilon) / len(best_actions)
            weight += self.epsilon / len(valid_actions)
            mean += weight * \
                (self.state_actions_value_1[state][valid_action] +
                 self.state_actions_value_2[state][valid_action])
        return mean * 0.5

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump((self.state_actions_value_1,
                        self.state_actions_value_2), f)

    def load(self, path: str) -> None:
        with open(path, "rb") as f:
            self.state_actions_value_1, self.state_actions_value_2 = \
                pickle.load(f)
