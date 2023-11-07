from typing import List, Optional
import numpy as np
import pickle


class Agent:
    def __init__(self, alpha: float, gamma: float, epsilon: float,
                 n_actions: int, n_step: int) -> None:
        assert 0. < alpha <= 1.
        assert 0. < gamma < 1.
        assert 0. <= epsilon <= 1.
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.n_actions = n_actions
        self.n_step = n_step
        self.state_actions_value = {}
        self.state_memory = []
        self.action_memory = []
        self.reward_memory = []

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

    def store_info(self, state: Optional[str] = None,
                   action: Optional[int] = None,
                   reward: Optional[float] = None) -> None:
        if state != None:
            if not state in self.state_actions_value:
                self.state_actions_value[state] = np.zeros(
                    self.n_actions, dtype=np.float32)
            self.state_memory.append(state)
        if action != None:
            self.action_memory.append(action)
        if reward != None:
            self.reward_memory.append(reward)

    def update(self) -> None:
        if len(self.state_memory) == self.n_step + 1:
            return_sum = 0.0
            for i in range(self.n_step):
                return_sum += self.reward_memory[i] * self.gamma ** i
            return_sum += self.state_actions_value[self.state_memory[-1]
                                                   ][self.action_memory[-1]] * \
                self.gamma ** self.n_step
            state = self.state_memory.pop(0)
            action = self.action_memory.pop(0)
            self.reward_memory.pop(0)
            self.state_actions_value[state][action] += self.alpha * \
                (return_sum - self.state_actions_value[state][action])

    def update_remaining_info(self) -> None:
        while len(self.state_memory) > 1:
            return_sum = 0.0
            for i in range(self.n_step):
                reward = self.reward_memory[i] if (
                    i < len(self.reward_memory)) else .0
                return_sum += reward * self.gamma ** i
            state = self.state_memory.pop(0)
            action = self.action_memory.pop(0)
            self.reward_memory.pop(0)
            self.state_actions_value[state][action] += self.alpha * \
                (return_sum - self.state_actions_value[state][action])
        self.clear_memory()

    def clear_memory(self) -> None:
        self.state_memory.clear()
        self.action_memory.clear()
        self.reward_memory.clear()

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(self.state_actions_value, f)

    def load(self, path: str) -> None:
        with open(path, "rb") as f:
            self.state_actions_value = pickle.load(f)
