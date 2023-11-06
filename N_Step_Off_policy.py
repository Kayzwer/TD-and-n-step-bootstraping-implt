from typing import List, Optional, Tuple
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
        self.target_action_prob_memory = []
        self.behaviour_action_prob_memory = []

    def choose_action(self, state: str, valid_actions: List[int]) -> Tuple[int, float, float]:
        if not state in self.state_actions_value:
            self.state_actions_value[state] = np.zeros(
                self.n_actions, dtype=np.float32)
        max_ = float('-inf')
        for valid_action in valid_actions:
            if self.state_actions_value[state][valid_action] > max_:
                max_ = self.state_actions_value[state][valid_action]
        best_actions = []
        for valid_action in valid_actions:
            if self.state_actions_value[state][valid_action] == max_:
                best_actions.append(valid_action)
        if np.random.uniform(0., 1.) <= self.epsilon:
            action = np.random.choice(valid_actions)
        else:
            action = np.random.choice(best_actions)
        behaviour_action_prob = self.epsilon / len(valid_actions)
        target_action_prob = 0.
        best_actions_len_inv = 1. / len(best_actions)
        if action in best_actions:
            behaviour_action_prob += (1. - self.epsilon) * best_actions_len_inv
            target_action_prob += best_actions_len_inv
        return action, target_action_prob, behaviour_action_prob

    def store_info(self, state: Optional[str] = None,
                   action: Optional[int] = None,
                   reward: Optional[float] = None,
                   target_action_prob: Optional[float] = None,
                   behaviour_action_prob: Optional[float] = None) -> None:
        if state != None:
            if not state in self.state_actions_value:
                self.state_actions_value[state] = np.zeros(
                    self.n_actions, dtype=np.float32)
            self.state_memory.append(state)
        if action != None:
            self.action_memory.append(action)
        if reward != None:
            self.reward_memory.append(reward)
        if target_action_prob != None:
            self.target_action_prob_memory.append(target_action_prob)
        if behaviour_action_prob != None:
            self.behaviour_action_prob_memory.append(behaviour_action_prob)

    def update(self) -> None:
        if len(self.state_memory) == self.n_step + 1:
            return_sum = 0.0
            importance_sampling_ratio = 1.
            for i in range(self.n_step):
                importance_sampling_ratio *= \
                    self.target_action_prob_memory[i] / \
                    self.behaviour_action_prob_memory[i]
                return_sum += self.reward_memory[i] * self.gamma ** i
            return_sum += self.state_actions_value[self.state_memory[-1]
                                                   ][self.action_memory[-1]] * \
                self.gamma ** self.n_step
            state = self.state_memory.pop(0)
            action = self.action_memory.pop(0)
            self.reward_memory.pop(0)
            self.target_action_prob_memory.pop(0)
            self.behaviour_action_prob_memory.pop(0)
            self.state_actions_value[state][action] += self.alpha * \
                importance_sampling_ratio * \
                (return_sum - self.state_actions_value[state][action])

    def update_remaining_info(self) -> None:
        while len(self.state_memory) > 1:
            return_sum = 0.0
            importance_sampling_ratio = 1.
            for i in range(self.n_step):
                importance_sampling_ratio *= \
                    (self.target_action_prob_memory[i] /
                     self.behaviour_action_prob_memory[i]) \
                    if (i < len(self.target_action_prob_memory)) else 1.
                reward = self.reward_memory[i] if (
                    i < len(self.reward_memory)) else .0
                return_sum += reward * self.gamma ** i
            state = self.state_memory.pop(0)
            action = self.action_memory.pop(0)
            self.reward_memory.pop(0)
            self.target_action_prob_memory.pop(0)
            self.behaviour_action_prob_memory.pop(0)
            self.state_actions_value[state][action] += self.alpha * \
                importance_sampling_ratio * \
                (return_sum - self.state_actions_value[state][action])
        self.clear_memory()

    def clear_memory(self) -> None:
        self.state_memory.clear()
        self.action_memory.clear()
        self.reward_memory.clear()
        self.target_action_prob_memory.clear()
        self.behaviour_action_prob_memory.clear()

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(self.state_actions_value, f)

    def load(self, path: str) -> None:
        with open(path, "rb") as f:
            self.state_actions_value = pickle.load(f)
