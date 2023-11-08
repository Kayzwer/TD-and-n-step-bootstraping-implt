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
        self.state_expected_value_memory = []
        self.importance_sampling_ratio_memory = []

    def choose_action(self, state: str, valid_actions: List[int]
                      ) -> Tuple[int, float]:
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
        action = np.random.choice(np.array(valid_actions, dtype=np.int8) if
                                  np.random.uniform(0., 1.) <= self.epsilon
                                  else np.array(best_actions, dtype=np.int8))
        behaviour_action_prob = self.epsilon / len(valid_actions)
        target_action_prob = 0.
        best_actions_len_inv = 1. / len(best_actions)
        if action in best_actions:
            behaviour_action_prob += (1. - self.epsilon) * best_actions_len_inv
            target_action_prob += best_actions_len_inv
        return action, target_action_prob / behaviour_action_prob

    def store_info(self, state: Optional[str] = None,
                   action: Optional[int] = None, reward: Optional[float] = None,
                   state_expected_value: Optional[float] = None,
                   importance_sampling_ratio: Optional[float] = None) -> None:
        if state != None:
            if not state in self.state_actions_value:
                self.state_actions_value[state] = np.zeros(
                    self.n_actions, dtype=np.float32)
            self.state_memory.append(state)
        if action != None:
            self.action_memory.append(action)
        if reward != None:
            self.reward_memory.append(reward)
        if state_expected_value != None:
            self.state_expected_value_memory.append(state_expected_value)
        if importance_sampling_ratio != None:
            self.importance_sampling_ratio_memory.append(
                importance_sampling_ratio)

    def get_state_mean_value(self, state: str, valid_actions: List[int]
                             ) -> float:
        if len(valid_actions) == 0:
            return 0.
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
        mean = 0.
        for valid_action in valid_actions:
            weight = 0.
            if valid_action in best_actions:
                weight += (1. - self.epsilon) / len(best_actions)
            weight += self.epsilon / len(valid_actions)
            mean += weight * self.state_actions_value[state][valid_action]
        return mean

    def update(self, done: bool) -> None:
        if len(self.state_memory) == self.n_step + 1:
            return_sum = 0.0
            for i in range(self.n_step - 1, -1, -1):
                if i == self.n_step - 1:
                    if done:
                        return_sum += self.reward_memory[i]
                    else:
                        return_sum += self.state_actions_value[
                            self.state_memory[i]][self.action_memory[i]]
                else:
                    return_sum = self.reward_memory[i] + self.gamma * (
                        self.importance_sampling_ratio_memory[i] * (
                            return_sum - self.state_actions_value[
                                self.state_memory[i]][self.action_memory[i]]) +
                        self.state_expected_value_memory[i])
            state = self.state_memory.pop(0)
            action = self.action_memory.pop(0)
            self.reward_memory.pop(0)
            self.state_expected_value_memory.pop(0)
            self.importance_sampling_ratio_memory.pop(0)
            self.state_actions_value[state][action] += self.alpha * \
                (return_sum - self.state_actions_value[state][action])

    def update_remaining_info(self) -> None:
        while (n := len(self.state_memory)) > 1:
            return_sum = 0.0
            for i in range(n - 2, -1, -1):
                if i == n - 2:
                    return_sum += self.reward_memory[i]
                else:
                    return_sum = self.reward_memory[i] + self.gamma * (
                        self.importance_sampling_ratio_memory[i] * (
                            return_sum - self.state_actions_value[
                                self.state_memory[i]][self.action_memory[i]]) +
                        self.state_expected_value_memory[i])
            state = self.state_memory.pop(0)
            action = self.action_memory.pop(0)
            self.reward_memory.pop(0)
            self.state_expected_value_memory.pop(0)
            self.importance_sampling_ratio_memory.pop(0)
            self.state_actions_value[state][action] += self.alpha * \
                (return_sum - self.state_actions_value[state][action])
        self.clear_memory()

    def clear_memory(self) -> None:
        self.state_memory.clear()
        self.action_memory.clear()
        self.reward_memory.clear()
        self.state_expected_value_memory.clear()
        self.importance_sampling_ratio_memory.clear()

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(self.state_actions_value, f)

    def load(self, path: str) -> None:
        with open(path, "rb") as f:
            self.state_actions_value = pickle.load(f)
