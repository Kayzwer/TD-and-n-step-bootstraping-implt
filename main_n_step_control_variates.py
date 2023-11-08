from TicTacToe import TicTacToe
from N_Step_Control_Variates import Agent


if __name__ == "__main__":
    env = TicTacToe()
    agent = Agent(1., .99, .2, 9, 2)

    state = env.reset()
    score = 0.
    done = False
    agent.store_info(state=state)
    while not done:
        action, importance_sampling_ratio = agent.choose_action(
            state, env.get_valid_actions())
        next_state, reward, done = env.step(action)
        score += reward
        agent.store_info(
            state=next_state, action=action, reward=reward,
            state_expected_value=agent.get_state_mean_value(
                next_state, env.get_valid_actions()),
            importance_sampling_ratio=importance_sampling_ratio)
        agent.update(done)
    agent.update_remaining_info()
    for key, value in agent.state_actions_value.items():
        print(f"{key}: {[x for x in agent.state_actions_value[key]]}")
