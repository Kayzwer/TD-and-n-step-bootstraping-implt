from TicTacToe import TicTacToe
from N_Step_Off_policy import Agent


if __name__ == "__main__":
    env = TicTacToe()
    agent = Agent(1., .99, .2, 9, 1)

    for i in range(10000000):
        state = env.reset()
        score = 0.
        done = False
        agent.store_info(state=state)
        while not done:
            action, target_action_prob, behaviour_action_prob = agent.choose_action(
                state, env.get_valid_actions())
            next_state, reward, done = env.step(action)
            score += reward
            agent.store_info(state=next_state, action=action, reward=reward,
                             target_action_prob=target_action_prob,
                             behaviour_action_prob=behaviour_action_prob)
            agent.update()
        agent.update_remaining_info()
        if (i + 1) % 50000 == 0:
            agent.save("./ttt.pkl")
        print(f"Episode: {i + 1}, Score: {score}")
