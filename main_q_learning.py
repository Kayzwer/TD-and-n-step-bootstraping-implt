from TicTacToe import TicTacToe
from Q_Learning import Agent


if __name__ == "__main__":
    env = TicTacToe()
    agent = Agent(.25, .99, .2, 9)

    for i in range(100000000):
        state = env.reset()
        done = False
        score = 0.
        while not done:
            action = agent.choose_action(state, env.get_valid_actions())
            next_state, reward, done = env.step(action)
            score += reward
            if not done:
                next_action = agent.choose_action_greedy(
                    next_state, env.get_valid_actions())
                agent.update(state, action, reward, next_state, next_action)
                state = next_state
            else:
                agent.update_for_terminal_state(state, action, reward)
            if (i + 1) % 50000:
                agent.save("./ttt.pkl")
        print(f"Episode: {i + 1}, Score: {score}")
