from TicTacToe import TicTacToe
from DoubleExpectedSARSA import Agent


if __name__ == "__main__":
    env = TicTacToe()
    agent = Agent(1., .99, .0, 9)

    for i in range(100000000):
        state = env.reset()
        done = False
        score = 0.
        action = agent.choose_action(state, env.get_valid_actions())
        while not done:
            next_state, reward, done = env.step(action)
            score += reward
            if not done:
                next_action = agent.choose_action(
                    next_state, env.get_valid_actions())
                agent.update(state, action, reward, next_state,
                             env.get_valid_actions())
                state, action = next_state, next_action
            else:
                agent.update_for_terminal_state(state, action, reward)
        if (i + 1) % 50000 == 0:
            agent.save("./ttt.pkl")
        print(f"Episode: {i + 1}, Score: {score}")
