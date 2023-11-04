from TicTacToe import TicTacToe
from ExpectedSARSA import Agent


if __name__ == "__main__":
    env = TicTacToe()
    agent = Agent(.25, .99, 0., 9)
    agent.load("./ttt.pkl")

    state = env.reset()
    done = False
    while True:
        action = agent.choose_action(state, env.get_valid_actions())
        action_row, action_col = divmod(action, 3)
        env.board[action_row, action_col] = 1
        if env.check_win(1) == 1:
            break
        print(env)
        player_action = int(input("Pos: "))
        if not player_action in env.get_valid_actions():
            raise Exception("Invalid Action")
        player_action_row, player_action_col = divmod(player_action, 3)
        env.board[player_action_row, player_action_col] = 2
        if env.check_win(2) == 2:
            break
    print(env)
