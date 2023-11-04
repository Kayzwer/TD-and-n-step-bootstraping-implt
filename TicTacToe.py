from typing import Tuple, List
import numpy as np


class TicTacToe:
    def __init__(self) -> None:
        self.board = np.zeros((3, 3), dtype=np.int8)
        self.board_mapper = {
            0: " ",
            1: "X",
            2: "O"
        }

    def reset(self) -> str:
        self.__init__()
        return self.state_repr()

    def state_repr(self) -> str:
        output = ""
        for row in self.board:
            for entry in row:
                output += self.board_mapper[entry]
        return output

    def step(self, action: int) -> Tuple[str, float, bool]:
        row, col = divmod(action, 3)
        assert (0 <= action <= 8) and (self.board[row, col] == 0)
        self.board[row, col] = 1
        if self.check_win(1) == 1:
            return self.state_repr(), 1.0, True
        valid_actions = self.get_valid_actions()
        if len(valid_actions) == 0:
            return self.state_repr(), 0., True
        ran_row, ran_col = divmod(np.random.choice(valid_actions), 3)
        self.board[ran_row, ran_col] = 2
        if self.check_win(2) == 2:
            return self.state_repr(), -1.0, True
        return self.state_repr(), 0., len(self.get_valid_actions()) == 0

    def get_valid_actions(self) -> List[int]:
        i = 0
        actions = []
        for row in self.board:
            for entry in row:
                if entry == 0:
                    actions.append(i)
                i += 1
        return actions

    def check_win(self, player: int) -> int:
        match_ = np.array([player for _ in range(3)])
        diag, inv_diag = [], []
        for i in range(3):
            diag.append(self.board[i, i])
            inv_diag.append(self.board[i, 2 - i])
            if np.all(self.board[:, i] == match_) or np.all(self.board[i] ==
                                                            match_):
                return player
        if np.all(diag == match_) or np.all(inv_diag == match_):
            return player
        return 0

    def __str__(self) -> str:
        output = ""
        for row in self.board:
            for entry in row:
                output += f"{self.board_mapper[entry]}|"
            output += "\n"
        return output
