from rich.console import Console
from rich import print
from matplotlib import pyplot as plt
import numpy as np

from dataclasses import dataclass
from typing import List, Callable

@dataclass
class LocalReaction:
    # typing: N = number of molecules (reactants + products + intermediates), R = number of reactions (that need multiplied concentrations, the number of things to math)(eg. [E][S])
    labels: List[str]                           # N+R
    d_dt: np.ndarray                            # N, N+R
    maths: List[Callable[..., float]]  # takes Nx1 to R_i
    mass_balances: List[List[int]]              # list of indicies that will be summed for mass balance

    def check_mass_balances(self) -> bool:
        print([ self.d_dt[self.mass_balances[0], :] ])
        print('\n', [ not np.sum(self.d_dt[self.mass_balances[0], :], axis=0).any() ])
        return all(
            not np.sum(self.d_dt[balance, :], axis=0).any()
            for balance in self.mass_balances)

if __name__ == '__main__':
    reaction = LocalReaction(
        labels=['[E]', '[P]', '[S]', '[ES]', '[E][S]'],
        d_dt = np.array([[0, 0, 0, 8  , -1],
                         [0, 0, 0, 8/5, -1],
                         [0, 0, 0, 4*8/5, 0],
                         [0, 0, 0, -7 ,  1]]),
        maths=[lambda E, _, S, es: E * S],
        mass_balances=[[0, 3], [1, 2, 3]]
    )

    assert reaction.check_mass_balances(), "conservation of mass didn't check out"


