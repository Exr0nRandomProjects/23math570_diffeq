from rich.console import Console
from rich import print
from matplotlib import pyplot as plt
# import numpy as np
import torch
from tqdm import tqdm, trange

from dataclasses import dataclass
from typing import List, Callable, Self

@dataclass
class LocalReaction:
    # typing: N = number of molecules (reactants + products + intermediates), R = number of reactions (that need multiplied concentrations, the number of things to math)(eg. [E][S])
    labels: List[str]                           # N+R
    d_dc: torch.Tensor                          # N, N+R
    maths: List[Callable[..., float]]  # takes Nx1 to R_i
    mass_balances: List[List[int]]              # list of indicies that will be summed for mass balance

    def check_mass_balances(self, verbose=False) -> bool:
        if verbose: print("mass balances:\n", self.d_dc[self.mass_balances[0], :], sep='')
        # print('\n', [ not torch.sum(self.d_dc[self.mass_balances[0], :], dim=0).any() ])
        return all(
            not torch.sum(self.d_dc[balance, :], dim=0).any()
            for balance in self.mass_balances)

    def step(self, concentrations: torch.Tensor) -> torch.Tensor:
        """Expects concentrations to be a tensor of N+R x (something), where something can be multiple dimensions that will be flattened then later unflattened"""
        # calced = torch.Tensor([f(*concentrations.flatten().tolist()) for f in self.maths])
        calced = torch.appyl
        with_math = torch.vstack([concentrations, calced])
        # print(with_math)
        print(self.d_dc.mul(with_math))
        new_conc = concentrations + self.d_dc.mul(with_math)
        return new_conc
>>>>>>> 58907fcda95bfbba0cc70a369eb724d7cff63516

if __name__ == '__main__':
    reaction = LocalReaction(
        labels=['[E]', '[P]', '[S]', '[ES]', '[E][S]'],
        d_dc = torch.Tensor([[0, 0, 0, 8  , -1],
                             [0, 0, 0, 8/5, -1],
                             [0, 0, 0, 4*8/5, 0],
                             [0, 0, 0, -8 ,  1]]),
        maths=[lambda E, _, S, es: E * S],
        mass_balances=[[0, 3], [1, 2, 3]]
    )

    assert reaction.check_mass_balances(), "conservation of mass didn't check out"

<<<<<<< HEAD
=======

    gridsize = (2, )
    concentrations = torch.Tensor([1, 0, 10, 0])\
            .repeat(*gridsize, 1).transpose(-1, 0)  # shape = (n_chems, y_rows, x_cols)
    print(concentrations)

    for _ in trange(1):
        print(concentrations.flatten())
        concentrations = reaction.step(concentrations)

>>>>>>> 58907fcda95bfbba0cc70a369eb724d7cff63516
