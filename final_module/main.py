from rich.console import Console
from rich import print
from matplotlib import pyplot as plt
# import numpy as np
import torch
from torch.masked import masked_tensor
from tqdm import tqdm, trange

from dataclasses import dataclass
from typing import List, Callable, Self

@dataclass
class LocalReaction:
    # typing: N = number of molecules (reactants + products + intermediates), R = number of reactions (that need multiplied concentrations, the number of things to math)(eg. [E][S])
    labels: List[str]                           # N+R
    d_dc: torch.Tensor                          # N, N+R
    # maths: List[Callable[..., float]]  # takes Nx1 to R_i
    maths_mask: torch.Tensor                    # N x R mask, cols is masks
    mass_balances: List[List[int]]              # list of indicies that will be summed for mass balance


    def check_mass_balances(self, verbose=False) -> bool:
        if verbose: print("mass balances:\n", self.d_dc[self.mass_balances[0], :], sep='')
        # print('\n', [ not torch.sum(self.d_dc[self.mass_balances[0], :], dim=0).any() ])
        return all(
            not torch.sum(self.d_dc[balance, :], dim=0).any()
            for balance in self.mass_balances)

    def step(self, concentrations: torch.Tensor, verbose=True) -> torch.Tensor:
        """Expects concentrations to be a tensor of N+R x (something), where something can be multiple dimensions that will be flattened then later unflattened"""
        # flatten the concentrations so that we can assume it's 1d
        n_chems, *grid_dims = concentrations.shape
        concentrations = concentrations.reshape(n_chems, -1)

        # calced = torch.Tensor([f(*concentrations.flatten().tolist()) for f in self.maths])
        # calculate the calculated values
        print(self.maths_mask)
        masked = concentrations.unsqueeze(0).repeat(self.maths_mask.shape[0], 1, 1)
        print(masked)
        print("------------")
        if verbose: print('masking shape', masked.shape, 'with', self.maths_mask.shape)
        # masked = masked[self.maths_mask, :]
        masked = masked_tensor(masked, self.maths_mask)
        print(masked)
        exit()
        masked_tensor(concentrations, self.maths_mask)

        calced = concentrations.repeat(self.maths_mask.shape[0], 1)


        # add calculated values to the previous concentrations
        with_math = torch.vstack([concentrations, calced])
        # print(with_math)
        print(self.d_dc.mul(with_math))
        new_conc = concentrations + self.d_dc.mul(with_math)
        return new_conc

if __name__ == '__main__':
    reaction = LocalReaction(
        labels=['[E]', '[P]', '[S]', '[ES]', '[E][S]'],
        d_dc = torch.tensor([[0, 0, 0, 8  , -1],
                             [0, 0, 0, 8/5, -1],
                             [0, 0, 0, 4*8/5, 0],
                             [0, 0, 0, -8 ,  1]]),
        maths_mask=torch.tensor([[True, False, True, False], [True, True, True, True]], dtype=torch.bool),
        # maths_mask=torch.tensor([[True, False, True, False]], dtype=torch.bool),
        mass_balances=[[0, 3], [1, 2, 3]]
    )

    assert reaction.check_mass_balances(), "conservation of mass didn't check out"


    gridsize = (2, 3)
    concentrations = torch.Tensor([1, 0, 10, 0])\
            .repeat(*gridsize, 1).transpose(-1, 0)  # shape = (n_chems, y_rows, x_cols)
    print(concentrations)

    for _ in trange(1):
        # print(concentrations.flatten())
        concentrations = reaction.step(concentrations)

