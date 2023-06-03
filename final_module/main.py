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
    maths_mask: torch.Tensor                    # R x N x1 mask, cols is masks
    mass_balances: List[List[int]]              # list of indicies that will be summed for mass balance

    def __post_init__(self):
        self.n_chems, tot_mathed = self.d_dc.shape
        self.n_maths = tot_mathed - self.n_chems
        assert len(self.labels) == tot_mathed, f"Expected {tot_mathed} labels from shape of d_dc but got {len(self.labels)}!"
        assert self.maths_mask.shape == torch.Size([self.n_maths, self.n_chems, 1]), f"Expected maths mask shape {self.n_chems}, {self.n_maths}, 1, got {self.maths_mask.shape}"
        self.check_mass_balances()

    def check_mass_balances(self, verbose=False) -> bool:
        if verbose: print("mass balances:\n", self.d_dc[self.mass_balances[0], :], sep='')
        # print('\n', [ not torch.sum(self.d_dc[self.mass_balances[0], :], dim=0).any() ])
        return all(
            not torch.sum(self.d_dc[balance, :], dim=0).any()
            for balance in self.mass_balances)

    def step(self, concentrations: torch.Tensor, verbose=True) -> torch.Tensor:
        """Expects concentrations to be a tensor of N+R x X, where X is the flattened grid dimensions"""
        # flatten the concentrations so that we can assume it's 1d
        n_chems, grid_dims_flat = concentrations.shape
        assert n_chems == self.n_chems

        # calced = torch.Tensor([f(*concentrations.flatten().tolist()) for f in self.maths])
        # calculate the calculated values by expanding concentration/maths_masks by the other and then taking the product along dim=1
        mask = self.maths_mask.expand([-1, -1, grid_dims_flat])
        masked = masked_tensor(concentrations.expand_as(mask), mask)
        calced = masked.prod(dim=1).to_tensor(0)

        # add calculated values to the previous concentrations
        with_math = torch.vstack([concentrations, calced])
        new_conc = concentrations + self.d_dc.mm(with_math)
        return new_conc

if __name__ == '__main__':
    reaction = LocalReaction(
        labels=['[E]', '[P]', '[S]', '[ES]', '[E][S]', 'everything'],
        d_dc = torch.tensor([[0, 0, 0, 8  , -1  , 0],
                             [0, 0, 0, 8/5, -1  , 0],
                             [0, 0, 0, 4*8/5, 0 , 0],
                             [0, 0, 0, -8 ,  1  , 0]]),
        maths_mask=torch.tensor([[True, False, True, False], [True, True, True, True]], dtype=torch.bool).unsqueeze(-1),
        # maths_mask=torch.tensor([[True, False, True, False]], dtype=torch.bool),
        mass_balances=[[0, 3], [1, 2, 3]]
    )

    assert reaction.check_mass_balances(), "conservation of mass didn't check out"


    gridsize = (2, 4)
    initial_concentrations = [1, 0, 10, 0]
    concentrations = torch.Tensor(initial_concentrations)\
            .repeat(*gridsize, 1).transpose(-1, 0)  # shape = (n_chems, y_rows, x_cols)
    print(concentrations)

    concentrations = concentrations.flatten(start_dim=1)

    for _ in trange(int(1e3)):
        concentrations = reaction.step(concentrations)

