from rich.console import Console
from rich import print
from matplotlib import pyplot as plt
# import numpy as np
import torch
from torch.masked import masked_tensor
from tqdm import tqdm, trange

from dataclasses import dataclass
from typing import List, Callable

from viz import Viz as Vis

@dataclass
class LocalReaction:
    # typing: N = number of molecules (reactants + products + intermediates), R = number of reactions (that need multiplied concentrations, the number of things to math)(eg. [E][S])
    labels: List[str]                           # N+R
    d_dc: torch.Tensor                          # N, N+R
    maths_masks: List[List[bool]]               # the len(R) x len(N) input of masks
    mass_balances: List[List[int]]              # list of indicies that will be summed for mass balance

    def __post_init__(self):
        self.n_chems, tot_mathed = self.d_dc.shape
        self.n_maths = tot_mathed - self.n_chems
        self.maths_mask = torch.tensor(self.maths_masks, dtype=torch.bool).unsqueeze(-1)
        assert len(self.labels) == tot_mathed, f"Expected {tot_mathed} labels from shape of d_dc but got {len(self.labels)}!"
        assert self.maths_mask.shape == torch.Size([self.n_maths, self.n_chems, 1]), f"Expected maths mask shape {self.n_chems}, {self.n_maths}, 1, got {self.maths_mask.shape}"
        assert self.check_mass_balances(verbose=True), "conservation of mass didn't check out"

    def check_mass_balances(self, verbose=False) -> bool:
        if verbose: print("mass balances:\n", self.d_dc[self.mass_balances[0], :], sep='')
        # print('\n', [ not torch.sum(self.d_dc[self.mass_balances[0], :], dim=0).any() ])
        return all(
            (torch.sum(self.d_dc[balance, :], dim=0).abs() < 1e-5).all()
            for balance in self.mass_balances)

    def step(self, concentrations: torch.Tensor, verbose=True) -> torch.Tensor:
        """Expects concentrations to be a tensor of N+R x X, where X is the flattened grid dimensions"""
        # flatten the concentrations so that we can assume it's 1d
        n_chems, grid_dims_flat = concentrations.shape
        assert n_chems == self.n_chems

        # calculate the calculated values by expanding concentration/maths_masks by the other and then taking the product along dim=1
        mask = self.maths_mask.expand([-1, -1, grid_dims_flat])
        masked = masked_tensor(concentrations.expand_as(mask), mask)
        calced = masked.prod(dim=1).to_tensor(0)

        # add calculated values to the previous concentrations
        with_math = torch.vstack([concentrations, calced])
        new_conc = concentrations + self.d_dc.mm(with_math)
        return new_conc


def diffusion_step(t, D):
    next_grid = torch.clone(t)

    # dirichelet
    # next_grid[0,:,:] = 0
    # next_grid[-1,:,:] = 0
    # next_grid[:,0,:] = 0
    # next_grid[:,-1,:] = 0

    # neumann
    next_grid[0,:,:] = next_grid[1,:,:]
    next_grid[-1,:,:] = next_grid[-2,:,:]
    next_grid[:,0,:] = next_grid[:,1,:]
    next_grid[:,-1,:] = next_grid[:,-2,:]

    for i in range(1, t.shape[0] - 1):
        for j in range(1, t.shape[1] - 1):
            for k in range(t.shape[2]):
                next_grid[i, j, k] = t[i, j, k] + D * \
                    (t[i+1, j, k] + t[i-1, j, k] + \
                    t[i, j+1, k] + t[i, j-1, k] - \
                    4*t[i, j, k])
    # self.tilemap = next_grid
    return next_grid


if __name__ == '__main__':
    reaction = LocalReaction(
        labels=['[E]', '[P]', '[S]', '[ES]', '[E][S]', 'everything'],
        d_dc = torch.tensor([[0, 0, 0, 8  , -1  , 0],
                             [0, 0, 0, 8/5, -1  , 0],
                             [0, 0, 0, 4*8/5, 0 , 0],
                             [0, 0, 0, -8 ,  1  , 0]])*0.01,
        maths_masks=[[True, False, True, False], [True, True, True, True]],
        mass_balances=[[0, 3], [1, 2, 3]]
    )

    gridsize = (6, 1)
    initial_concentrations = [1, 0, 10, 0]
    TIME_STEPS = int(1e3)
    VIS_STEPS = 5
    DIFFUSION_RATIO = 0.01
    VIS_INDICIES = { 1: '#00ff00', 2: '#ff0000', 0: '#0000aa' }


    world = torch.Tensor(initial_concentrations)\
            .repeat(*gridsize, 1).transpose(-1, 0)  # shape = (n_chems, y_rows, x_cols)
    print(world)

    v = Vis(colors=list(VIS_INDICIES.values()), x=gridsize[0], y=gridsize[1], save=True)
    v.append_frame(world[list(VIS_INDICIES.keys())].transpose(0, 2))
    v.visualize_seq()

    flatland = world.flatten(start_dim=1)

    # for i in trange(int(3e1), disable=True):
    for i in trange(int(1e5)):
        flatland = reaction.step(flatland)
        # world = flatland.reshape([-1, *gridsize])
        # world = diffusion_step(world, DIFFUSION_RATIO)
        # flatland = world.flatten(start_dim=1)

        # print(flatland)
        # print('\n\n')

        if i % VIS_STEPS == 0:
            v.append_frame(flatland[list(VIS_INDICIES.keys())].transpose(0, 1).reshape(*gridsize, -1))
            v.next_frame()
