import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from numba import njit
from tqdm import tqdm, trange

STEPS = int(1e5)
DELTA_T = 5e-4
BOUNDS = (-5, 5)

time = np.linspace(0, STEPS * DELTA_T, STEPS)

# init_points = [ np.array([0.5, 0.5, 0.5]), np.array([-0.5, -0.5, -0.5]) ]
init_points = [ np.array([12, 12, 12]), np.array([-12, -12, -12]) ]
init_sep_vector = init_points[1] - init_points[0]

# @njit
def lorentz(init_con, steps, delta_t):
    sigma = 10
    rho = 28
    beta = 8/3
    d_dt = lambda x, y, z : np.array([
            sigma * (y - x),
            x * (rho - z) - y,
            x*y - beta * z
        ])

    # ret = [init_con]
    # for _ in range(steps):
    #     ret.append(ret[-1] + d_dt(ret[-1][0], ret[-1][1], ret[-1][2]) * delta_t)
    # return np.array(ret)

    ret = np.zeros((steps, 3))
    ret[0, :] = init_con
    for i in range(1, ret.shape[0]):
        ret[i, :] = ret[i-1, :] + d_dt(ret[i-1, 0], ret[i-1, 1], ret[i-1, 2]) * delta_t
    print(ret)

    return ret

# with tqdm(total=STEPS*len(init_points), desc="simulating") as pbar:
#     trails = [lorentz(p, STEPS, DELTA_T, pbar) for p in init_points]

trails = [lorentz(p, STEPS, DELTA_T) for p in init_points]

print("trails shape:", trails[0].shape)

fig, (chart_ax, space_ax) = plt.subplots(1, 2, figsize=(8, 4))
lines = [space_ax.plot([], [])[0] for _ in trails]

# space_ax.plot(trails[0][:, 0], trails[0][:, 1], label='the plot')
# plt.show()

def update(num, trails, lines, pbar):
    for trail, line in zip(trails, lines):
        line.set_data(trail[:num, 0], trail[:num, 1])
        line.axes.axis([*BOUNDS, *BOUNDS])
    return lines


with tqdm(total=STEPS) as pbar:
    ani = animation.FuncAnimation(fig, update, STEPS, fargs=[trails, lines, pbar],
                              interval=25, blit=True)
ani.save('test.gif')
plt.show()

