import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import json

DURATION = 600
STEP = 3
# STEP = 100

plt.style.use('dark_background')

with open(f'magnitudes_{DURATION}.json', 'r') as rf:
    y = np.array(json.load(rf))
x = np.linspace(0, DURATION, len(y))

x = x[::STEP]
y = y[::STEP]

print('max y:', max(y))

fig, ax = plt.subplots()
line, = ax.plot(x, y, color='#f8791c')
best_fit_line, = ax.plot([0], [0])


def update(num, x, y, line):
    line.set_data(x[:num], y[:num])
    # line.axes.axis([0, DURATION, 0, 0.0002])
    # line.axes.axis([0, DURATION, 0, 7])
    # line.axes.axis([100, 150, 0, 0.1])
    line.axes.axis([0, 60, 0, 7])
    if (num > 10):
        m, b = np.polyfit(x[:num], y[:num], 1)
        # best_fit_line.set_data([0, x[num]], [b, m*x[num]+b])
    return [line, best_fit_line]

ani = animation.FuncAnimation(fig, update, int(len(x)/10), fargs=[x, y, line],
                              interval=5, blit=True)
ani.save('test.mp4')
plt.show()

