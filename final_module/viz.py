import time
s = time.time()
from dataclasses import dataclass as dc
from rich import print
import numpy as np
from typing import List, Callable
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
print("imports took", round(time.time() - s), "seconds")

class Viz:
    def __init__(self, colors = ["#FF0000", "#004AFF"], x=10, y=10, show=True, save=False, save_path="./vids/out"):
        self.tilemap = np.zeros((x, y, 3), dtype=np.uint8)
        self.colors = colors
        self.xy = (x, y)

        self.new_seq = []
        self.show = show
        self.fig, self.ax = plt.subplots()
        self.image_plot = None
        self.frame_idx = 0
        self.ani = None
        self.save = save
        self.save_path = save_path

    def inp_to_tilemap(self, inp):
        for i in range(self.xy[0]):
            for j in range(self.xy[1]):
                self.tilemap[i, j] = self.mix_colors(torch.nn.functional.softmax(inp[i, j]), self.colors)

    def append_frame(self, inp):
        self.inp_to_tilemap(inp)
        self.new_seq.append(self.tilemap)

    ####################
    # COLOR CONVERSION #
    ####################
    def hex_to_rgb(self, hex_color: str) -> List[int]:
        hex_color = hex_color.lstrip('#')
        return [int(hex_color[i:i+2], 16) for i in range(0, len(hex_color), 2)]

    def rgb_to_hex(self, rgb_color: List[int]) -> str:
        return '#' + ''.join([f'{i:02x}' for i in rgb_color])

    def mix_colors(self, percentages: List[float], hex_colors: List[str]) -> str:
        if len(percentages) != len(hex_colors):
            raise ValueError("Input lists must have the same length")

        result_color = [0, 0, 0]
        for percentage, hex_color in zip(percentages, hex_colors):
            rgb_color = self.hex_to_rgb(hex_color)
            for i in range(3):
                result_color[i] += percentage * rgb_color[i]
        result_color = [int(round(c.item())) for c in result_color]
        return result_color

    ####################
    # RENDER FUNCTIONS #
    ####################
    def update(self, frame):
        self.image_plot.set_array(self.new_seq[frame])
        return [self.image_plot]

    def next_frame(self, event=None):
        # This function will be called to trigger the next frame
        if self.frame_idx < len(self.new_seq) - 1:
            self.frame_idx += 1
            self.update(self.frame_idx)
            self.fig.canvas.draw()
            plt.pause(0.001)

    def visualize_seq(self):
        self.image_plot = self.ax.imshow(self.new_seq[0])

        self.fig.canvas.mpl_connect('key_press_event', self.next_frame)
        if self.show:
            plt.axis('off')
            plt.ion()
            plt.show()

    def save_frames(self):
        if self.save:
            self.ani = animation.FuncAnimation(self.fig, self.update, frames=len(self.new_seq), interval=1)
            if self.save_path.endswith(".gif"):
                self.ani.save(f"{self.save_path}", writer='imagemagick', fps=60)
                print(f"Frames saved as GIF at {self.save_path}")
            elif self.save_path.endswith(".mp4"):
                self.ani.save(self.save_path, writer='imagemagick', fps=60)
                print(f"Frames saved as MP4 at {self.save_path}")



if __name__ == '__main__':
    TIME_STEPS = 100
    v = Viz(save=True, save_path="./vids/out.gif")

    # input is a 3d array of x,y,concentrations
    # example input:
    inp = torch.rand((10, 10, 2))
    v.append_frame(inp)
    v.visualize_seq()

    for i in range(TIME_STEPS):
        # inp = torch.rand((10, 10, 2))
        inp = v.diffusion_step(inp, 0.01)
        v.append_frame(inp)
        v.next_frame()
    v.save_frames()
