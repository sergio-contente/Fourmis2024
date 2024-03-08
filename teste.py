import pygame as pg
import numpy as np
Nloc = 5
resolution = 25,25

pg.init()
a = np.array(Nloc, dtype=np.uint8)
print(a, type(a))

a_maze = maze.Maze(resolution, 12345)
screen = pg.display.set_mode(resolution)
mazeImg = a_maze.display