# -------------------------------------------------------
# Assignment 1
# Written by Andre Parsons-Legault 40031363
# For COMP 472 Section JX – Summer 2020
# --------------------------------------------------------

# Using A* to find an optimal path between two coordinates on a map.
# https://basemaptutorial.readthedocs.io/en/latest/shapefile.html
# https://github.com/GeospatialPython/pyshp

import numpy as np
from math import floor
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import geopandas

#from mpl_toolkits.axes_grid1 import ImageGrid
import shapefile

# Defines high crime rate vs. low crime rate.
THRESHOLD = 0.5

# Size of a block in the grid
BLOCK_SIZE_X = 0.002 # should be <= 0.002 (recommended)
BLOCK_SIZE_Y = 0.002 # should be <= 0.002 (recommended)

NUM_LABELS_X = 5
NUM_LABELS_Y = 5

# four coordinates
P1 = {'x': -73.59, 'y': 45.49}
P2 = {'x': -73.59, 'y': 45.53}
P3 = {'x': -73.55, 'y': 45.53}
P4 = {'x': -73.55, 'y': 45.49}

""" Using first points as anchors, get position in square.
An input such as (-73.59, 45.49) returns 0, 0 as this point is located
in the bottom left cell (using grid_size=0.002).
"""
def get_pos(x, y):
    pos_x = floor((x - P1['x']) / BLOCK_SIZE_X)
    pos_y = floor((y - P1['y']) / BLOCK_SIZE_Y)
    return pos_x, pos_y

def get_num_blocks(num1, num2, size):
    return round((num1 - num2) / size)

def get_labels(start, end, num, size):
    labels = [0, start]
    current = start
    jump = (end - start) / num
    for i in range(num):
        current = current + jump
        labels.append(round(current, 3))
    
    return labels, ticker.MultipleLocator(jump / size)
    
def f_func():
    return g_func() + h_func()

def g_func(point):
    cell = get_cell(point)
    if cell:
        pass

def get_grid(points, num_x, num_y):
    grid = np.zeros(shape=(num_blocks_x, num_blocks_y), dtype=int)

    for point in points:
        x_pos, y_pos = get_pos(point.x, point.y)
        grid[x_pos][y_pos] = grid[x_pos][y_pos] + 1

    return grid

def show_cell_counts(pc, ax, grid, num_x, num_y):
    counter = 0
    for p in pc.get_paths():
        print(p.vertices[:,:]) # bottom-right, top-right, top-left, bottom-left
        break
        x, y = p.vertices[-2:,:].mean(0)
        ax.text(x, y, grid[floor(counter / num_x)][counter % num_y], c='white')
        counter = counter + 1

""" This class is a dict that represents which points are bound to which grid cells. """
class CrimeGraph:
    def __init__(self, coords):
        pass

# https://matplotlib.org/3.1.1/gallery/images_contours_and_fields/pcolor_demo.html#sphx-glr-gallery-images-contours-and-fields-pcolor-demo-py
if __name__ == '__main__':
    num_blocks_x = get_num_blocks(P4['x'], P1['x'], BLOCK_SIZE_X)
    num_blocks_y = get_num_blocks(P2['y'], P1['y'], BLOCK_SIZE_Y)

    gdf = geopandas.read_file('crime_data')
    grid = get_grid(gdf['geometry'], num_blocks_x, num_blocks_y)
    grid_colors = np.copy(grid)

    avg = np.average(grid)
    std = np.std(grid)

    flat = grid.flatten()
    sorted = np.sort(flat)
    # if you have a threshold of 0.25, anything after 1/4 of the indexes should be indicated as blocks
    # if you have a threshold of 0.8, anything after 4/5 of the indexes should be indicated as blocks
    cap_i = int(((len(sorted) * (THRESHOLD))-1))
    cap = sorted[cap_i]

    # disable every grid whose cap is less than {cap}
    # new discovery: 0 is purple and 1 is yellow
    for i in range(len(grid_colors)):
        for j in range(len(grid_colors[i])):
            num = grid[i][j]
            grid_colors[i][j] = 1 if num >= cap else 0 # its a block

    print(grid_colors[6][4])
    print(grid[6][4])
    fig, ax = plt.subplots(1,1)
    ax.set_title('Montreal Crime grid with size {0}x{1}'.format(BLOCK_SIZE_X, BLOCK_SIZE_Y))
    x_labels, major_locator = get_labels(P1['x'], P4['x'], 5, BLOCK_SIZE_X)
    y_labels, minor_locator = get_labels(P1['y'], P3['y'], 5, BLOCK_SIZE_Y)

    ax.xaxis.set_major_locator(major_locator)
    ax.set_xticklabels(x_labels)
    ax.yaxis.set_major_locator(minor_locator)
    ax.set_yticklabels(y_labels)
    #pcm = ax.pcolormesh(np.swapaxes(grid_colors, 0, 1), edgecolors='white')
    pcm = ax.pcolormesh(grid_colors, edgecolors='white')
    x1, y1= 1,1
    x2, y2= 1,2
    ax.annotate("",
            color='white',
            xy=(x1, y1), xycoords='data',
            xytext=(x2, y2), textcoords='data',
            arrowprops=dict(facecolor='white', arrowstyle="-"))
    show_cell_counts(pcm, ax, grid, num_blocks_x, num_blocks_y)

    c = CrimeGraph((P1, P2, P3, P4))
    plt.show()
