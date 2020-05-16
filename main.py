# -------------------------------------------------------
# Assignment 1
# Written by Andre Parsons-Legault 40031363
# For COMP 472 Section JX – Summer 2020
# --------------------------------------------------------

# Using A* to find an optimal path between two coordinates on a map.
# https://basemaptutorial.readthedocs.io/en/latest/shapefile.html
# https://github.com/GeospatialPython/pyshp

import numpy as np
from math import ceil
import matplotlib.pyplot as plt
from matplotlib import colors
import geopandas

#from mpl_toolkits.axes_grid1 import ImageGrid
import shapefile

# Defines high crime rate vs. low crime rate.
THRESHOLD = 0.5

# Size of a block in the grid
BLOCK_SIZE_X = 0.002 # should be <= 0.002 (recommended)
BLOCK_SIZE_Y = 0.002 # should be <= 0.002 (recommended)

# four coordinates
P1 = {'x': -73.59, 'y': 45.49}
P2 = {'x': -73.59, 'y': 45.53}
P3 = {'x': -73.55, 'y': 45.53}
P4 = {'x': -73.55, 'y': 45.49}

DEFAULT_X_RIGHT = -73.55

DEFAULT_Y_TOP = 45.53

class CrimePoint:
    color = 'purple' # to prefix with 'xdcd:'

class CrimeBlock:
    total = 0

    start_x = 0
    end_x = 0

    start_y = 0
    end_y = 0

    def __init__(self, data):
        for record in data:
            total = total + 1

class CrimeGrid:
    mean = 0
    std_dev = 0 # holds the stats for the entire grid
    blocks = [] # holds all the blocks in the grid

    def __init__(self, data):
        for record in data:
            pass

#TODO: put in CrimeGrid class
def calculate_area(p1, p2, p3, p4):
    total = (((p1['x'] * p2['y'] ) - (p1['y'] * p2['x'])) +
            ((p2['x'] * p3['y']) - (p2['y'] * p3['x'])) +
            ((p3['x'] * p4['y']) - (p3['y'] * p4['x'])))
    return abs(total) / 2

""" Using first points as anchors, get position in square. """
def get_pos(x, y):
    pos_x = ceil((x - P1['x']) / BLOCK_SIZE_X) - 1
    pos_y = ceil((y - P1['y']) / BLOCK_SIZE_Y) - 1
    return pos_x, pos_y

def avg(grid):
    total = 0
    size = 0
    for x in range(len(grid)):
        for y in range(len(grid[x])):
            total = total + grid[x][y]
    avg = total / (len(grid) * len(grid[0]))
    return avg

if __name__ == '__main__':
    num_blocks_x = round((P4['x'] - P1['x']) / BLOCK_SIZE_X)
    num_blocks_y = round((P2['y'] - P1['y']) / BLOCK_SIZE_Y)
    grid = np.zeros(shape=(num_blocks_x, num_blocks_y), dtype=int)


    gdf = geopandas.read_file('crime_data')
    pos = gdf['geometry']
    for point in pos:
        x_pos, y_pos = get_pos(point.x, point.y)
        grid[x_pos][y_pos] = grid[x_pos][y_pos] + 1

    print(grid.flatten())
    print(avg(grid))
    
    shp = shapefile.Reader('crime_data/crime_dt', encoding='cp863')
    list_x = []
    list_y = []


    for sr in shp.shapeRecords():
        for x, y in sr.shape.points:
            list_x.append(x)
            list_y.append(y)
    test = plt.hist2d(list_x, list_y, bins=40, cmin=0.5)
    print(test)
    plt.show()
