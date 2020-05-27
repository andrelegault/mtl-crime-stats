# -------------------------------------------------------
# Assignment 1
# Written by Andre Parsons-Legault 40031363
# For COMP 472 Section JX – Summer 2020
# --------------------------------------------------------

# Using A* to find an optimal path between two coordinates on a map.
# https://basemaptutorial.readthedocs.io/en/latest/shapefile.html
# https://github.com/GeospatialPython/pyshp

from math import floor, sqrt
from random import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import geopandas

#from mpl_toolkits.axes_grid1 import ImageGrid
#import shapefile

####################### BEGIN CONSTANTS #######################

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

COST_FREE_FREE = 1.0
COST_FREE_DIAG = 1.5
COST_FREE_BLOCK = 1.3

######################### END of CONSTANTS ######################### 


######################### BEGIN UTILS ######################### 
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

def f_func():
    return g_func() + h_func()

def g_func(point):
    cell = get_cell(point)
    if cell:
        pass

def get_labels(start, end, num, size):
    #TODO: 0 is a quick-fix
    labels = [0, start]
    current = start
    jump = (end - start) / num
    for i in range(num):
        current = current + jump
        labels.append(round(current, 3))
    
    return labels, ticker.MultipleLocator(jump / size)
    

def show_cell_counts(pc, ax, grid, num_x, num_y):
    counter = 0
    for p in pc.get_paths():
        b_r, t_r, t_l, b_l, test = p.vertices
        x, y = p.vertices[-2:,:].mean(0)
        ax.text(x, y, grid.cells[floor(counter / num_x)][counter % num_y].crimes, c='white')
        counter = counter + 1
######################### END of UTILS ######################### 


class CrimeGrid:
    def __init__(self, points, num_blocks_x, num_blocks_y):
        self.cells = np.empty((num_blocks_x, num_blocks_y), dtype=CrimeCell)

        for i in range(num_blocks_x):
            for j in range(num_blocks_y):
                self.cells[i][j] = CrimeCell()

        for point in points:
            x_i, y_i = CrimeCell.get_pos(point.x, point.y)
            cell = self.cells[x_i][y_i]
            cell.x, cell.y = x_i, y_i
            #self.cells[x_i][y_i].crimes = self.cells[x_i][y_i].crimes + 1
            cell.crimes = cell.crimes + 1

        self.set_crime_blocks(THRESHOLD)

    def avg(self):
        num = np.size(self.cells)
        total = 0.0
        for i in range(len(self.cells)):
            for j in range(len(self.cells[i])):
                cell = self.cells[i][j]
                total = total + cell.crimes

        return total / num

    def std(self, avg):
        num = np.size(self.cells)
        total_dev = 0
        for cell in np.nditer(self.cells, flags=['refs_ok']):
            diff_sq = (cell.crimes - avg) ** 2
            total_dev = total_dev + diff_sq

        return round(sqrt(total_dev / num), 2)

    def set_crime_blocks(self, threshold):
        flat = self.cells.flatten()
        avg = self.avg()

        # threshold of 0.25: anything after 1/4 of the indexes should be indicated as blocks
        # threshold of 0.8: anything after 4/5 of the indexes should be indicated as blocks
        sorted = np.sort(flat)
        cap_i = int(((len(sorted) * (threshold))-1))
        crime_cap = sorted[cap_i].crimes

        for i in range(len(self.cells)):
            for j in range(len(self.cells[i])):
                if self.cells[i][j].crimes >= crime_cap:
                    self.cells[i][j].block = True

    def get_mask(self, val=1):
        mask = np.zeros(self.cells.shape)
        for i in range(len(self.cells)):
            for j in range(len(self.cells[i])):
                if self.cells[i][j].block is True:
                    mask[i][j] = val
        return mask

    def link_cells(self):
        for i in range(len(grid.cells)):
            for j in range(len(grid.cells[i])):
                self.set_costs(self)

    def set_neighbors(self):
        self.graph[(x, y)] = {(x_other, y_other): distance((x, y), (x2, y2))}
        cell = self.cells[i][j]

        for x in range(len(self.grid)):
            for y in range(len(self.grid[0])):
                cell = self.cells[x][y]
                cell.neighbors = {}
                for i in range(-1, 2):
                    for j in range(-1, 2):
                        eff_x, eff_y = x+i, y+j
                        if (eff_x >= 0 and eff_x =< len(grid)-1) and (eff_y >= 0 and eff_y<=len(grid[0])-1):
                            self.add_neighbor(cell, self.cells[eff_x, eff_y])
                        """
                        if i > 0 and i < len(grid)-1:
                            left, right = # something
                            if j > 0 and j < len(grid[0])-1:
                                top_right, top_left, bot_left, bot_right = ...
                                diag_top_right, diag_top_left, diag_bot_right, diag_bot_left = ...
                        """
                        self.neighbors = {}
                        if i == 0:
                            left, diag_bot_left, diag_top_left = None, None, None
                            right = #something
                        elif i == len(grid)-1:
                            right, diag_bot_right, diag_top_right = None, None, None
                            left = #something
                        else:
                            right, left = #something and some other thing

                        if j == 0:
                            bot, diag_bot_right, diag_bot_left = None, None, None
                        if j == len(grid[0])-1:
                            top, diag_top_right, diag_top_left = None, None, None

                        bot, top, right, left = cells[i][j-1], cells[i][j+1], cells[i+1][j], cells[i-1][j]
                        diag_bot_right, diag_bot_left, diag_top_right, diag_top_left = cells[i+1][j-1], cells[i-1][j-1], cells[i+1][j+1], cells[i-1][j+1]
                        self.graph[cell][self.graph[]]

    def is_valid_neighbor(self, cell, other):
        if (cell.x != other.x) and (cell.y != other.y): # diagonal movement
            if cell.x < other.x and cell.y < other.y: # top right
                return cell.blocked is False
            if cell.x > other.x and cell.y < other.y: # top left
                return self.cells[cell.x-1].blocked is False
            if cell.x < other.x and cell.y > other.y: # bottom right
                return self.cells[cell.x][cell.x-1]
            else: # bottom left
                return self.cells[cell.x-1][cell.x-1]
        elif (cell.x == other.x) and (cell.y != other.y): # vertical movement
            if cell.y < other.y: # up
                return cell.blocked is False and self.cells[cell.x-1][cell.y].blocked is False
            else: #down
                return self.cells[cell.x-1][cell.y].blocked is False and self.cells[cell.x-1][cell.y-1].blocked is False
        elif (cell.x != other.x) and (cell.y == other.y): # horizontal movement
            if (cell.x < other.x):
                return cell.blocked is False
            else:
                return other.blocked is False
        else:
            print("Going to itself, what?")

                

class CrimeCell:
    def __init__(self, crimes=0, block=False):
        self.crimes = crimes
        self.block = block

    def __gt__(self, other):
        return self.crimes > other.crimes

    def __lt__(self, other):
        return self.crimes < other.crimes

    def __str__(self):
        return "crimes: {0}, block: {1}".format(self.crimes, self.block)

    """ Movements have already been filtered.
    n is assumed to be a valid neighbor of self."""
    def g(self, n):

    def f(self, n):
        return self.g(n) + self.h(n)


    @staticmethod
    def get_pos(x, y):
        x_i = floor((x - P1['x']) / BLOCK_SIZE_X)
        y_i = floor((y - P1['y']) / BLOCK_SIZE_Y)
        return x_i, y_i


""" This class is a dict that represents which points are bound to which grid cells. """

# pts_cell = { (x, y) -> (i, j) }
# and then
# cell_connections = { cell -> [{someCell -> distance}, {...}]}, would depend on if cell is block or not or if it's an edge or not
# f(n) = g(n) + h(n)
# f(h) : estimate of total cost through n to goal
# g(n) : actual cost from start to n, i.e., adding 1.5 or 1 depending if block or not
# h(n) : estimate cost from n to goal, using dx, dy
# maintain depth/cost count, give preference to shorter/least expensive paths
# heuristic algorithm is diagonal as per http://theory.stanford.edu/~amitp/GameProgramming/Heuristics.html
# since we support 8 directions
class CrimeGraph:
    def __init__(self, grid):
        #self.open_list = []
        #self.closed_list = []
        self.graph = {}
        for i in range(len(grid.cells)):
            for j in range(len(grid.cells[i])):
                self.set_costs(grid, i, j)
            self.graph[(x, y)] = {(x_other, y_other): distance((x, y), (x2, y2))}

    def set_costs(self, grid, i, j):
        cells = grid.cells
        cell = grid.cells[i][j]

        for x in range(-1, 2):
            for y in range(-1, 2):
                cell = grid[x][y]
        """
        if i > 0 and i < len(grid)-1:
            left, right = # something
            if j > 0 and j < len(grid[0])-1:
                top_right, top_left, bot_left, bot_right = ...
                diag_top_right, diag_top_left, diag_bot_right, diag_bot_left = ...
        """
                
        node = Node()
        if i == 0:
            left, diag_bot_left, diag_top_left = None, None, None
            right = #something
        elif i == len(grid)-1:
            right, diag_bot_right, diag_top_right = None, None, None
            left = #something
        else:
            right, left = #something and some other thing

        if j == 0:
            bot, diag_bot_right, diag_bot_left = None, None, None
        if j == len(grid[0])-1:
            top, diag_top_right, diag_top_left = None, None, None

        bot, top, right, left = cells[i][j-1], cells[i][j+1], cells[i+1][j], cells[i-1][j]
        diag_bot_right, diag_bot_left, diag_top_right, diag_top_left = cells[i+1][j-1], cells[i-1][j-1], cells[i+1][j+1], cells[i-1][j+1]
        self.graph[cell][self.graph[cells[


            

class Node:
    def __init__(self, ):
        self.x = x
        self.y = y
        self.parent = parent

    #def g(self, n):
        #return .parent.g() + self.parent.cost_to(self)

    def h(self, goal):
        dx = abs(self.x - goal.x)
        dy = abs(self.y - goal.y)
        return 

# https://matplotlib.org/3.1.1/gallery/images_contours_and_fields/pcolor_demo.html#sphx-glr-gallery-images-contours-and-fields-pcolor-demo-py
if __name__ == '__main__':
    num_blocks_x = get_num_blocks(P4['x'], P1['x'], BLOCK_SIZE_X)
    num_blocks_y = get_num_blocks(P2['y'], P1['y'], BLOCK_SIZE_Y)

    gdf = geopandas.read_file('crime_data')
    grid = CrimeGrid(gdf['geometry'], num_blocks_x, num_blocks_y)

    fig, ax = plt.subplots(1,1)
    ax.set_title('Montreal Crime grid with size {0}x{1} and threshold set to {2}'.format(BLOCK_SIZE_X, BLOCK_SIZE_Y, THRESHOLD))
    x_labels, major_locator = get_labels(P1['x'], P4['x'], 5, BLOCK_SIZE_X)
    y_labels, minor_locator = get_labels(P1['y'], P3['y'], 5, BLOCK_SIZE_Y)

    ax.xaxis.set_major_locator(major_locator)
    ax.set_xticklabels(x_labels)
    ax.yaxis.set_major_locator(minor_locator)
    ax.set_yticklabels(y_labels)

    color_mask = grid.get_mask()
    pcm = ax.pcolormesh(color_mask, edgecolors='white')

    x1, y1= 1,1
    x2, y2= 1,2
    ax.annotate("",
            color='white',
            xy=(x1, y1), xycoords='data',
            xytext=(x2, y2), textcoords='data',
            arrowprops=dict(facecolor='white', arrowstyle="-"))

    show_cell_counts(pcm, ax, grid, num_blocks_x, num_blocks_y)

    c = CrimeGraph(grid)

    plt.show()
