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
from heapq import heappush, heappop
from decimal import Decimal, getcontext
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import geopandas

# from mpl_toolkits.axes_grid1 import ImageGrid
# import shapefile

# Defines high crime rate vs. low crime rate.
THRESHOLD = 0.5

# Size of a block in the grid
BLOCK_SIZE_X = Decimal(0.002)  # should be <= 0.002 (recommended)
BLOCK_SIZE_Y = Decimal(0.002)  # should be <= 0.002 (recommended)

NUM_LABELS_X = 5
NUM_LABELS_Y = 5

# four coordinates
P1 = {'x': Decimal(-73.59), 'y': Decimal(45.490)}
P2 = {'x': Decimal(-73.59), 'y': Decimal(45.530)}
P3 = {'x': Decimal(-73.55), 'y': Decimal(45.530)}
P4 = {'x': Decimal(-73.55), 'y': Decimal(45.490)}

COST_FREE_FREE = 1.0
COST_FREE_DIAG = 1.5
COST_FREE_BLOCK = 1.3


def get_num_blocks(num1, num2, size):
    return round((num1 - num2) / size)

def get_labels(start, end, num, size):
    # TODO: 0 is a quick-fix
    labels = [0, float(start)]
    current = start
    jump = (end - start) / num
    for i in range(num):
        current = current + jump
        labels.append(float(current))
    
    return labels, ticker.MultipleLocator(jump / size)
    
def show_cell_counts(pc, ax, grid, num_x, num_y):
    counter = 0
    for p in pc.get_paths():
        b_r, t_r, t_l, b_l, test = p.vertices
        x, y = p.vertices[-2:, :].mean(0)
        i = floor(counter / num_x)
        j = counter % num_y
        ax.text(x, y, grid.cells[i][j].crimes, c='white')
        counter += 1

# pq's https://towardsdatascience.com/introduction-to-priority-queues-in-python-83664d3178c3
class AStar:
    @staticmethod
    def search(grid, start, goal):
        open_list = [] # to be evaluate
        closed_list = [] # already evaluated
        heappush(open_list, (start.f(), start)) # sort by f, using tuples cuz cant compare CrimeCell since __gt__ already implemented for sorting

        done = False

        while not done:
            cur = heappop(open_list)
            closed_list.append(cur)

            if cur is goal: # 'is' compares references while == compares values
                return

            for neighbour in cur.neighbours():
                new_f = neighbour.f(grid)
                if neighbour.blocked is False or neighbour in closed_list:
                    continue

                if neighbour.f() < cur.f() or neighbour not in open_list:
                    neighbour.set_f()
                    neighbour.parent = cur
                    if neighbour not in open_list:
                        heappush(open_list, (neighbour.f(), neighbour))

class CrimeGrid:
    def __init__(self, points, num_blocks_x, num_blocks_y):
        self.cells = np.empty((num_blocks_x, num_blocks_y), dtype=CrimeCell)

        for i in range(num_blocks_x):
            for j in range(num_blocks_y):
                self.cells[i][j] = CrimeCell()

        for point in points:
            x_i, y_i = CrimeCell.get_pos(point.x, point.y)
            cell = self.cells[x_i][y_i]
            cell.x, cell.y = Decimal(x_i), Decimal(y_i)
            # self.cells[x_i][y_i].crimes = self.cells[x_i][y_i].crimes + 1
            cell.crimes += 1

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
        for i in range(len(self.cells)):
            for j in range(len(self.cells[i])):
                cell = self.cells[i][j]
                diff_sq = (Decimal(cell.crimes) - Decimal(avg)) ** 2
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
                    self.cells[i][j].blocked = True

    def get_mask(self, val=1):
        mask = np.zeros(self.cells.shape)
        for i in range(len(self.cells)):
            for j in range(len(self.cells[i])):
                if self.cells[i][j].blocked:
                    mask[i][j] = val
        return mask

    def set_neighbours(self):
        cell = self.cells[i][j]

        for x in range(len(self.grid)):
            for y in range(len(self.grid[0])):
                cell = self.cells[x][y]
                for i in range(-1, 2):
                    for j in range(-1, 2):
                        if i == 0 and j == 0: # node cannot have itself as a neighbour
                            continue

                        eff_x, eff_y = x+i, y+j
                        if (eff_x >= 0 and eff_x <= len(grid)-1) and (eff_y >= 0 and eff_y<= len(grid[0])-1): # valid node
                            cell.neighbours.append(self.cells[eff_x, eff_y])

class CrimeCell:
    def __init__(self, crimes=0, block=False):
        self.crimes = crimes
        self.blocked = block
        self.neighbours = []
        self.f = 0

    def __gt__(self, other):
        return self.crimes > other.crimes

    def __lt__(self, other):
        return self.crimes < other.crimes

    def __eq__(self, other):
        return self.crimes == other.crimes

    def __str__(self):
        return "crimes: {0}, block: {1}".format(self.crimes, self.blocked)

    def f(self, grid, cur, goal):
        cost_g = CrimeCell.calc_g(grid, cur, self)
        cost_h = self.__calc_h(goal)
        self.f = g + h
        return self.f

    """
    h is the estimate cost from one node to a target node.
    it underestimates the cost because it doesn't take into account blocks.
    """
    def __calc_h(self, goal):
        dx = abs(self.x - goal.x)
        dy = abs(self.y - goal.y)
        dmax = max(dx, dy)
        dmin = min(dx, dy)
        return (COST_FREE_DIAG * dmin) + COST_FREE_FREE * (dmax - dmin)

    """
    Using first points as anchors, get position in square.
    An input such as (-73.59, 45.49) returns 0, 0 as this point is located
    in the bottom left cell (using grid_size=0.002).
    """
    @staticmethod
    def get_pos(x, y):
        pos_x = floor((Decimal(x) - Decimal(P1['x'])) / BLOCK_SIZE_X)
        pos_y = floor((Decimal(y) - Decimal(P1['y'])) / BLOCK_SIZE_Y)
        return pos_x, pos_y

    """
    g is the cost to move from one node to a node.
    assumes nodes are neighbours.
    """
    @staticmethod
    def calc_g(grid, node, neighbour):
        one = Decimal(1)
        if (node.x != neighbour.x) and (node.y != neighbour.y): # diagonal movement
            if node.x < neighbour.x and node.y < neighbour.y: # top right
                return 1.5 if not node.blocked else 1000
            elif node.x > neighbour.x and node.y < neighbour.y: # top left
                return 1.5 if not grid.cells[(node.x)-one].blocked else 1000
            elif node.x < neighbour.x and node.y > neighbour.y: # bottom right
                return 1.5 if not grid.cells[node.x][(node.x)-Decimal(1)].blocked else 1000
            else: # bottom left
                return 1.5 if not grid.cells[(node.x)-one][(node.x)-one].blocked else 1000
        elif (node.x == neighbour.x) and (node.y != neighbour.y): # vertical movement
            if node.x == 0 or node.x == len(grid): # no border traversal allowed
                return 1000
            else: # inside grid
                if node.y < neighbour.y: # up
                    if node.blocked != grid.cells[(node.x)-one][node.y].blocked:
                        return 1.3
                    else:
                        return 1 if not node.blocked else 1000
                else: # down
                    lower_left_b = grid.cells[(node.x)-one][(node.y)-one].blocked
                    below_b = grid.cells[node.x][(node.y)-one].blocked
                    if lower_left_b != below_b:
                        return 1.3
                    else:
                        return 1 if not below_b else 1000
        elif (node.x != neighbour.x) and (node.y == neighbour.y): # horizontal movement
            if node.y == 0 or node.y == len(grid[0]): # no border traversal allowed
                return 1000
            else:
                if (node.x < neighbour.x): # left
                    lower_left_b = grid.cells[(node.x)-one][(node.y)-one].blocked
                    left_b = grid.cells[(node.x)-one][node.y].blocked
                    if lower_left_b != left_b:
                        return 1.3
                    return 1 if not left_b else 1000
                else:  # right
                    below_b = grid.cells[node.x][(node.y)-one].blocked
                    if node.blocked != below_b:
                        return 1.3
                    else:
                        return 1 if not node.blocked else 1000
        else:  # node == neighbour, i.e., itself
            return 0



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

# https://matplotlib.org/3.1.1/gallery/images_contours_and_fields/pcolor_demo.html# sphx-glr-gallery-images-contours-and-fields-pcolor-demo-py
if __name__ == '__main__':
    getcontext().prec = 4
    num_blocks_x = get_num_blocks(P4['x'], P1['x'], BLOCK_SIZE_X)
    num_blocks_y = get_num_blocks(P2['y'], P1['y'], BLOCK_SIZE_Y)

    gdf = geopandas.read_file('crime_data')
    grid = CrimeGrid(gdf['geometry'], num_blocks_x, num_blocks_y)

    fig, ax = plt.subplots(1,1)
    ax.set_title('Montreal Crime grid with size {0}x{1} and threshold set to {2}'.format(BLOCK_SIZE_X, BLOCK_SIZE_Y, THRESHOLD))
    x_labels, major_locator = get_labels(P1['x'], P4['x'], 5, BLOCK_SIZE_X)
    y_labels, minor_locator = get_labels(P1['y'], P3['y'], 5, BLOCK_SIZE_Y)

    ax.xaxis.set_major_locator(major_locator)
    print(x_labels)
    ax.set_xticklabels(x_labels)
    ax.yaxis.set_major_locator(minor_locator)
    ax.set_yticklabels(y_labels)

    avg = grid.avg()
    std = grid.std(avg)
    print("Average: {0}\nStandard deviation: {1}".format(avg, std))

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

    # TODO: make start and end depend on user input
    p_start, p_end = (-73.59, 45.49), (-73.588, 45.492)
    start_x, start_y, = CrimeCell.get_pos(p_start[0], p_start[1])
    end_x, end_y =CrimeCell.get_pos(p_end[0], p_end[1])
    start = grid.cells[start_x][start_y]
    end = grid.cells[end_x][end_y]
    ax.text(0, 0, "s", color='red')
    ax.text(1, 1, "e", color='red')

    print(start.y, end.y)
    print(end_x, end_y)

    g = CrimeCell.calc_g(grid, start, grid.cells[(start.x)+1, (start.y)+1])
    #print(g)

    #plt.show()
