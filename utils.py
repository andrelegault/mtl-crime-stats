# -------------------------------------------------------
# Assignment 1
# Written by Andre Parsons-Legault 40031363
# For COMP 472 Section JX – Summer 2020
# --------------------------------------------------------

from decimal import Decimal
import matplotlib.ticker as ticker

# Defines high crime rate vs. low crime rate.
THRESHOLD = 0.5

# Size of a block in the grid
BLOCK_SIZE = 0.002

# Default start x location.
p_start_x = -73.59
# Default start y location.
p_start_y = 45.49

# Default end x location.
p_end_x = -73.586
# Default end y location.
p_end_y = 45.492

# Number of labels to display on the x-axis.
NUM_LABELS_X = 5
# Number of labels to display on the y-axis.
NUM_LABELS_Y = 5

# Four coordinates representing the area covered by the data.
P1 = {'x': Decimal(-73.59), 'y': Decimal(45.490)}
P2 = {'x': Decimal(-73.59), 'y': Decimal(45.530)}
P3 = {'x': Decimal(-73.55), 'y': Decimal(45.530)}
P4 = {'x': Decimal(-73.55), 'y': Decimal(45.490)}

# Cost of going between 2 non-blocked cells.
COST_FREE_FREE = 1.0
# Cost of going between a non-blocked cell, diagonally.
COST_FREE_DIAG = 1.5
# Cost of going between a blocked and non-blocked cell.
COST_FREE_BLOCK = 1.3


def configure():
	""" Configures user settings, using falling back to defaults if not provided. """
	global THRESHOLD, BLOCK_SIZE, p_start_x, p_start_y, p_end_x, p_end_y
	i_threshold = input("Please select a threshold (enter to default to 0.5): ")
	i_block_size = input("Please select a block size (enter to default to 0.002): ")
	i_p_start_x = input("Please input start x (enter to default tp -73.59): ")
	i_p_start_y = input("Please input start y (enter to default to 45.49): ")
	i_p_end_x = input("Please input end x (enter to default to -73.586): ")
	i_p_end_y = input("Please input end y (enter to default to 45.492): ")

	if i_threshold != '':
		THRESHOLD = float(i_threshold)
	if i_block_size != '':
		BLOCK_SIZE = float(i_block_size)
	if i_p_start_x != '':
		p_start_x = float(i_p_start_x)
	if i_p_start_y != '':
		p_start_y = float(i_p_start_y)
	if i_p_end_x != '':
		p_end_x = float(i_p_end_x)
	if i_p_end_y != '':
		p_end_y = float(i_p_end_y)


def get_num_blocks(num1, num2, size):
	""" Returns the number of blocks on one axis. """
	return round(float((num1 - num2)) / size)


def get_labels(start, end, num, size):
	""" Returns the labels for an axis. """
	labels = [0, float(start)]
	current = float(start)
	jump = float((end - start)) / num
	for i in range(int(num)):
		current = current + jump
		labels.append(round(float(current), 3))

	return labels, ticker.MultipleLocator(jump / size)


def add_cell_counts(ax, grid):
	""" Displays the total number of crimes for every block on the map. """
	for i in range(len(grid.cells)):
		for j in range(len(grid.cells[0])):
			ax.text(j + 0.25, i + 0.25, grid.cells[i][j].crimes, ha='center', va='center', c='white')


def add_path(ax, path, start):
	""" Displays the path obtained from A* algorithm. """
	for i in range(len(path) - 1):
		ax.annotate('', color='blue', xy=(path[i][1], path[i][0]), xycoords='data',
		            xytext=(path[i + 1][1], path[i + 1][0]), textcoords='data',
		            arrowprops=dict(ec='blue', facecolor='blue', arrowstyle="-"))
