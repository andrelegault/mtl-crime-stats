# -------------------------------------------------------
# Assignment 1
# Written by Andre Parsons-Legault 40031363
# For COMP 472 Section JX – Summer 2020
# --------------------------------------------------------

from decimal import Decimal, getcontext
import matplotlib.pyplot as plt
import geopandas
import utils
from crime_grid import CrimeGrid
from crime_cell import CrimeCell
from a_star import AStar

# f(h) : estimate of total cost through n to goal
# g(n) : actual cost from start to n, i.e., adding 1.5 or 1 depending if block or not
# h(n) : estimate cost from n to goal, using dx, dy
# f(n) = g(n) + h(n)
# maintain depth/cost count, give preference to shorter/least expensive paths
# heuristic algorithm is diagonal as per http://theory.stanford.edu/~amitp/GameProgramming/Heuristics.html
# since we support 8 directions


if __name__ == '__main__':
	getcontext().prec = 4
	utils.configure()
	num_blocks = utils.get_num_blocks(utils.P4['x'], utils.P1['x'], utils.BLOCK_SIZE)

	gdf = geopandas.read_file('crime_data')
	grid = CrimeGrid(gdf['geometry'], num_blocks)

	fig, ax = plt.subplots(1, 1)
	ax.set_title(
		'Montreal Crime grid with size {0}x{1} and threshold set to {2}'.format(utils.BLOCK_SIZE, utils.BLOCK_SIZE,
		                                                                        utils.THRESHOLD))
	x_labels, major_locator = utils.get_labels(utils.P1['x'], utils.P4['x'], 5, utils.BLOCK_SIZE)
	y_labels, minor_locator = utils.get_labels(utils.P1['y'], utils.P3['y'], 5, utils.BLOCK_SIZE)

	ax.xaxis.set_major_locator(major_locator)
	ax.set_xticklabels(x_labels)
	ax.yaxis.set_major_locator(minor_locator)
	ax.set_yticklabels(y_labels)

	avg = grid.avg()
	std = grid.std(avg)
	print("Average: {0}\nStandard deviation: {1}".format(avg, std))

	color_mask = grid.get_mask()
	pcm = ax.pcolormesh(color_mask, edgecolors='white')

	utils.add_cell_counts(ax, grid)

	start_i, start_j, = CrimeCell.get_pos(utils.p_start_x, utils.p_start_y)
	end_i, end_j = CrimeCell.get_pos(utils.p_end_x, utils.p_end_y)
	start = grid.cells[start_i][start_j]
	end = grid.cells[end_i][end_j]
	ax.text(start.y, start.x, "s", color='red')
	ax.text(end.y, end.x, "e", color='red')

	what = AStar.search(grid, start, end)
	path = AStar.get_path(end, start)
	utils.add_path(ax, path, start)

	plt.show()

	print("Program terminated")
