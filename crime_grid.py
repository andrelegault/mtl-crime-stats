# -------------------------------------------------------
# Assignment 1
# Written by Andre Parsons-Legault 40031363
# For COMP 472 Section JX – Summer 2020
# --------------------------------------------------------

from crime_cell import CrimeCell
import utils
import numpy as np
from decimal import Decimal
from math import sqrt


class CrimeGrid:
	""" Class containing the cells representing the plot. """
	def __init__(self, points, num_blocks):
		""" Create the grid, and count the number of crimes in each cell. """
		self.cells = np.empty((num_blocks, num_blocks), dtype=CrimeCell)

		for i in range(num_blocks):
			for j in range(num_blocks):
				self.cells[i][j] = CrimeCell()
				self.cells[i][j].x = i
				self.cells[i][j].y = j

		for point in points:
			x_i, y_i = CrimeCell.get_pos(point.x, point.y)
			cell = self.cells[x_i][y_i]
			# self.cells[x_i][y_i].crimes = self.cells[x_i][y_i].crimes + 1
			cell.crimes += 1

		self.set_crime_blocks(utils.THRESHOLD)
		self.set_neighbours()

	def print_crimes(self):
		""" Debugging method used to show the number of crimes in each cell. """
		crimes = np.zeros((20, 20))
		for i in range(len(self.cells)):
			for j in range(len(self.cells[0])):
				crimes[i][j] = self.cells[i][j].crimes

		print(crimes)

	def avg(self):
		""" Computes the average number of crimes in the grid. """
		num = np.size(self.cells)
		total = 0.0
		for i in range(len(self.cells)):
			for j in range(len(self.cells[i])):
				cell = self.cells[i][j]
				total = total + cell.crimes

		return total / num

	def std(self, avg):
		""" Computes the standard deviation of the number of crimes in the grid. """
		num = np.size(self.cells)
		total_dev = 0
		for i in range(len(self.cells)):
			for j in range(len(self.cells[i])):
				cell = self.cells[i][j]
				diff_sq = float((Decimal(cell.crimes) - Decimal(avg))) ** 2
				total_dev = total_dev + diff_sq

		return round(sqrt(total_dev / num), 2)

	def set_crime_blocks(self, threshold):
		""" Determines which blocks are crime blocks based on the threshold. """
		flat = self.cells.flatten()

		# threshold of 0.25: anything after 1/4 of the indexes should be indicated as blocks
		# threshold of 0.8: anything after 4/5 of the indexes should be indicated as blocks
		sorted = np.sort(flat)
		cap_i = int(((len(sorted) * (threshold)) - 1))
		crime_cap = sorted[cap_i].crimes
		print(crime_cap)

		for i in range(len(self.cells)):
			for j in range(len(self.cells[i])):
				if self.cells[i][j].crimes >= crime_cap:
					self.cells[i][j].blocked = True

	def get_mask(self, val=1):
		"""
		Creates the color mask using grid.

		IF a node is a block, its entry in the color mask is set to 1.
		"""
		mask = np.zeros(self.cells.shape)
		for i in range(len(self.cells)):
			for j in range(len(self.cells[i])):
				if self.cells[i][j].blocked:
					mask[i][j] = val
		return mask

	def set_neighbours(self):
		""" Sets the neighbours attribute of each cell. """
		for x in range(len(self.cells)):
			for y in range(len(self.cells[0])):
				cell = self.cells[x][y]
				for i in range(-1, 2):
					for j in range(-1, 2):
						if i == 0 and j == 0:  # node cannot have itself as a neighbour
							continue

						eff_x, eff_y = x + i, y + j
						if (0 <= eff_x <= len(self.cells) - 1) and (0 <= eff_y <= len(self.cells[0]) - 1):  # valid node
							cell.neighbours.append(self.cells[eff_x][eff_y])
