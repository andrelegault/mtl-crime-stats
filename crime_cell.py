# -------------------------------------------------------
# Assignment 1
# Written by Andre Parsons-Legault 40031363
# For COMP 472 Section JX – Summer 2020
# --------------------------------------------------------

import utils
from math import floor
from decimal import Decimal


class CrimeCell:
	""" Class representing one cell in the grid. """
	def __init__(self, crimes=0, blocked=False):
		self.crimes = crimes
		self.blocked = blocked
		self.neighbours = []

	def __gt__(self, other):
		return self.crimes > other.crimes

	def __lt__(self, other):
		return self.crimes < other.crimes

	def __str__(self):
		return "X: {0}, Y: {1}, crimes: {2}, blocked: {3}, f: {4} = {5} + {6}".format(self.x, self.y, self.crimes,
		                                                                              self.blocked, self.f, self.g,
		                                                                              self.h)

	def calc_f(self, grid, cur, goal):
		""" Computes f by summing g and h. """
		self.g = CrimeCell.calc_g(grid, cur, self, goal)
		self.h = self.calc_h(goal)
		self.f = self.g + self.h
		return self.f

	def calc_h(self, goal):
		""" Determines the h cost when going from self to a goal node, using diagonal distance. """
		dx = abs(self.x - goal.x)
		dy = abs(self.y - goal.y)
		dmax = max(dx, dy)
		dmin = min(dx, dy)
		return (utils.COST_FREE_DIAG * dmin) + utils.COST_FREE_FREE * (dmax - dmin)

	@staticmethod
	def get_pos(x, y):
		"""
		Using first points as anchors, gets position in grid.

		An input such as (-73.59, 45.49) returns 0, 0 as this point is located
		in the bottom left cell (using grid_size=0.002).
		(-73.588, 45.49) -> [0,1]
		"""
		pos_y = floor(float((Decimal(x) - Decimal(utils.P1['x']))) / utils.BLOCK_SIZE)
		pos_x = floor(float((Decimal(y) - Decimal(utils.P1['y']))) / utils.BLOCK_SIZE)
		return pos_x, pos_y

	@staticmethod
	def calc_g(grid, node, neighbour, goal):
		"""
		Determines the cost of g from going to a node to its neighbour.

		Follows what is shown in section 2.1.5 of the assignment instructions.
		A cost of 1000 represents an illegal movement.
		"""
		if node.x == neighbour.x and node.y == neighbour.y:
			return 0
		if (node.x != neighbour.x) and (node.y != neighbour.y):  # diagonal movement
			if node.x < neighbour.x and node.y < neighbour.y:  # top right
				return 1.5 if not node.blocked else 1000
			elif node.x < neighbour.x and node.y > neighbour.y:  # top left
				return 1.5 if not grid.cells[node.x][node.y - 1].blocked else 1000
			elif node.x > neighbour.x and node.y < neighbour.y:  # bottom right
				return 1.5 if not grid.cells[node.x - 1][node.y + 1].blocked else 1000
			else:  # bottom left
				return 1.5 if not grid.cells[node.x - 1][node.y - 1].blocked else 1000
		elif (node.x != neighbour.x) and (node.y == neighbour.y):  # vertical movement
			if node.y == 0 or node.y == len(grid.cells) - 1 or\
					(goal != neighbour and neighbour.x == 0 or neighbour.x == len(grid.cells)-1): # no border
				# traversal allowed, unless its a goal
				return 1000
			if node.x < neighbour.x:  # up
				if node.blocked != grid.cells[node.x][node.y - 1].blocked:
					return 1.3
				else:
					return 1 if not node.blocked else 1000
			else:  # down
				lower_left_b = grid.cells[node.x - 1][node.y - 1].blocked
				below_b = grid.cells[node.x - 1][node.y].blocked
				if lower_left_b != below_b:
					return 1.3
				else:
					return 1 if not below_b else 1000
		elif (node.x == neighbour.x) and (node.y != neighbour.y):  # horizontal movement
			if node.x == 0 or node.x == len(grid.cells[0]) - 1 or \
					(goal != neighbour and neighbour.y == 0 or neighbour.y == len(grid.cells[0])-1):  # no border
				# traversal allowed, unless its a goal
				return 1000
			if node.y > neighbour.y:  # left
				lower_left_b = grid.cells[node.x - 1][node.y - 1].blocked
				left_b = grid.cells[node.x][node.y - 1].blocked
				if lower_left_b != left_b:
					return 1.3
				return 1 if not left_b else 1000
			else:  # right
				below_b = grid.cells[node.x - 1][node.y].blocked
				if node.blocked != below_b:
					return 1.3
				else:
					return 1 if not node.blocked else 1000
