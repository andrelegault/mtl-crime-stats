# -------------------------------------------------------
# Assignment 1
# Written by Andre Parsons-Legault 40031363
# For COMP 472 Section JX – Summer 2020
# --------------------------------------------------------

from heapq import heappush, heappop


class AStar:
	"""
	Class with the A* algorithm.

	Uses a priority to maintain the open list of nodes.
	"""

	@staticmethod
	def search(grid, start, goal):
		""" Search algorithm used to find a path from one node to another. """
		open_list = []  # to be evaluate
		closed_list = []  # already evaluated
		heappush(open_list, (start.calc_f(grid, start, goal),
		                     start))  # sort by f, using tuples cuz cant compare CrimeCell since __gt__ already
		# implemented for sorting

		done = False

		while len(open_list) > 0:
			temp = heappop(open_list)
			if temp is None:
				print('not possible')
				return None
			else:
				cur = temp[1]
			closed_list.append(cur)

			if cur is goal:
				return cur

			for neighbour in cur.neighbours:
				neighbour.calc_f(grid, cur, goal)
				if neighbour.f >= 1000 or neighbour in closed_list:
					continue

				cur.calc_f(grid, cur, goal)
				if neighbour.f < cur.f or neighbour not in open_list:
					neighbour.f += cur.g
					neighbour.parent = cur
					if neighbour not in open_list:
						heappush(open_list, (neighbour.f, neighbour))

	@staticmethod
	def get_path(end, start):
		""" Returns a list containing every node on the way from start to end. """
		if not end:
			print('no path found')
		path = []

		current = end

		while current is not start:
			path.append((current.x, current.y))
			current = current.parent

		path.append((start.x, start.y))
		path.reverse()
		return path
