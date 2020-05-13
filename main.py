# -------------------------------------------------------
# Assignment 1
# Written by Andre Parsons-Legault 40031363
# For COMP 472 Section JX – Summer 2020
# --------------------------------------------------------

# Using A* to find an optimal path between two coordinates on a map.
# https://basemaptutorial.readthedocs.io/en/latest/shapefile.html
# https://github.com/GeospatialPython/pyshp

import numpy as np
import matplotlib.pyplot as plt
import shapefile

if __name__ == '__main__':
    # size of grid should be <= 0.002 (recommended)
    shp = shapefile.Reader('crime_data/crime_dt', encoding='cp863')
    list_x = []
    list_y = []

    for sr in shp.shapeRecords():
        for x, y in sr.shape.points:
            list_x.append(x)
            list_y.append(y)

    plt.plot(list_x, list_y)
    plt.show()
    
