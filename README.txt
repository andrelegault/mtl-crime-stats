Assignment 1 for COMP472 summer 2020 by Andre Parsons-Legault (40031363)

The following modules are required prior to running the launch script (`main.py`):
- numpy
- geopandas
- decimal
- matplotlib

USAGE

The script can be launched by running: python main.py

The script will begin by prompting for the following:
- A threshold (defaults to 0.5)
- A block size (defaults to 0.002)
- A start location x (defaults to -73.59)
- A start location y (defaults to 45.49)
- An end location x (defaults to -73.586)
- An end location y (defaults to 45.492)

Once provided, the script will generate a path from the start location to the end location.

BUGS
- Goes in an infinite loop when the end location is not reachable.
- Doesn't support start/end on top and right edges.
