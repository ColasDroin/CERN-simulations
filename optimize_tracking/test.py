import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
from generate_line import generate_line


# tracking scans
r_min = 2
r_max = 10
radial_list = np.linspace(r_min, r_max, 2 * 16 * (r_max - r_min), endpoint=False)

n_angles = 5
theta_list = np.linspace(0, 90, n_angles + 2)[1:-1]

particle_list = [
    (particle_id, ii[0], ii[1])
    for particle_id, ii in enumerate(itertools.product(radial_list, theta_list))
]
particle_list = list(np.array_split(particle_list, 15))


tracker, line_bb_for_tracking_dict = generate_line()
