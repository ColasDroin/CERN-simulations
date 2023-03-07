import json
import yaml
import time
import logging

import numpy as np
import pandas as pd

import xtrack as xt
import xpart as xp


def track(line_dict, particle_df, epsn_1, epsn_2, delta_max, n_turns):
    ##########################################
    # Read line, part_on_co, one-turn matrix #
    ##########################################

    p_co = xp.Particles.from_dict(line_dict["particle_on_tracker_co"])
    R_matrix = np.array(line_dict["RR_finite_diffs"])
    line = xt.Line.from_dict(line_dict)

    #####################################################
    # Get normalized coordinateds of particles to track #
    #####################################################

    r_vect = particle_df["normalized amplitude in xy-plane"].values
    theta_vect = particle_df["angle in xy-plane [deg]"].values * np.pi / 180  # [rad]

    A1_in_sigma = r_vect * np.cos(theta_vect)
    A2_in_sigma = r_vect * np.sin(theta_vect)

    ####################################################
    # Generate particles object (physical coordinates) #
    ####################################################

    particles = xp.build_particles(
        particle_on_co=p_co,
        x_norm=A1_in_sigma,
        y_norm=A2_in_sigma,
        delta=delta_max,
        R_matrix=R_matrix,
        scale_with_transverse_norm_emitt=(epsn_1, epsn_2),
    )
    particles.particle_id = particle_df.particle_id.values

    #################
    # Symplify line #
    #################

    line.remove_inactive_multipoles(inplace=True)
    line.remove_zero_length_drifts(inplace=True)
    line.merge_consecutive_drifts(inplace=True)
    # line.merge_consecutive_multipoles(inplace=True)

    #################
    # Build tracker #
    #################

    tracker = xt.Tracker(line=line)

    ############################
    # Save initial coordinates #
    ############################

    pd.DataFrame(particles.to_dict()).to_parquet("input_particles.parquet")

    ##########
    # Track! #
    ##########

    a = time.time()
    tracker.track(particles, turn_by_turn_monitor=False, num_turns=n_turns)
    b = time.time()

    print(f"Elapsed time: {b-a} s")
    print(f"Elapsed time per particle per turn: {(b-a)/particles._capacity/ n_turns*1e6} us")

    # pd.DataFrame(particles.to_dict()).to_parquet("output_particles.parquet")
