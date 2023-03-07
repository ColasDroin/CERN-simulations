import numpy as np

import xobjects as xo
import xtrack as xt
import xpart as xp

## Generate a simple line
line = xt.Line(
    elements=[
        xt.Drift(length=2.0),
        xt.Multipole(knl=[0, 1.0], ksl=[0, 0]),
        xt.Drift(length=1.0),
        xt.Multipole(knl=[0, -1.0], ksl=[0, 0]),
    ],
    element_names=["drift_0", "quad_0", "drift_1", "quad_1"],
)

## Attach a reference particle to the line (optional)
## (defines the reference mass, charge and energy)
line.particle_ref = xp.Particles(p0c=6500e9, q0=1, mass0=xp.PROTON_MASS_EV)  # eV

## Choose a context
context = xo.ContextCpu()  # For CPU
# context = xo.ContextCupy()      # For CUDA GPUs
# context = xo.ContextPyopencl()  # For OpenCL GPUs

## Transfer lattice on context and compile tracking code
tracker = line.build_tracker(_context=context)

## Build particle object on context
n_part = 200
particles = xp.Particles(
    p0c=6500e9,  # eV
    q0=1,
    mass0=xp.PROTON_MASS_EV,
    x=np.random.uniform(-1e-3, 1e-3, n_part),
    px=np.random.uniform(-1e-5, 1e-5, n_part),
    y=np.random.uniform(-2e-3, 2e-3, n_part),
    py=np.random.uniform(-3e-5, 3e-5, n_part),
    zeta=np.random.uniform(-1e-2, 1e-2, n_part),
    delta=np.random.uniform(-1e-4, 1e-4, n_part),
    _context=context,
)

## Track (saving turn-by-turn data)
n_turns = 100
tracker.track(particles, num_turns=n_turns, turn_by_turn_monitor=True)

## Turn-by-turn data is available at:
tracker.record_last_track.x
tracker.record_last_track.px


print(tracker)
