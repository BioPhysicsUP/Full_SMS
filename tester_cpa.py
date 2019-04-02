from smsh5 import H5dataset, Particle
from change_point import ChangePoints


file = H5dataset("LHCII_630nW.h5")
# test.particles[0].cpa.run_cpa(confidence=0.99)
file.particles[2].cpts.run_cpa(confidence=0.99, run_levels=True)
pass