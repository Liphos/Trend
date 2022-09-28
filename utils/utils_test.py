import numpy as np
from utils import logical_and_arrays


assert np.array_equal(logical_and_arrays([np.array([True, False])]), np.array([True, False]))
assert np.array_equal(logical_and_arrays([np.array([True, False]), np.array([True, True])]), np.array([True, False]))

assert np.array_equal(logical_and_arrays([np.array([True, False, False, True]), np.array([False, False, True, True])]), np.array([False, False, False, True]))