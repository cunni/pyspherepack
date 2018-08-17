import pytest
from pyspherepack import Box
import numpy as np

# set up a module scoped box so that the test box (and really the pack()) is only instantiated once.
@pytest.fixture(scope="module")
def box_11packed():
    b = Box(11,n_iters=50000)
    b.pack()
    return b

def test_box_instance():
    # create Box
    b = Box(41) # 41 balls
    assert True

def test_pack_two():
    # create Box
    b = Box(2,n_iters=10000)
    b.pack()
    assert np.isclose(b.ball_radius(),np.sqrt(2)/2,.01)

def test_density(box_11packed):
    assert box_11packed.density() > 60 # permissive, just to make sure

def test_radius(box_11packed):
    assert box_11packed.ball_radius() > 0.15 # permissive, just to make sure


