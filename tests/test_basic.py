import pytest
from pyspherepack import Box
import numpy as np

def test_Box_instance():
    # create Box
    b = Box(41) # 41 balls
    assert True

def test_pack_two():
    # create Box
    # make this a fixture so pack can run and be used for multiple tests
    b = Box(2,n_iters=10000)
    b.pack()
    assert np.isclose(b.ball_radius(),np.sqrt(2)/2,.01)


