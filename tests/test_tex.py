import pytest
from pyspherepack import ManyBox
import numpy as np


def test_tex_best():
    ManyBox.tex_best(clamp_edge=10.0,scaled_rad=10.25)    
    assert True
