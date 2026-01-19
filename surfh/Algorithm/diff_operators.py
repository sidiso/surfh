import numpy as np
from aljabr import LinOp

# much faster than Difference_Operator_Sep, and supposed to give same output
class NpDiff_r(LinOp): # dim = 3
    def __init__(self, maps_shape):
        super().__init__(
            ishape=maps_shape,
            oshape=maps_shape,
        )
    
    def forward(self, x):
        return - np.diff(np.pad(x, ((0, 0), (1, 0), (0, 0)), 'wrap'), axis=1)
    
    def adjoint(self, y):
        return np.diff(np.pad(y, ((0, 0), (0, 1), (0, 0)), 'wrap'), axis=1)

class NpDiff_c(LinOp): # dim = 3
    def __init__(self, maps_shape):
        super().__init__(
            ishape=maps_shape,
            oshape=maps_shape,
        )
    
    def forward(self, x):
        return - np.diff(np.pad(x, ((0, 0), (0, 0), (1, 0)), 'wrap'), axis=2)
    
    def adjoint(self, y):
        return np.diff(np.pad(y, ((0, 0), (0, 0), (0, 1)), 'wrap'), axis=2)