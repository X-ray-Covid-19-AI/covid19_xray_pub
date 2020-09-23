from typing import NewType
import numpy as np

Image = NewType('Image', np.ndarray)
DirPath = ('DirPath', str)
FilePath = ('FilePath', str)