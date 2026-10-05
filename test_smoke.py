import numpy as np
from skin_model.utils import rgb_to_ita
print('numpy', np.__version__)
arr = np.array([[[200,160,140],[50,40,30]],[[180,140,100],[20,20,20]]], dtype=np.uint8)
ita = rgb_to_ita(arr)
print('ITA shape:', ita.shape)
print(ita)
import importlib
m = importlib.import_module('skin_model.train')
print('train module loaded, make_model exists:', hasattr(m, 'make_model'))
