"""Handles version imports
   NOTE: if using `tensorflow.keras` imports, use `os.environ["TF_KERAS"] = '1'`,
         else the default '0' will be assumed for `keras` imports.
"""
import os
import tensorflow as tf

try:
    import keras
    KERAS_23 = bool(keras.__version__[:3] == '2.3')
except:
    KERAS_23 = None

TF_KERAS = bool(os.environ.get("TF_KERAS", '0') == '1')
TF_2 = bool(tf.__version__[0] == '2')

from .optimizers_v2 import CustomOptimizer, NCustomOptimizer, SGDW
from .utils import get_weight_decays, fill_dict_in_order
from .utils import reset_seeds, K_eval

__version__ = '1.38'
