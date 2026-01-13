import os
import sys
import logging
import copy

sys.path.append("../")
adaptive_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
if adaptive_root not in sys.path:
    sys.path.append(adaptive_root)
data_dir = os.path.join(adaptive_root, "notebooks", "fake_data")

from simulation import fitters, simulators, utils, estimators

logger = logging.getLogger("cmdstanpy")
logger.addHandler(logging.NullHandler())
logger.propagate = False
logger.setLevel(logging.CRITICAL)

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
