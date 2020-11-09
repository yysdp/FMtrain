import pandas as pd
import numpy as np
from math import exp
import numpy as np
from numpy import *
from random import normalvariate  # 正态分布
from datetime import datetime
import pandas as pd

from src.FM import preprocessData

preprocessData
data_path="ratings.csv"

datas = pd.read_csv(data_path).values
print(datas[0:10])
print(datas.shape)

