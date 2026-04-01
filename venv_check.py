import sys
import os
import torch
import pandas as pd
print(sys.executable)
print('torch', getattr(torch, '__version__', 'n/a'))
print('pandas', pd.__version__)
print('cwd', os.getcwd())
