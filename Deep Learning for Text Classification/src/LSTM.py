from RNNwo import RNNwo
from RNNw import RNNw
import sys

mode = sys.argv[1]
if mode == "wo":
    RNNwo()
else:
    RNNw()