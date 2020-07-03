from CNNwo import CNNwo
from CNNw import CNNw
import sys

mode = sys.argv[1]
if mode == "wo":
    CNNwo()
else:
    CNNw()