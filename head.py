
from matplotlib import pyplot as plt
import numpy as np
import cv2
import h5py
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.metrics import accuracy_score as acc
import pickle
import csv
import filecmp
import time
import zipfile as zp



# Constants.

IMG = 0
FNT = 1
TXT = 2
CBB = 3
WBB = 4



labels = [b'Open Sans',
          b'Sansation', 
          b'Titillium Web',
          b'Ubuntu Mono',
          b'Alex Brush']


labDict = {l : i for i, l in enumerate(labels)}
