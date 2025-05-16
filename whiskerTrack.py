import numpy as np
import cv2
import pandas as pd

polygonMask = cv2.imread("polygonMask.png", cv2.IMREAD_GRAYSCALE)
polygonMask = cv2.merge([polygonMask, polygonMask, polygonMask])
