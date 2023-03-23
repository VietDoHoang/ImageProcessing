import cv2    
import numpy as np   
import matplotlib.pyplot as plt 

#  bộ lọc trung bình
def loc_trung_binh(img):
    locTB3x3 = np.array(([1/9, 1/9, 1/9],
                     [1/9, 1/9, 1/9],
                     [1/9, 1/9, 1/9]), dtype="float")
    # imgTB = Tich_chap(img,locTB3x3)
    imgTB = cv2.filter2D(img,-1,locTB3x3)
    return imgTB
#  bộ lọc trung bình có trọng số
def loc_trung_binh_trong_so(img):
    locTB3x3_trong_so = np.array(([1/16, 2/16, 1/16],
                              [2/16, 4/16, 2/16],
                              [1/16, 2/16, 1/16]), dtype="float")
    imgTB = cv2.filter2D(img,-1,locTB3x3_trong_so)
    return imgTB
# Định nghĩa bộ lọc Gaussian


def loc_Gaussian(img):
    locGaussian3x3 = np.array(([0.0751/4.8976, 0.1238/4.8976, 0.0751/4.8976],
                           [0.1238/4.8976, 0.2042/4.8976, 0.1238/4.8976],
                           [0.0751/4.8976, 0.1238/4.8976, 0.0751/4.8976]), dtype="float")
    imgGaussian = cv2.GaussianBlur(img,(5,5),0)
    imgGaussian = cv2.cvtColor(imgGaussian, cv2.COLOR_BGR2RGB)
    return imgGaussian
#hàm lọc trung vị
def loc_trung_vi(img):
    img_new = cv2.medianBlur(img,5)
    return img_new
def minimumBoxFilter(img):
    # Creates the shape of the kernel
    size = (3, 3)
    shape = cv2.MORPH_RECT
    kernel = cv2.getStructuringElement(shape, size)

    # Applies the minimum filter with kernel 3x3
    imgResult = cv2.erode(img, kernel)
    return imgResult

def maximumBoxFilter(img):
  # Creates the shape of the kernel
  size = (3,3)
  shape = cv2.MORPH_RECT
  kernel = cv2.getStructuringElement(shape, size)
  # Applies the maximum filter with kernel NxN
  imgResult = cv2.dilate(img, kernel)
  return imgResult
