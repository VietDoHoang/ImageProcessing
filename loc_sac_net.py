import cv2    
import numpy as np  
import matplotlib.pyplot as plt 
import Loc_min_anh as lma
def bo_loc_Sobel(img):
    # Sobel theo hướng X
    locSobelX = np.array(([-1,-2,-1],
                      [ 0, 0, 0],
                      [ 1, 2, 1]), dtype="float")
    # bộ lọc Sobel theo hướng Y
    locSobelY = np.array(([-1, 0, 1],
                      [-2, 0, 2],
                      [ 1, 0, 1]), dtype="float")
    imageSobelX = cv2.filter2D(img,-1,locSobelX)
    imageSobelY = cv2.filter2D(img,-1,locSobelY)
    imageSobelXY = imageSobelX + imageSobelY
    imageSobel_ketqua = img + imageSobelXY
    return imageSobel_ketqua

def bo_loc_Robert_Cross_Gradient(img):
    # Bộ lọc theo hướng chéo 1
    loc_Robert_Cross1 = np.array(([0, 0, 0],
                              [0,-1, 0],
                              [0, 0, 1]), dtype="float")
    # Bộ lọc theo hướng chéo 2
    loc_Robert_Cross2 = np.array(([0, 0, 0],
                                [0, 0,-1],
                                [0, 1, 0]), dtype="float")
    img_Robert_Cross1 = cv2.filter2D(img,-1,loc_Robert_Cross1)
    img_Robert_Cross2 = cv2.filter2D(img,-1,loc_Robert_Cross2)
    img_Robert_Cross1_2 = img_Robert_Cross1+img_Robert_Cross2
    img_Robert_Cross_final = img + img_Robert_Cross1_2
    return img_Robert_Cross_final
    
def bo_loc_laplacian(img):
    # mặt nạ lọc Laplacian
    locLaplacian = np.array(([0, 1, 0],
                               [1, -4, 1],
                               [0, 1, 0]), dtype="float")
    imageLaplacian = cv2.filter2D(img,-1,locLaplacian)
    imageLaplacian_ketqua = img -imageLaplacian;
    return imageLaplacian_ketqua

