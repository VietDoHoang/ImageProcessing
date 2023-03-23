from PyQt5.QtWidgets import QMainWindow,QApplication, QFileDialog,QMessageBox
from PyQt5 import uic,QtGui, QtCore
import sys
import cv2   
import numpy as np   
import Loc_min_anh as lma
import loc_sac_net as lsn

class MY_UI(QMainWindow):
    def __init__(self):
        super().__init__()
        uic.loadUi("DaPhuongTien.ui",self)
        self.show()
        # xét event for Widgets
        #menufile
        self.actionOpen.triggered.connect(self.open_Img)
        self.actionSave.triggered.connect(self.save_Img)
        #menu loc min
        self.actionLoc_Trung_Binh.triggered.connect(self.loc_Trung_Binh)
        self.actionLoc_Trung_Vi.triggered.connect(self.loc_Trung_Vi)
        self.actionLoc_Min.triggered.connect(self.loc_Min)
        self.actionLoc_Max.triggered.connect(self.loc_Max)
        self.action_Loc_Gaussian.triggered.connect(self.loc_Gaussian)
        # Loc sac net
        self.actionBo_loc_Sobel.triggered.connect(self.loc_Sobel)
        self.actionBo_loc_laplacian.triggered.connect(self.loc_Laplacian)
        self.actionBo_Loc_RCG.triggered.connect(self.loc_Robert_Cross_Gradient)
        #zoom
        self.zoom.triggered.connect(self.zom_out)
        self.small_image.triggered.connect(self.small_img)
        #button 
        self.gammarScrollBar.valueChanged.connect(self.Gammar_)
        self.reset_image.clicked.connect(self.reset)
        self.convert_gray.clicked.connect(self.chuyen_anh_xam)
        self.quay_anh.clicked.connect(self.rotation)
        self.can_bang_anh.clicked.connect(self.histogram_Equalization)
        self.minCanny.valueChanged.connect(self.canny)
        self.maxCanny.valueChanged.connect(self.canny)

        self.image1=None
        self.image2=None
        self.imageTemp=None
        self.isGray = False
    
            
    def open_Img(self):
        file = QFileDialog.getOpenFileName(self,"Open Image",filter="*.png *.gif *.jpg *.jpeg *.tif")
        print(file[0])
        if file:
            self.loadImage(file[0])
        else:
            print("none file")
    def save_Img(self):
        file = QFileDialog.getSaveFileName(self,"Save Image","Image File(*.png)")
        if file :
            cv2.imwrite(file[0],self.image1)
        else :
            print('error')
    
    def loadImage(self,url):
        self.image1 = cv2.imread(url)
        self.imageTemp = self.image1
        self.image2 = self.image1
        self.displayImage()
        self.displayImage(2)
        
    def displayImage(self,window=1,resize =False):
        qformat = QtGui.QImage.Format_Indexed8
        if len(self.image1.shape) ==3 :
            if(self.image1.shape[2]) ==4:
                qformat = QtGui.QImage.Format_RGBA8888
            else:
                qformat = QtGui.QImage.Format_RGB888
        img = QtGui.QImage(self.image1, self.image1.shape[1],self.image1.shape[0],self.image1.strides[0], qformat)
        img = img.rgbSwapped()
        if window ==1:
            self.label_image_1.setPixmap(QtGui.QPixmap.fromImage(img))
            self.label_image_1.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
        if window ==2:
            self.label_image_2.setPixmap(QtGui.QPixmap.fromImage(img))
            self.label_image_2.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
    def zom_out(self):
        self.image1 = cv2.resize(self.image1,None,fx=1.5,fy=1.5,interpolation=cv2.INTER_CUBIC)
        self.image2 = self.image1
        self.displayImage(2)
        self.displayImage(1)
    def small_img(self):
        self.image1 = cv2.resize(self.image1,None,fx=0.5,fy=0.5,interpolation=cv2.INTER_CUBIC)
        self.image2 = self.image1
        self.displayImage(2)
        self.displayImage(1)
    
    def loc_Trung_Binh(self):
        self.image1 = lma.loc_trung_binh(self.image1)
        self.displayImage(2)
        self.image2 = self.image1
        
    def loc_Trung_Binh_Trong_So(self):
        self.image1 = lma.loc_trung_binh_trong_so(self.image1)
        self.displayImage(2)
        self.image2 = self.image1
    
    def loc_Trung_Vi(self):
        self.image1 = lma.loc_trung_vi(self.image1)
        self.displayImage(2)
        self.image2 = self.image1
    
    def loc_Min(self):
        # self.image1 = cv2.cvtColor(self.image1,cv2.COLOR_BGR2GRAY)
        self.image1 = lma.minimumBoxFilter(self.image1)
        self.displayImage(2)
        self.image2 = self.image1
        
    def loc_Max(self):
        # self.image1 = cv2.cvtColor(self.image1,cv2.COLOR_BGR2GRAY)
        self.image1 = lma.maximumBoxFilter(self.image1)
        self.displayImage(2)
        self.image2 = self.image1
    
    def loc_Gaussian(self):
        self.image1 = lma.loc_Gaussian(self.image1)
        self.displayImage(2)
        self.image2 = self.image1
        
    def loc_Sobel(self):
        self.image1 = lsn.bo_loc_Sobel(self.image1)
        self.displayImage(2)
        self.image2 = self.image1
        
    def loc_Laplacian(self):
        self.image1 = lsn.bo_loc_laplacian(self.image1)
        self.displayImage(2)
        self.image2 = self.image1
        
    def loc_Robert_Cross_Gradient(self):
        self.image1 = lsn.bo_loc_Robert_Cross_Gradient(self.image1)
        self.displayImage(2)
        self.image2 = self.image1
    
    def reset(self):
        self.image1 = self.imageTemp
        self.displayImage(1)
        self.displayImage(2)
        self.convert_gray.setChecked(False)
        self.image2 = self.image1
        
        
    def chuyen_anh_xam(self):
        if self.isGray == False:
            self.image1 = cv2.cvtColor(self.image1, cv2.COLOR_BGR2GRAY)
            self.displayImage(2)
            self.iGray = True
        
        
    def AboutMessage(self):
        QMessageBox.about(self,"Đỗ Hoàng Việt - 20198272 \n"
                          "Nguyễn Thành Long - 20198242")
    
    def rotation(self):
        rows ,cols , steps =  self.image1.shape
        M = cv2.getRotationMatrix2D((cols/2,rows/2),90,1)
        self.image1 = cv2.warpAffine(self.image1,M,(cols,rows))
        self.displayImage(1)
        self.displayImage(2)
        self.image2 = self.image1
    
    def histogram_Equalization(self):
        # chuyến sang dạng YUV
        img_temp = cv2.cvtColor(self.image1,cv2.COLOR_RGB2YUV)
        img_temp [:,2:,0] = cv2.equalizeHist(img_temp [:,2:,0])    
        self.image1 = cv2.cvtColor(img_temp,cv2.COLOR_YUV2RGB)
        self.displayImage(2)
        self.image2 = self.image1
        
    def Gammar_(self):
        self.image1 =self.image2
        gamma = self.gammarScrollBar.value()/100
        self.image1 = np.power(self.image1,gamma)
        max_val = np.max(self.image1.ravel())
        self.image1 = self.image1/max_val *255
        self.image1 = self.image1.astype(np.uint8)
        # self.image2 = self.image1
        self.displayImage(2)
        
    def canny(self):
        self.image1 = self.image2
        if self.isGray:
        # cannytemp = cv2.cvtColor(self.image1,cv2.COLOR_BGR2GRAY)
            self.image1 = cv2.Canny(self.image1,self.minCanny.value(),self.maxCanny.value())
            self.displayImage(2)
        else:
            cannytemp = cv2.cvtColor(self.image1 ,cv2.COLOR_BGR2GRAY)
            self.image1 = cv2.Canny(cannytemp,self.minCanny.value(),self.maxCanny.value())
            self.displayImage(2)
            self.isGray = True
            
        
        
if __name__ == '__main__':
    app = QApplication(sys.argv)
    my_ui = MY_UI()
    app.exec_()
    
    