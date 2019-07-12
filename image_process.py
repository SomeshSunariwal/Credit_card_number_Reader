import numpy as np
import cv2
import os
import random
from scipy.ndimage import convolve

class dataset:
    def __init__(self, directory):
        self.directory = directory
        print(self.directory)
    def make_dir(self, typ):
        if not(os.path.exists(self.directory)):
            os.mkdir(self.directory)
            return 0 
        for i in range(10):
            self.directory_1 = self.directory+str(typ)+"/"+str(i)
            print(self.directory_1)
            if not(os.path.exists(self.directory_1)):
                   os.makedirs(self.directory_1)
            else:
                   pass
                
class image_processs:
    def DigitAugmentation(self, frame, dim=32):
        frame = cv2.resize(frame, None, fx=2, fy=2, interpolation = cv2.INTER_CUBIC)
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        random_int = np.random.randint(0, 9)
        
        if random_int%2 == 0:
            frame = self.add_noise(frame)
        if random_int%3 == 0:
            frame = self.pixelated(frame)
        if random_int%2 == 0:
            frame = self.stretch(frame)
        frame = cv2.resize(frame, (dim, dim), interpolation = cv2.INTER_AREA)
        return frame
    
    def add_noise(self, frame):
        uni = random.uniform(0.01,0.05)
        ran = np.random.rand(frame.shape[0], frame.shape[1])
        noise = frame.copy()
        noise[ran < uni] = 0
        noise[ran > 1-uni] = 1
        return noise
    def pixelated(self, frame):
        dim = np.random.randint(8, 12)
        frame = cv2.resize(frame, (dim,dim), interpolation=cv2.INTER_AREA)
        frame = cv2.resize(frame, (16,16), interpolation=cv2.INTER_AREA)
        return frame
    def stretch(self, image):
        ran = np.random.randint(0,3)*2
        if np.random.randint(0,2) == 0:
            frame = cv2.resize(image, (32, ran+32), interpolation = cv2.INTER_AREA)
            return frame[int(ran/2):int(ran+32)-int(ran/2), 0:32]
        else:
            frame = cv2.resize(image, (ran+32, 32), interpolation = cv2.INTER_AREA)
            return frame[0:32, int(ran/2):int(ran+32)-int(ran/2)]
    def processesing(self, frame, inv=False):
        try:
            gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        except:
            gray_image = frame
            pass
        if inv == False:
            _, th2 = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:
            _, th2 = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        frame = cv2.resize(th2, (32, 32), interpolation= cv2.INTER_AREA)
        return frame
    
class prepare_dataset(image_processs):
    def __init__(self, image_path, dimension=[(None,None),(None,None)], shift_dis=(None, None), inv=None):
        self.path = cv2.imread(image_path,0)
        _, self.path = cv2.threshold(self.path, 0, 255, cv2.THRESH_BINARY_INV +cv2.THRESH_OTSU)
        self.top_x = dimension[0][0]
        self.top_y = dimension[0][1]
        self.bottom_x = dimension[1][0]
        self.bottom_y = dimension[1][1]
        self.topx_dis = shift_dis[0]
        self.botx_dis = shift_dis[1]
        self.bool = inv
    def train(self, n_class, n_samples, nseq):
        for i in range(n_class):
            if i > 0:
                self.top_x = self.top_x + self.topx_dis
                self.bottom_x = self.bottom_x + self.botx_dis
            roi = self.path[self.top_y:self.bottom_y, self.top_x:self.bottom_x]
            print("Complete train class :{}".format(i))
            for j in range(n_samples):
                roi2 = image_processs.DigitAugmentation(self,roi)
                roi_otsu = image_processs.processesing(self,roi2, inv = self.bool)
                #cv2.imshow("otsu", roi_otsu)
                cv2.imwrite("credit_data/train/"+str(i)+"./_"+str(nseq)+"_"+str(j)+".jpg", roi_otsu)
    def test(self, n_class, n_samples, nseq):
        for i in range(n_class):
            if i > 0:
                self.top_x = self.top_x + self.topx_dis
                self.bottom_x = self.bottom_x + self.botx_dis
            roi = self.path[self.top_y:self.bottom_y, self.top_x:self.bottom_x]
            print("Complete test class :{}".format(i))
            for j in range(n_samples):
                roi2 = image_processs.DigitAugmentation(self, roi)
                roi_otsu = image_processs.processesing(self, roi2, inv = self.bool)
                #cv2.imshow("otsu", roi_otsu)
                cv2.imwrite("credit_data/test/"+str(i)+"/_"+str(nseq)+"_"+str(j)+".jpg", roi_otsu)
                
class crop:
    def order_points(self, pts):
        rect = np.zeros((4, 2), dtype = "float32")
        s = pts.sum(axis = 1)

        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]

        diff = np.diff(pts, axis = 1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect

    def four_point_transform(self, image, pts):
        rect = self.order_points(pts)
        (tl, tr, br, bl) = rect
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))
        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))
        dst = np.array([[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1],
                                [0, maxHeight - 1]], dtype = "float32")
        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
        return warped


    def doc_Scan(self, image):
        image_1 = image.copy()
        image = cv2.resize(image, (image_1.shape[0], 500))
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(gray, 75, 200)
        contours, hierarchy  = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key = cv2.contourArea, reverse = True)[:5]
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < (image_1.shape[0]*500/3):
                print("Error Image Invalid")
                return("ERROR")
            area = cv2.contourArea(cnt)
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            # Find Square Contour
            if len(approx) == 4:
                square = approx
                break
        cv2.drawContours(image, [square], -1, (0, 255, 0), 2)
        cv2.imshow("Outline", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        warped = self.four_point_transform(image_1, square.reshape(4, 2) * (image_1.shape[0]/500))
        cv2.resize(warped, (640,403), interpolation = cv2.INTER_AREA)
        warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        warped = warped.astype("uint8") * 255
        cv2.imshow("Extracted image", warped)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return warped