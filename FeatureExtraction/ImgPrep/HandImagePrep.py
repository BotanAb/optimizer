import glob
import importlib.util
import cv2

# Trying to import cv2, if this is not installed ffmpeg install will be checked
# try:
#
#     print("cv2 correctly installed")
#     cv2NotInstalled = False
# except:
#     print("cv2 not installed, trying ffmpeg now...")
#     cv2NotInstalled = True

import pandas as pd
import numpy as np
import math
import copy
import os
from pathlib import Path
import shutil

import imutils
import matplotlib
import matplotlib.pyplot as plt
from joblib.numpy_pickle_utils import xrange
from matplotlib.backends.backend_pdf import PdfPages

import skimage
from skimage import exposure
from skimage.util import invert
from skimage.filters import threshold_otsu, try_all_threshold
from skimage.color import rgb2gray
from skimage.util import img_as_float
from skimage.morphology import skeletonize


#######################################################################################################
class HandImagePrep:
    def capture(self, path, idx, feature):

        cap = cv2.VideoCapture(path)

        fps = cap.get(cv2.CAP_PROP_FPS)
        print("Framerate : {0} fps".format(fps))

        j = 0
        i = idx
        count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            print(ret)
            print(frame)

            if ret:
                if not self.hasMotionBlur(frame):
                    cv2.imwrite('./ImgPrep/inputImages/' + feature + '/' + feature + str(j) + str(i) + '.jpg', frame)
                    count += 10  # skip 9 frames
                else:
                    print('Frame ' + str(count + 1) + ' from ' + feature + ' is too blurry')
                    count += 1  # don't skip any frames if the current frame has motion blur

                cap.set(1, count)
                i += 1
            else:
                cap.release()
                # cv2.destroyAllWindows()
                break

    def checkIfCv2Exists(self):
        cv2_spec = importlib.util.find_spec('cv2')
        print(cv2_spec)
        # Attempting ffmpeg install if cv2 is not correctly installed

        if not cv2_spec:
            return False

        return True

    def checkIfFfmpegExists(self):
        try:
            os.system('ffmpeg')
            print("ffmpeg correctly installed")
        except:
            print(
                "Task terminated due to insufficient installed modules, please install cv2 or ffmpeg first and retry")
            ffmpegInstalled = False
            return ffmpegInstalled

    def hasMotionBlur(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        fm = self.varianceOfLaplacian(gray)

        # if the focus measure is less than the supplied threshold (10),
        # then the image should be considered "blurry"
        if fm < 10:
            return True
        return False

    def varianceOfLaplacian(self, image):
        # compute the Laplacian of the image and then return the focus
        # measure, which is simply the variance of the Laplacian
        return cv2.Laplacian(image, cv2.CV_64F).var()

    def moveVideo(self, path, feature):
        Path("./Videos/usedVideos/" + feature).mkdir(parents=True, exist_ok=True)
        shutil.move(path, "./Video/usedVideos/" + feature + '/' + feature + str(
            len(os.listdir("./Video/usedVideos/" + feature)) + 1) + ".mp4")

    def vidToImg(self, features):
        for feature in features:
            paths = glob.glob('./Videos/' + feature + '/*.mp4',
                              recursive=True)  # kijkt in video + feature en zoekt alle mp4 bestanden (array van strings)
            print("capturing frames from " + feature)
            # print(paths)

            if paths:
                Path('./ImgPrep/' + feature).mkdir(parents=True,
                                                   exist_ok=True)  # maakt nieuwe map aan met de featurenaam in IMGprep
                cv2Installed = self.checkIfCv2Exists()
                ffmpegInstalled = False
                if not cv2Installed:
                    ffmpegInstalled = self.checkIfFfmpegExists()
                for idx, path in enumerate(paths):
                    if cv2Installed:
                        self.capture(path, idx, feature)
                    elif ffmpegInstalled:
                        self.captureffmpeg(path)
                    else:
                        print("Er is niks geinstalleerd")
                        quit()
                        self.moveVideo(paths[idx], type)
            else:
                print("No videos found in map: " + feature)
        print("Done capturing frames")

    def captureffmpeg(self, path, feature, idx):
        cmd = 'ffmpeg -i ' + path + ' -f image2 "./ImgPrep/inputImages/' + feature + '/' + feature + str(
            idx) + '-' + '%05d' + '.jpg"'
        print(cmd)
        os.system(cmd)

    def calculateFingers(self, res):
        #  convexity defect
        hull = cv2.convexHull(res, returnPoints=False)
        if len(hull) > 5:
            defects = cv2.convexityDefects(res, hull)
            if defects is not None:
                cnt = 0
                for i in range(defects.shape[0]):  # calculate the angle
                    s, e, f, d = defects[i][0]
                    start = tuple(res[s][0])
                    end = tuple(res[e][0])
                    far = tuple(res[f][0])
                    a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
                    b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
                    c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
                    angle = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c))  # cosine theorem
                    if angle <= math.pi / 2:  # angle less than 90 degree, treat as fingers
                        cnt += 1
                if cnt > 0:
                    return cnt + 1
                else:
                    return 0
        return 0

    def do_smoothing(self, image):
        p2, p98 = np.percentile(image, (2, 98))
        img_rescale = exposure.rescale_intensity(image, in_range=(p2, p98))
        return cv2.bilateralFilter(img_rescale, 5, 50, 100)  # Smoothing

    def remove_background(self, image):
        backgroundremoval = cv2.createBackgroundSubtractorMOG2(0, 50)
        fgmask = backgroundremoval.apply(image)
        kernel = np.ones((3, 3), np.uint8)
        fgmask = cv2.erode(fgmask, kernel, iterations=1)
        return cv2.bitwise_and(image, image, mask=fgmask)

    def create_skinmask(self, image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower = np.array([0, 48, 80], dtype="uint8")
        upper = np.array([20, 255, 255], dtype="uint8")
        skinMask = cv2.inRange(hsv, lower, upper)
        return cv2.blur(skinMask, (2, 2))

    def getcontoursandhull(self, image):
        ret, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY)
        skinMask1 = copy.deepcopy(thresh)  # Thresholden
        contours, hierarchy = cv2.findContours(skinMask1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # Contour
        return contours, hierarchy

    def moveImages(self, paths, type):
        i = 0
        for path in paths:
            print(path)
            shutil.move(path, "./ImgPrep/usedImages/" + type + str(i) + ".jpg")
            i += 1

    #######################################################################################################
    def prepHandImages(self, features):
        for feature in features:
            paths = glob.glob('./ImgPrep/inputImages/' + feature + '/*.jpg', recursive=True)

            print(feature)
            print(paths)

            j = 0
            for path in paths:

                outdir = "./ImageSim/inputImages/" + feature + "/"
                Path(outdir).mkdir(parents=True, exist_ok=True)
                oFnam = outdir + os.path.basename(path)

                #######################################################################################################
                #
                # Image preparation
                #
                #######################################################################################################
                print("Get " + path)
                # print(path)
                img = cv2.imread(path)

                # plt.imshow(img)
                # plt.show()

                smoothing = self.do_smoothing(img)
                # plt.imshow(smoothing)
                # plt.show()

                background_removal = self.remove_background(smoothing)
                # plt.imshow(background_removal)
                # plt.show()

                skinMask = self.create_skinmask(background_removal)
                # plt.imshow(skinMask)
                # plt.show()

                contours, hierarchy = self.getcontoursandhull(skinMask)
                # print(contours)
                # print(hierarchy)

                length = len(contours)
                maxArea = -1
                drawing = np.zeros(img.shape, np.uint8)

                if length > 0:

                    for index, c in enumerate(contours):
                        area = cv2.contourArea(c)

                        if area < 250:  # Om kleine area's te skippen
                            continue
                        if area > maxArea:
                            maxArea = area
                            maxC = c
                            res = contours[index]

                    if maxArea > 0:
                        hull = cv2.convexHull(res)  # Make hull
                        count = 1  # calculateFingers(res, drawing)
                        if count > 0:  # Maak alleen skin als vingers er zijn
                            # print(count)

                            # Draw contours
                            cv2.drawContours(drawing, [hull], 0, (0, 0, 255), 2)  # Maak bounding hull
                            cv2.drawContours(drawing, [res], 0, (255, 0, 0), 2)  # Maak bounding hull

                            # Maak bounding box
                            bounding_rect = cv2.boundingRect(maxC)
                            x = bounding_rect[0] - 100
                            y = bounding_rect[1] - 100
                            xlen = bounding_rect[2] + (2 * 100)
                            ylen = bounding_rect[3] + (2 * 100)
                            if y < 0:
                                y = 0
                            if x < 0:
                                x = 0

                            cv2.rectangle(img, (x, y),
                                          (x + xlen, y + ylen),
                                          (255, 255, 255),
                                          2)
                            if xlen > 400 or ylen > 400:
                                try:
                                    # Crop image
                                    crop_img = img[y:y + ylen,
                                               x:x + xlen]
                                    cv2.imwrite(oFnam, crop_img)
                                    j += 1
                                    # print("Save "+oFnam)
                                except:
                                    print("er gaat iets fout")
                # plt.imshow(img)
                # plt.show()

                # plt.imshow(drawing)
                # plt.show()

                # plt.imshow(crop_img)
                # plt.show()

                # cv2.imshow("drawing", drawing)
                # cv2.waitKey(0)
        # moveImages(paths,feature)


#######################################################################################################

print(__name__)
if __name__ == '__main__':
    # features = ["fist", "palm", "thumb"]
    imageprep = HandImagePrep()
    # imageprep.vidToImg(features)
    # imageprep.prepHandImages(features)
    # imageprep.checkIfCv2Exists()