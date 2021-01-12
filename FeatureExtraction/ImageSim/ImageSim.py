# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import glob
import random
import shutil
from PIL import Image
from PIL import ImageEnhance
import os


#######################################################################################################
class ImageSim:

    def SimImages(self, features):
        for feature in features:

            def plot_img(images, odir="./DatasetImages/"):
                i = int(len(os.listdir(odir)))
                print(i)
                for img in images:
                    Image.fromarray(img).save(odir + feature + (str(i) + ".jpg"))
                    print("image saved")
                    file_object = open("DatasetImages/mytrain.csv", 'a')
                    file_object.write(feature + str(i) + ".jpg," + str((features.index(feature))) + " \n")
                    file_object.close()
                    i += 1

            #######################################################################################################

            def imageEnhance(img, sel):

                if sel == 1:
                    return np.asarray(ImageEnhance.Contrast(Image.fromarray(img, 'RGB')).enhance(
                        np.random.choice(np.linspace(0.5, 1.5, 5))))
                elif sel == 2:
                    return Image.fromarray(img, 'RGB').rotate(np.random.choice([0, 90, 180, 270]))
                elif sel == 3:
                    return cv2.flip(img, np.random.choice([0, 1, -1])) if np.random.choice([0, 1]) else img
                return img

            def increase_enhance_dataset():
                out = [np.asarray(Image.open(img)) for img in paths]
                # increase and enhance dataset
                for i in range(50):
                    sel = random.randint(0, 4)
                    idx = random.randint(0, len(out) - 1)
                    out.append(np.asarray(imageEnhance(out[idx], sel)))

                # enhance images
                for i in range(1000):
                    sel = random.randint(0, 4)
                    idx = random.randint(0, len(out) - 1)
                    out[idx] = np.asarray(imageEnhance(out[idx], sel))

                return out

            #######################################################################################################

            def moveImages(paths, type):

                i = 0
                for path in paths:
                    print(path)
                    shutil.move(path, "ImageSim/usedImages/" + type + str(i) + ".jpg")
                    i += 1

            #######################################################################################################

            np.random.seed(1)

            paths = glob.glob('./ImageSim/inputImages/' + feature + '/*.jpg', recursive=True)

            if paths:
                out = [np.asarray(Image.open(img)) for img in paths]
                # increase and enhance dataset
                for i in range(50):
                    sel = random.randint(0, 4)
                    idx = random.randint(0, len(out) - 1)
                    out.append(np.asarray(imageEnhance(out[idx], sel)))

                # enhance images
                for i in range(1000):
                    sel = random.randint(0, 4)
                    idx = random.randint(0, len(out) - 1)
                    out[idx] = np.asarray(imageEnhance(out[idx], sel))

                plot_img(out)
                moveImages(paths, feature)
            else:
                print("no pictures in folder: " + feature)


if __name__ == '__main__':
    features = ["fist", "palm", "thumb"]
    test = ImageSim()
    test.SimImages(features)
