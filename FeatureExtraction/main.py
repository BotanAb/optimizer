import sys
import traceback
from ImgPrep.HandImagePrep import HandImagePrep
from ImageSim.ImageSim import ImageSim
from ConvNeuralNetwork.CNN import CNN
from errorHandling import generic_exception_handler
from ConvNeuralNetwork.Modules.predict import predict
#from ConvNeuralNetwork.Modules.Performance import writeTrainingAccuracy
#from ConvNeuralNetwork.Modules.Performance import writeTrainingLossAccuracy

def main():
    try:
        features = ["fist", "palm", "thumb"]
        #prepimage = HandImagePrep()
        #prepimage.vidToImg(features)
        #prepimage.prepHandImages(features)
        #imagesim = ImageSim()
        #imagesim.SimImages(features)
        cnn = CNN()
        cnn.trainCNN(features)

        image = 'fist56.jpg'
        #print(predict(image, features))

        optimizers = [0.001, 0.002, 0.003, 0.004]
        loss = [[1.5, 1.1, 1.2, 0.9, 0.3], [1.5, 1.1, 1.2, 0.9, 0.3], [1.5, 1.1, 1.2, 0.9, 0.3], [1.5, 1.1, 1.2, 0.9, 0.3]]
        accuracy = [[3, 2, 3, 4, 6], [2, 3, 2, 5, 7], [4, 5, 1, 3, 5], [5, 1, 4, 6, 8]]

        #loss = [1.5, 1.1, 1.2, 0.9, 0.3]
        #accuracy = [3, 2, 3, 4, 6]

        #writeTrainingAccuracy(accuracy, optimizers, accuracy)
        #writeTrainingLossAccuracy(loss, accuracy, loss)

    except:
        generic_exception_handler()
    finally:
        print("einde programma")


if __name__ == '__main__':
    main()
