import matplotlib as plt
import matplotlib.pyplot as plt
import itertools
import numpy as np
from sklearn.metrics import confusion_matrix
import pandas as pd
import os


LOG_PATH = './log/'
PERF_LOG_FILE = 'Performance statistics'


def writeTrainingPerformance(trainLoss, trainAccuracy, lbllist, predlist, features):
    logFile = PERF_LOG_FILE
    initLog(logFile)

    logText(logFile, "---Training performance---")

    writeTrainingLoss(trainLoss, logFile)
    writeTrainingAccuracy(trainAccuracy, logFile)
    plotConfusionMatrix(lbllist, predlist, features, 'train')


def writeTestingPerformance(testLoss, testAccuracy, lbllist, predlist, features):
    logFile = PERF_LOG_FILE
    logText(logFile, "---Testing performance---")

    print("Loss of the network on the test images: " + str(round(testLoss, 2)))
    print("Accuracy of the network on the test images: %d %%" % testAccuracy)
    logText(logFile, 'Loss: ' + str(round(testLoss, 2)))
    logText(logFile, 'Accuracy: ' + str(round(testAccuracy, 2)) + '%')
    plotConfusionMatrix(lbllist, predlist, features, 'test')


def writeTrainingLoss(loss, logFile):
    df = createDataFrame(loss, 'Loss')

    plt.title("Loss performance training")
    plt.plot(df["Epoch"], df["Loss"])
    # plt.xticks(rotation=45)
    plt.xlabel("Epoch")
    plt.ylabel("Loss (batch)")

    # plt.show()
    plt.savefig(LOG_PATH + 'Training loss.png')
    plt.close()

    logDataFrame(logFile, df)


def writeTrainingAccuracy(accuracy, logFile):
    df = createDataFrame(accuracy, 'Accuracy')

    plt.title("Accuracy performance training")
    plt.plot(df["Epoch"], df["Accuracy"])
    # plt.xticks(rotation=45)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (batch)")

    # plt.show()
    plt.savefig(LOG_PATH + 'Training accuracy.png')
    plt.close()

    logDataFrame(logFile, df)


def plotConfusionMatrix(lbllist, predlist, classes, type):
    confusionMatrix = confusion_matrix(lbllist, predlist)

    # print(confusionMatrix)

    plt.imshow(confusionMatrix, interpolation="nearest", cmap=plt.cm.Blues)
    if type == 'train':
        plt.title("Confusion matrix training")
    elif type == 'test':
        plt.title("Confusion matrix testing")
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = "d"
    thresh = confusionMatrix.max() / 2.
    for i, j in itertools.product(range(confusionMatrix.shape[0]), range(confusionMatrix.shape[1])):
        plt.text(j, i, format(confusionMatrix[i, j], fmt), horizontalalignment="center",
                 color="white" if confusionMatrix[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    # plt.show()
    if type == 'train':
        plt.savefig(LOG_PATH + 'Confusion matrix training.png')
    elif type == 'test':
        plt.savefig(LOG_PATH + 'Confusion matrix testing.png')
    plt.close()


def createDataFrame(array, type):
    dataFrame = pd.DataFrame()
    dataFrame['Epoch'] = list(range(1, len(array) + 1))
    dataFrame[type] = np.round(array, 2)
    return dataFrame


def logDataFrame(name, df):
    df.to_csv(LOG_PATH + name + '.txt', sep='\t', index=False, mode='a')
    logText(name, '')


def logText(name, text):
    f = open(LOG_PATH + name + '.txt', 'a')
    f.write(text + '\n')


def initLog(name):
    if not os.path.exists(LOG_PATH):
        os.makedirs(LOG_PATH)
    open(LOG_PATH + name + '.txt', 'w')