import matplotlib as plt
import matplotlib.pyplot as plt
import itertools
import numpy as np
from sklearn.metrics import confusion_matrix
import pandas as pd
import os


LOG_PATH = './log/'
PERF_LOG_FILE = 'Performance statistics'


def getAdaptedOptimizer(optimizer, iter):
    for g in optimizer.param_groups:
        g['lr'] = 0.001 * iter * 2
        print(g['lr'])

    return optimizer


def writeTrainingPerformance(trainLossPerIteration, trainAccuracyPerIteration, learning_rates, times, labelListPerIteration, predictedListPerIteration, features):
    logFile = PERF_LOG_FILE
    initLog(logFile)

    logText(logFile, "---Training performance---")

    _, _, finalAccuracyList = getOptimizerLists(trainAccuracyPerIteration)

    best_optimizer = finalAccuracyList.index(max(finalAccuracyList))
    worst_optimizer = finalAccuracyList.index(min(finalAccuracyList))

    print('Best optimizer: ' + str(best_optimizer))
    print('Worst optimizer: ' + str(worst_optimizer))

    # writeTrainingOptimizers(learning_rates, logFile)
    plotTrainingAccuracy(trainAccuracyPerIteration)
    plotTrainingLossAndAccuracy(trainLossPerIteration[best_optimizer], trainAccuracyPerIteration[best_optimizer])
    logTrainingLossAndAccuracy(trainLossPerIteration, trainAccuracyPerIteration, learning_rates, times, logFile)
    plotConfusionMatrix(labelListPerIteration[best_optimizer], predictedListPerIteration[best_optimizer], features, 'train')


def writeTestingPerformance(testLoss, testAccuracy, labelList, predictedlist, features):
    logFile = PERF_LOG_FILE
    logText(logFile, "---Testing performance---")

    print("Loss of the network on the test images: " + str(round(testLoss, 2)))
    print("Accuracy of the network on the test images: %d %%" % testAccuracy)
    logText(logFile, 'Loss: ' + str(round(testLoss, 2)))
    logText(logFile, 'Accuracy: ' + str(round(testAccuracy, 2)) + '%')
    plotConfusionMatrix(labelList, predictedlist, features, 'test')


def writeTrainingOptimizers(learning_rates, logFile):
    #logText(logFile, "Optimizers")

    for index, lr in enumerate(learning_rates):
        logText(logFile, "Optimizer " + str(index + 1))
        logText(logFile, "Learning rate: " + str(lr))


def plotTrainingAccuracy(trainAccuracyPerIteration):
    dataFrame, fullAccuracyList, finalAccuracyList = getOptimizerLists(trainAccuracyPerIteration)

    styles = []
    colors = []

    for i in range(0, len(trainAccuracyPerIteration)):
        styles.append('.--')
        colors.append('grey')

    styles[finalAccuracyList.index(max(finalAccuracyList))] = '.-'
    colors[finalAccuracyList.index(max(finalAccuracyList))] = 'green'
    styles[finalAccuracyList.index(min(finalAccuracyList))] = '.-'
    colors[finalAccuracyList.index(min(finalAccuracyList))] = 'red'

    dataFrame.plot(x='Epoch', y=fullAccuracyList, style=styles, color=colors, legend=False)
    # plt.xticks(rotation=45)
    plt.title("Accuracy performance training")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (batch)")

    # plt.show()
    plt.savefig(LOG_PATH + 'Training accuracy.png')
    plt.close()


def plotTrainingLossAndAccuracy(loss, accuracy):
    dataFrame = pd.DataFrame()
    dataFrame['Epoch'] = list(range(1, len(accuracy) + 1))
    dataFrame['Loss'] = np.round(loss, 2)
    dataFrame['Accuracy'] = np.round(accuracy, 2)

    dataFrame.plot(x='Epoch', y=["Loss", "Accuracy"])
    # plt.xticks(rotation=45)
    plt.title("Loss and accuracy performance training")
    plt.xlabel("Epoch")
    plt.ylabel("Performance (batch)")

    # plt.show()
    plt.savefig(LOG_PATH + 'Training loss and accuracy.png')
    plt.close()

    #logDataFrame(logFile, dataFrame)


def logTrainingLossAndAccuracy(loss, accuracy, learning_rates, times, logFile):
    logText(logFile, "\n")

    for index in range(0, len(accuracy)):
        logText(logFile, "Optimizer " + str(index + 1))
        logText(logFile, "Learning rate: " + str(learning_rates[index]))

        dataFrame = pd.DataFrame()
        dataFrame['Epoch'] = list(range(1, len(accuracy[0]) + 1))
        dataFrame['Loss'] = np.round(getColumn(index, loss), 2)
        dataFrame['Accuracy'] = np.round(getColumn(index, accuracy), 2)

        logDataFrame(logFile, dataFrame)
        logText(logFile, "Training time: " + str(round(times[index]/1000, 2)) + " seconds")
        logText(logFile, "\n")


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


def getOptimizerLists(accuracy):
    dataFrame = pd.DataFrame()
    dataFrame['Epoch'] = list(range(1, len(accuracy[0]) + 1))

    fullAccuracyList = []
    finalAccuracyList = []

    n = 1
    for acc in accuracy:
        name = 'Optimizer' + str(n)
        fullAccuracyList.append(name)
        finalAccuracyList.append(acc[-1])
        dataFrame[name] = np.round(acc, 2)
        n += 1

    return dataFrame, fullAccuracyList, finalAccuracyList


def getColumn(col_num, data):
    newColumn = []
    for index, column in enumerate(data):
        if index == col_num:
            newColumn = column
    return newColumn


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