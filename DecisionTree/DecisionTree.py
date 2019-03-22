# 在https://github.com/PnYuan/Machine-Learning_ZhouZhihua基础上修改

import re
import math
import random
import numpy as np
import pandas as pd
import sys
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

sys.setrecursionlimit(100000)
# predefined threshold to stop the tree growing too deep
threshold_entropy = 0.25
threshold_gini = 0.25

class Node(object):
    def __init__(self, feature, label, leaves):
        self.feature = feature
        self.label = label
        self.leaves = leaves


def RandomList(low, high, length):
    randomList = []
    if length>=0:
        length = int(length)
        i = 0
        while i < length:
            a = random.randint(low, high)
            if a not in randomList:
                randomList.append(a)
                i += 1
    return randomList

# generate a decision tree
# criterion: 'entropy' or 'gini'
# purning: None or 'pre' or 'post'
def DecisionTree(dataSet, isContinuous, criterion, purning=None):
    if purning == None:
        root = TreeGenerate(dataSet, criterion, isContinuous)
    elif purning == 'pre':
        trainSet, vldSet, _, _ = train_test_split(dataSet, dataSet[dataSet.columns[-1]], test_size=1/3, random_state=1)
        root = TreePrePurning(trainSet, vldSet, criterion, isContinuous)
    elif purning == 'post':
        trainSet, vldSet, _, _ = train_test_split(dataSet, dataSet[dataSet.columns[-1]], test_size=1 / 3,
                                                  random_state=1)
        root = TreeGenerate(trainSet, criterion, isContinuous)
        # acc = PredictAccuracy(root, vldSet, isContinuous)
        TreePostPurning(root, vldSet, isContinuous)
    else:
        print('Wrong pruning command!')
        root = TreeGenerate(dataSet, criterion, isContinuous)
    return root


def TreePrePurning(trainSet, vldSet, criterion, isContinuous):
    newNode = Node(None, None, {})
    labels = trainSet[trainSet.columns[-1]]
    labelCount = CountLabels(labels)
    if labelCount:
        newNode.label = max(labelCount, key=labelCount.get)
        if len(labelCount) == 1 or len(labels) == 0:
            return newNode
        currentAcc = PredictAccuracy(newNode, vldSet, isContinuous)

        newNode.feature, divValue, gain = BestFeature(trainSet, criterion, isContinuous)
        if criterion == 'entropy' and gain < threshold_entropy:
            newNode.feature = None
            return newNode
        elif criterion == 'gini' and gain < threshold_gini: # 虽然理论上应根据gini指数选取，但实验发现gini指数阈值不好设定，容易导致递归过深而堆栈溢出
            newNode.feature = None
            return newNode

        if divValue == 0:
            valueCount = CountValue(trainSet[newNode.feature])
            for value in valueCount:
                subData = trainSet[trainSet[newNode.feature].isin([value])]
                subData = subData.drop(newNode.feature, 1)
                child = Node(None, None, {})
                childLabel = subData[subData.columns[-1]]
                childLabelCount = CountLabels(childLabel)
                child.label = max(childLabelCount, key=childLabelCount.get)
                newNode.leaves[value] = child
            newAcc = PredictAccuracy(newNode, vldSet, isContinuous)
            if newAcc > currentAcc:
                for value in valueCount:
                    subData = trainSet[trainSet[newNode.feature].isin([value])]
                    subData = subData.drop(newNode.feature, 1)
                    newNode.leaves[value] = TreeGenerate(subData, criterion, isContinuous)
            else:
                newNode.feature = None
                newNode.leaves = {}

        else:
            left = "<=%.3f" % divValue
            right = ">%.3f" % divValue
            leftSet = trainSet[trainSet[newNode.feature] <= divValue ]
            rightSet = trainSet[trainSet[newNode.feature] > divValue ]

            newLeft = Node(None, None, {})
            newRight = Node(None, None, {})
            leftLabelCount = CountLabels(leftSet[leftSet.columns[-1]])
            rightLabelCount = CountLabels(rightSet[rightSet.columns[-1]])
            newLeft.label = max(leftLabelCount, key=leftLabelCount.get)
            newRight.label = max(rightLabelCount, key=rightLabelCount.get)
            newNode.leaves[left] = newLeft
            newNode.leaves[right] = newRight
            newAcc = PredictAccuracy(newNode, vldSet, isContinuous)
            if newAcc > currentAcc:
                newNode.leaves[left] = TreeGenerate(leftSet, criterion, isContinuous)
                newNode.leaves[right] = TreeGenerate(rightSet, criterion, isContinuous)
            else:
                newNode.feature = None
                newNode.leaves = {}

    return newNode


def TreePostPurning(root, vldSet, isContinuous):
    if root.feature == None:  # reach leaves
        return PredictAccuracy(root, vldSet, isContinuous)
    acc = 0
    if isContinuous[int(root.feature)-1] == True:
        for key in list(root.leaves):
            num = re.findall(r"\d+\.?\d*", key)
            divValue = float(num[0])
            break
        left = "<=%.3f" % divValue
        right = ">%.3f" % divValue
        leftSet = vldSet[vldSet[root.feature] <= divValue]
        rightSet = vldSet[vldSet[root.feature] > divValue]
        accLeft = TreePostPurning(root.leaves[left], leftSet, isContinuous)
        accRight = TreePostPurning(root.leaves[right], rightSet, isContinuous)
        if accLeft == -1 or accRight == -1:
            return -1
        elif len(vldSet[vldSet.columns[-1]]) == 0:
            return 0
        else:
            acc += accLeft * len(leftSet[leftSet.columns[-1]]) / len(vldSet[vldSet.columns[-1]])
            acc += accRight * len(rightSet[rightSet.columns[-1]]) / len(vldSet[vldSet.columns[-1]])
    else:
        valueCount = CountValue(vldSet[root.feature])
        for value in list(valueCount):
            subSet = vldSet[vldSet[root.feature].isin([value])]  # get sub set
            if value in root.leaves:
                accSub = TreePostPurning(root.leaves[value], subSet, isContinuous)
            else:
                accSub = PredictAccuracy(root, subSet, isContinuous)
            if accSub == -1:  # -1 means no pruning back from this child
                return -1
            else:
                acc += accSub * len(subSet.index) / len(vldSet.index)
    # calculating the test accuracy on this node
    node = Node(None, root.label, {})
    newAcc = PredictAccuracy(node, vldSet, isContinuous)
    # check if need pruning
    if newAcc >= acc:
        root.feature = None
        root.leaves = {}
        return newAcc
    else:
        return -1


def TreeGenerate(dataSet, criterion, isContinuous):
    newNode = Node(None, None, {})
    labels = dataSet[dataSet.columns[-1]]
    labelCount = CountLabels(labels)
    if labelCount:
        newNode.label = max(labelCount, key=labelCount.get)
        if len(labelCount) == 1 or len(labels) == 0:
            return newNode

        newNode.feature, divValue, gain = BestFeature(dataSet, criterion, isContinuous)
        if criterion == 'entropy' and gain < threshold_entropy:
            newNode.feature = None
            return newNode
        elif criterion == 'gini' and gain < threshold_gini:
            newNode.feature = None
            return newNode

        if divValue == 0:
            valueCount = CountValue(dataSet[newNode.feature])
            for value in valueCount:
                subData = dataSet[dataSet[newNode.feature].isin([value])]
                subData = subData.drop(newNode.feature, 1)
                newNode.leaves[value] = TreeGenerate(subData, criterion, isContinuous)
        else:
            left = "<=%.3f" % divValue
            right = ">%.3f" % divValue
            leftSet = dataSet[dataSet[newNode.feature] <= divValue ]
            rightSet = dataSet[dataSet[newNode.feature] > divValue ]
            newNode.leaves[left] = TreeGenerate(leftSet, criterion, isContinuous)
            newNode.leaves[right] = TreeGenerate(rightSet, criterion, isContinuous)

    return newNode


def Predict(root, sample, isContinuous):
    while root.feature != None:
        if isContinuous[int(root.feature)-1] == True:
            for key in list(root.leaves):
                num = re.findall(r"\d+\.?\d*",key)
                divValue = float(num[0])
                break
            if sample[root.feature].values[0] <= divValue:
                key = "<=%.3f" % divValue
                root = root.leaves[key]
            else:
                key = ">%.3f" % divValue
                root = root.leaves[key]
        else:
            key = sample[root.feature].values[0]
            if key in root.leaves:
                root = root.leaves[key]
            else:
                break
    return root.label


def PredictAccuracy(root, sampleSet, isContinuous):
    if len(sampleSet.index) == 0:
        return 0
    acc = 0
    for i in sampleSet.index:
        label = Predict(root, sampleSet[sampleSet.index == i], isContinuous)
        if label == sampleSet[sampleSet.columns[-1]][i]:
            acc += 1
    return acc / len(sampleSet.index)


def CountLabels(labels):
    labelCount = {}
    for label in labels:
        if label in labelCount:
            labelCount[label] += 1
        else:
            labelCount[label] = 1
    return labelCount


def CountValue(data):
    valueCount = {}
    for v in data:
        if v in valueCount:
            valueCount[v] += 1
        else:
            valueCount[v] = 1
    return valueCount


def BestFeature(dataSet, criterion, isContinuous):
    if criterion == 'entropy':
        gain = 0
        bestFeature = None
        divValue = 0
        for feature in dataSet.columns[1:-1]:
            tempGain, tempDiv = CalculateGain(dataSet, feature, isContinuous)
            if tempGain >= gain:
                gain = tempGain
                bestFeature = feature
                divValue = tempDiv
        return bestFeature, divValue, gain
    elif criterion == 'gini':
        giniIndex = CalculateGini(dataSet[dataSet.columns[-1]])
        gini = giniIndex
        bestFeature = None
        divValue = 0
        for feature in dataSet.columns[1:-1]:
            tempGini, tempDiv = CalculateGiniIndex(dataSet, feature, isContinuous)
            if tempGini <= gini:
                gini = tempGini
                bestFeature = feature
                divValue = tempDiv
        gain = giniIndex - gini
        return bestFeature, divValue, gain
    else:
        print('Wrong criterion!')
        bestFeature = -1
        divValue = 0
        gain = 0
        return bestFeature, divValue, gain


def CalculateGain(dataSet, index, isContinuous):
    gain = CalculateEntropy(dataSet.values[:, -1])
    divValue = 0
    n = len(dataSet[index])
    if isContinuous[int(index)-1] == True:
        ent = {}
        dataSet = dataSet.sort_values([index], ascending=1)
        dataSet = dataSet.reset_index(drop=True)
        data = dataSet[index]
        label = dataSet[dataSet.columns[-1]]
        for i in range(n-1):
            # if data[i] != data[i+1]:
            div = (data[i] + data[i+1]) / 2
            ent[div] = (i+1) * CalculateEntropy(label[0:i])/n + (n-i-1) * CalculateEntropy(label[i+1:])/n
        divValue, maxEnt = min(ent.items(), key=lambda x:x[1])
        gain -= maxEnt
    else:
        data = dataSet[index]
        label = dataSet[dataSet.columns[-1]]
        valueCount = CountValue(data)
        for key in valueCount:
            keyLabel = label[data == key]
            gain -= valueCount[key] * CalculateEntropy(keyLabel) / n
    return gain, divValue


def CalculateEntropy(labels):
    ent = 0
    n = len(labels)
    labelCount = CountLabels(labels)
    for key in labelCount:
        ent -= (labelCount[key]/n) * math.log2(labelCount[key]/n)
    return ent


def CalculateGiniIndex(dataSet, feature, isContinuous):
    giniIndex = 0
    divValue = 0
    n = len(dataSet[feature])
    if isContinuous[int(feature)-1] == True:
        gini = {}
        dataSet = dataSet.sort_values([feature], ascending=1)
        dataSet = dataSet.reset_index(drop=True)
        data = dataSet[feature]
        label = dataSet[dataSet.columns[-1]]
        for i in range(n-1):
            # if data[i] != data[i+1]:
            div = (data[i] + data[i+1]) / 2
            gini[div] = (i+1) * CalculateGini(label[0:i])/n + (n-i-1) * CalculateGini(label[i+1:])/n
        divValue, giniIndex = min(gini.items(), key=lambda x: x[1])
    else:
        data = dataSet[feature]
        label = dataSet[dataSet.columns[-1]]
        valueCount = CountValue(data)
        for key in valueCount:
            keyLabel = label[data == key]
            giniIndex += valueCount[key] * CalculateGini(keyLabel) / n
    return giniIndex, divValue


def CalculateGini(labels):
    gini = 1
    n = len(labels)
    labelCount = CountLabels(labels)
    for key in labelCount:
        p = labelCount[key] / n
        gini -= p * p
    return gini


if __name__ == "__main__":

    with open("data/iris.csv", mode='r') as data_file:  # threshold: ent 0.3 gini 0.3
        dataSet = pd.read_csv(data_file, header=0)
    n = len(dataSet[dataSet.columns[-1]])
    isContinuous = [True, True, True, True]

    with open("data/hayes-roth.csv", mode='r') as data_file:  # threshold: ent 0.01 gini 0.01
        dataSet = pd.read_csv(data_file, header=0) 
    n = len(dataSet[dataSet.columns[-1]])
    isContinuous = [False, False, False]

    with open("data/BreastTissue.csv", mode='r') as data_file: # threshold: 0.1
        dataSet = pd.read_csv(data_file, header=0)
    isContinuous = [True, True, True, True, True, True, True, True, True]

    with open("data/Immunotherapy.csv", mode='r') as data_file:
        dataSet = pd.read_csv(data_file, header=0)
    isContinuous = [False, True, True, True, False, True, True]

    kf = KFold(n_splits=5)

    labels = dataSet[dataSet.columns[-1]]
    labelCount = CountLabels(labels)
    KFoldTrainSet = {}
    KFoldTestSet = {}
    flag = 0
    for label in labelCount:
        subSet = dataSet[dataSet[dataSet.columns[-1]] == label]
        n = len(subSet[subSet.columns[-1]])
        t = 0
        for train_index, test_index in kf.split(subSet):
            if flag == 0:
                KFoldTestSet[t] = subSet.iloc[test_index]
                KFoldTrainSet[t] = subSet.iloc[train_index]
                t += 1
            else:
                KFoldTestSet[t] = pd.concat([KFoldTestSet[t], subSet.iloc[test_index]])
                KFoldTrainSet[t] = pd.concat([KFoldTrainSet[t], subSet.iloc[train_index]])
                t += 1
        if flag == 0:
            flag = 1
    acc_id3 = np.zeros([5, 4])
    acc_cart = np.zeros([5, 4])
    for t in range(5):
        trainSet = KFoldTrainSet[t]
        testSet = KFoldTestSet[t]
        root = DecisionTree(trainSet, isContinuous, criterion='entropy')
        acc = PredictAccuracy(root, testSet, isContinuous)
        acc_id3[t, 0] = acc
        print(acc)
        root = DecisionTree(trainSet, isContinuous, criterion='entropy', purning='pre')
        acc = PredictAccuracy(root, testSet, isContinuous)
        acc_id3[t, 1] = acc
        print(acc)
        root = DecisionTree(trainSet, isContinuous, criterion='entropy', purning='post')
        acc = PredictAccuracy(root, testSet, isContinuous)
        acc_id3[t, 2] = acc
        print(acc)
        clf = DecisionTreeClassifier(criterion='entropy')
        clf.fit(trainSet[trainSet.columns[1:-1]], trainSet[trainSet.columns[-1]])
        y = clf.predict(testSet[testSet.columns[1:-1]])
        acc = accuracy_score(testSet[testSet.columns[-1]], y)
        acc_id3[t, 3] = acc
        print(acc)

        root = DecisionTree(trainSet, isContinuous, criterion='gini')
        acc = PredictAccuracy(root, testSet, isContinuous)
        acc_cart[t, 0] = acc
        print(acc)
        root = DecisionTree(trainSet, isContinuous, criterion='gini', purning='pre')
        acc = PredictAccuracy(root, testSet, isContinuous)
        acc_cart[t, 1] = acc
        print(acc)
        root = DecisionTree(trainSet, isContinuous, criterion='gini', purning='post')
        acc = PredictAccuracy(root, testSet, isContinuous)
        acc_cart[t, 2] = acc
        print(acc)
        clf = DecisionTreeClassifier(criterion='gini')
        clf.fit(trainSet[trainSet.columns[1:-1]], trainSet[trainSet.columns[-1]])
        y = clf.predict(testSet[testSet.columns[1:-1]])
        acc = accuracy_score(testSet[testSet.columns[-1]], y)
        acc_cart[t, 3] = acc
        print(acc)

    result = np.hstack([acc_id3, acc_cart])
    df = pd.DataFrame(result, columns=["id3", "id3-pre", "id3-post", "id3-sklearn", "cart", "cart-pre", "cart-post", "cart-sklearn"])
    df.to_csv("result.csv", index=True)
