# 参考：https://github.com/PnYuan/Machine-Learning_ZhouZhihua/tree/master/ch4_decision_tree
# 将ID3和CART，剪枝融合在同一函数中了，通过在pruning中选择None,'pre','post'控制剪枝及criterion中选择'entropy''gini'选择划分依据
# 未解决问题：堆栈溢出

import re
import math
import random
import pandas as pd
import sys

sys.setrecursionlimit(5000)
isContinuous = [True, True, True, True]

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


def DecisionTree(dataSet, criterion, purning=None):
    if purning == None:
        root = TreeGenerate(dataSet, criterion)
    elif purning == 'pre':
        n = len(dataSet[dataSet.columns[-1]])
        trainIndex = RandomList(1, n, n/4)
        trainSet = dataSet.iloc[trainIndex]
        vldSet = dataSet.drop(trainIndex)
        root = TreePrePurning(trainSet, vldSet, criterion)
    elif purning == 'post':
        n = len(dataSet.iloc[0])
        trainIndex = RandomList(1, n, n/4)
        trainSet = dataSet.iloc[trainIndex]
        vldSet = dataSet.drop(trainIndex)
        root = TreeGenerate(trainSet, criterion)
        acc = PredictAccuracy(root, vldSet)
        print('vld accuracy before pruning:')
        print(acc)
        acc = TreePostPurning(root, vldSet)
        print('vld accuracy after pruning:')
        print(acc)
    else:
        print('Wrong pruning command!')
        root = TreeGenerate(dataSet, criterion)
    return root


def TreePrePurning(trainSet, vldSet, criterion):
    newNode = Node(None, None, {})
    labels = trainSet[trainSet.columns[-1]]
    labelCount = CountLabels(labels)
    if labelCount:
        newNode.label = max(labelCount, key=labelCount.get)
        if len(labelCount) == 1 or len(labels) == 0:
            return newNode
        currentAcc = PredictAccuracy(newNode, vldSet)

        newNode.feature, divValue = BestFeature(trainSet, criterion)
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
            newAcc = PredictAccuracy(newNode, vldSet)
            if newAcc > currentAcc:
                for value in valueCount:
                    subData = trainSet[trainSet[newNode.feature].isin([value])]
                    subData = subData.drop(newNode.feature, 1)
                    newNode.leaves[value] = TreeGenerate(subData, criterion)
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
            newAcc = PredictAccuracy(newNode, vldSet)
            if newAcc > currentAcc:
                newNode.leaves[left] = TreeGenerate(leftSet, criterion)
                newNode.leaves[right] = TreeGenerate(rightSet, criterion)
            else:
                newNode.feature = None
                newNode.leaves = {}

    return newNode


def TreePostPurning(root, vldSet):
    if root.feature == None:  # reach leaves
        return PredictAccuracy(root, vldSet)

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

        accLeft = TreePostPurning(root.leaves[left], leftSet)
        accRight = TreePostPurning(root.leaves[right], rightSet)

        if accLeft == -1 or accRight == -1:
            return -1
        else:
            acc += accLeft * len(leftSet.index) / len(vldSet.index)
            acc += accRight * len(rightSet.index) / len(vldSet.index)

    else:
        valueCount = CountValue(vldSet[root.feature])
        for value in list(valueCount):
            subSet = vldSet[vldSet[root.feature].isin([value])]  # get sub set
            accSub = TreePostPurning(root.leaves[value], subSet)
            if accSub == -1:  # -1 means no pruning back from this child
                return -1
            else:
                acc += accSub * len(subSet.index) / len(vldSet.index)

    # calculating the test accuracy on this node
    node = Node(None, root.label, {})
    newAcc = PredictAccuracy(node, vldSet)

    # check if need pruning
    if newAcc >= acc:
        root.feature = None
        root.leaves = {}
        return newAcc
    else:
        return -1


def TreeGenerate(dataSet, criterion):
    newNode = Node(None, None, {})
    labels = dataSet[dataSet.columns[-1]]
    labelCount = CountLabels(labels)
    if labelCount:
        newNode.label = max(labelCount, key=labelCount.get)
        if len(labelCount) == 1 or len(labels) == 0:
            return newNode

        newNode.feature, divValue = BestFeature(dataSet, criterion)
        if divValue == 0:
            valueCount = CountValue(dataSet[newNode.feature])
            for value in valueCount:
                subData = dataSet[dataSet[newNode.feature].isin([value])]
                subData = subData.drop(newNode.feature, 1)
                newNode.leaves[value] = TreeGenerate(subData, criterion)
        else:
            left = "<=%.3f" % divValue
            right = ">%.3f" % divValue
            leftSet = dataSet[dataSet[newNode.feature] <= divValue ]
            rightSet = dataSet[dataSet[newNode.feature] > divValue ]
            newNode.leaves[left] = TreeGenerate(leftSet, criterion)
            newNode.leaves[right] = TreeGenerate(rightSet, criterion)

    return newNode


def Predict(root, sample):
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
            key = sample[root.feaeture].values[0]
            if key in root.leaves:
                root = root.leaves[key]
            else:
                break
    return root.label


def PredictAccuracy(root, sampleSet):
    if len(sampleSet.index) == 0:
        return 0
    acc = 0
    for i in sampleSet.index:
        label = Predict(root, sampleSet[sampleSet.index == i])
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


def BestFeature(dataSet, criterion):
    if criterion == 'entropy':
        gain = 0
        for feature in dataSet.columns[1:-1]:
            tempGain, tempDiv = CalculateGain(dataSet, feature)
            if tempGain > gain:
                gain = tempGain
                bestFeature = feature
                divValue = tempDiv
    elif criterion == 'gini':
        gini = float('Inf')
        for feature in dataSet.columns[1:-1]:
            tempGini, tempDiv = CalculateGiniIndex(dataSet, feature)
            if tempGini < gini:
                gini = tempGini
                bestFeature = feature
                divValue = tempDiv
    else:
        print('Wrong criterion!')
        bestFeature = -1
        divValue = 0
        gain = 0
    return bestFeature, divValue


def CalculateGain(dataSet, index):
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


def CalculateGiniIndex(dataSet, feature):
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
        label = data[data.columns[-1]]
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
    with open("data/iris.csv", mode='r') as data_file:
        dataSet = pd.read_csv(data_file, header=0)
    root = DecisionTree(dataSet, criterion='entropy')
    acc = PredictAccuracy(root, dataSet)
    print(acc)
