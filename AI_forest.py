# -*- coding: utf-8 -*-
"""
Created on Sun Jun 13 20:59:04 2021

@author: vincent
"""
""" 
Some Basic Data

number: 150
index: sepal_length,sepal_width,petal_length,petal_width,species
species: Iris-setosa, Iris-versicolor, Iris-virginica 

TREE
leftChild: <= thrashold
rightChild: > thrashold
"""

import csv
from random import shuffle, randint, sample
from copy import deepcopy
import matplotlib.pyplot as plt

"""
Adjestable Variable
analysing used
"""
TEST_TRAIN_RATIO = 1/5
ATT_BAGGING_RATIO = 3/4
SAM_BAGGING_RATIO = 4/5

CROSS_VALID_SPLIT_NUM = 10
MULTI_VALID_TIME = 10

MIN_SAMPLE = 5
MIN_GINI = 0.1
MAX_LEVEL = 3

TREE_NUM = 3

"""
Non-adjestable Variable
"""
DATA_NUM = 150
TRAIN_NUM = int(DATA_NUM * (1-TEST_TRAIN_RATIO))
TEST_NUM = int(DATA_NUM * TEST_TRAIN_RATIO)

ATT_NUM = 4
ATT_BAGGING_NUM = int(ATT_BAGGING_RATIO * ATT_NUM)
ATT_REMOVE_NUM = int((1 - ATT_BAGGING_RATIO) * ATT_NUM)

SAM_NUM = int(SAM_BAGGING_RATIO * TRAIN_NUM)

ATTRIBUTES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
FILE_PATH = "./iris.csv"

TEST_DATA_NUM = int(DATA_NUM / CROSS_VALID_SPLIT_NUM)

TOTAL_GENERATE_TREE_NUM = TREE_NUM * CROSS_VALID_SPLIT_NUM * MULTI_VALID_TIME

"""
Non-adjestable Variable
Record Used Global Variables
"""
Tree_Num = 0

Shallowest_Tree_Level = 1000
Deepest_Tree_Level = 0
Average_Tree_Level = 0

Largest_Tree_Node_Num = 0
Smallest_Tree_Node_Num = 1000
Average_Tree_Node_Num = 0

tree_Inspect = list()

"""
Adjestable Variable
Relation Used Global Variables
"""
RELATION_MAX_LEVEL_INTERVAL = [10, 8 ,6, 4, 3, 2]
RELATION_MIN_GINI_INTERVAL = [0.4, 0.42, 0.44, 0.46, 0.48, 0.5]
RELATION_TREE_NUM_INTERVAL = [1, 3, 5, 7, 9, 11, 13]
RELATION_SAMPLE_NUM_INTERVAL = [10, 20, 30, 40, 50, 60]

FLAG_USEING_MAX_LEVEL_RELATION = False
FLAG_USING_MIN_GINI_RELATION = False
FLAG_USING_TREE_NUM_INTERVAL = False
FLAG_USING_SAMPLE_NUM_INTERVAL = False

"""
Non-adjestable Variable
dynamicly calculate total tree number
"""
if FLAG_USEING_MAX_LEVEL_RELATION: TOTAL_GENERATE_TREE_NUM *= len(RELATION_MAX_LEVEL_INTERVAL)
if FLAG_USING_MIN_GINI_RELATION: TOTAL_GENERATE_TREE_NUM *= len(RELATION_MIN_GINI_INTERVAL)
if FLAG_USING_TREE_NUM_INTERVAL: 
    TOTAL_GENERATE_TREE_NUM /= TREE_NUM
    for mul in RELATION_TREE_NUM_INTERVAL:
        TOTAL_GENERATE_TREE_NUM *= mul
if FLAG_USING_SAMPLE_NUM_INTERVAL: TOTAL_GENERATE_TREE_NUM *= len(RELATION_SAMPLE_NUM_INTERVAL)


"""
Function
Goal: Finding Relation 
"""
def RelationLevelAndAccuracy():
    global raw_data
    global MAX_LEVEL
    
    temp_MAX_LEVEL = MAX_LEVEL
    acc_list = list()
    avg_level_list = list()
    
    for max_level, i in zip(RELATION_MAX_LEVEL_INTERVAL, range(len(RELATION_MAX_LEVEL_INTERVAL))):
        MAX_LEVEL = max_level
        
        data = deepcopy(raw_data)
        preprocessing(data)
        acc_list += [MultipleValid(data)]
        GetRecordResult()
        avg_level_list += [Average_Tree_Level]
        RecordReset()
        print("Relation Level Process: "+str(i+1)+" / "+str(len(RELATION_MAX_LEVEL_INTERVAL)))
    
    MAX_LEVEL = temp_MAX_LEVEL
    
    y = acc_list
    x = list()
    for max_level, avg_level in zip(RELATION_MAX_LEVEL_INTERVAL, avg_level_list):
        x += ["max: "+str(max_level)+"\navg: "+str(round(avg_level, 2))]
    
    plt.plot(x, y)
    plt.title("Maximum node Level <-> Accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("Max Level")
    plt.show()

def RelationLGiniAndAccuracy():
    global raw_data
    global MIN_GINI
    
    temp_MIN_GINI = MIN_GINI
    acc_list = list()
    
    for min_gini, i in zip(RELATION_MIN_GINI_INTERVAL, range(len(RELATION_MIN_GINI_INTERVAL))):
        MIN_GINI = min_gini
        
        data = deepcopy(raw_data)
        preprocessing(data)
        acc_list += [MultipleValid(data)]
        print("Relation Gini Process: "+str(i+1)+" / "+str(len(RELATION_MIN_GINI_INTERVAL)))
    
    MIN_GINI = temp_MIN_GINI
    
    y = acc_list
    x = RELATION_MIN_GINI_INTERVAL
    
    plt.plot(x, y)
    plt.title("Minimum Gini's impurity <-> Accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("Min Gini")
    plt.show()
    
def RelationSampleNumAndAccuracy():
    global raw_data
    global MIN_SAMPLE
    
    temp_MIN_SAMPLE = MIN_SAMPLE
    acc_list = list()
    
    for Sample_Num, i in zip(RELATION_SAMPLE_NUM_INTERVAL, range(len(RELATION_SAMPLE_NUM_INTERVAL))):
        MIN_SAMPLE = Sample_Num
        
        data = deepcopy(raw_data)
        preprocessing(data)
        acc_list += [MultipleValid(data)]
        print("Relation Sample Number Process: "+str(i+1)+" / "+str(len(RELATION_SAMPLE_NUM_INTERVAL)))
    
    MIN_SAMPLE = temp_MIN_SAMPLE
    
    y = acc_list
    x = RELATION_SAMPLE_NUM_INTERVAL
    
    plt.plot(x, y)
    plt.title("Minimum Sample Number <-> Accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("Sample Number")
    plt.show()
    
def RelationTreeNumAndAccuracy():
    global raw_data
    global TREE_NUM
    
    temp_Tree_Num = TREE_NUM
    acc_list = list()
    
    for Tree_Num, i in zip(RELATION_TREE_NUM_INTERVAL, range(len(RELATION_TREE_NUM_INTERVAL))):
        TREE_NUM = Tree_Num
        
        data = deepcopy(raw_data)
        preprocessing(data)
        acc_list += [MultipleValid(data)]
        print("Relation Tree Number Process: "+str(i+1)+" / "+str(len(RELATION_TREE_NUM_INTERVAL)))
    
    TREE_NUM = temp_Tree_Num
    
    y = acc_list
    x = RELATION_TREE_NUM_INTERVAL
    
    plt.plot(x, y)
    plt.title("Tree Number <-> Accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("Tree Number")
    plt.show()

"""
Record function
"""        
def RecordUpdate(level, nodeNum, tree):
    global Tree_Num 
    
    global Shallowest_Tree_Level
    global Deepest_Tree_Level
    global Average_Tree_Level
    
    global Largest_Tree_Node_Num
    global Smallest_Tree_Node_Num
    global Average_Tree_Node_Num
    
    global tree_Inspect
    
    Tree_Num += 1
    
    Shallowest_Tree_Level = min(level, Shallowest_Tree_Level)
    Deepest_Tree_Level = max(level ,Deepest_Tree_Level)
    Average_Tree_Level += level
    
    Smallest_Tree_Node_Num = min(nodeNum, Smallest_Tree_Node_Num)
    Largest_Tree_Node_Num = max(nodeNum, Largest_Tree_Node_Num)
    Average_Tree_Node_Num += nodeNum
    
    if randint(1, TOTAL_GENERATE_TREE_NUM) <= 3: tree_Inspect += [tree]

def GetRecordResult():
    global Tree_Num 
    
    global Shallowest_Tree_Level
    global Deepest_Tree_Level
    global Average_Tree_Level
    
    global Largest_Tree_Node_Num
    global Smallest_Tree_Node_Num
    global Average_Tree_Node_Num
    
    Average_Tree_Level /= Tree_Num
    Average_Tree_Node_Num /= Tree_Num
    
    print("Shallowest_Tree_Level", Shallowest_Tree_Level)
    print("Deepest_Tree_Level", Deepest_Tree_Level)
    print("Average_Tree_Level", Average_Tree_Level)
    print()
    
    print("Smallest_Tree_Node_Num", Smallest_Tree_Node_Num)
    print("Largest_Tree_Node_Num", Largest_Tree_Node_Num)
    print("Average_Tree_Node_Num", Average_Tree_Node_Num)

def RecordReset():
    global Tree_Num 
    
    global Shallowest_Tree_Level
    global Deepest_Tree_Level
    global Average_Tree_Level
    
    global Largest_Tree_Node_Num
    global Smallest_Tree_Node_Num
    global Average_Tree_Node_Num
    
    global Tree_Inspect
    
    Tree_Num = 0

    Shallowest_Tree_Level = 1000
    Deepest_Tree_Level = 0
    Average_Tree_Level = 0
    
    Largest_Tree_Node_Num = 0
    Smallest_Tree_Node_Num = 1000
    Average_Tree_Node_Num = 0
    
    Tree_Inspect = list()
    
"""
Class
"""
class Tree:
    def __init__(self):
        self._root = 0
        self._node_stack = list()
        self._last_stack_index = -1
        self._node_num = 0
        self._deepest_level = None
        
        self._min_sample = MIN_SAMPLE
        self._min_gini = MIN_GINI
        self._max_level = MAX_LEVEL
        
        self._use_attributes = None
        self._use_att_index = None
        
        self.LEAF_NODE = -1
        self.UNDEFINE = -2
        
    def BuildTree(self, data, use_att_index):
        self._use_att_index = use_att_index
        self._use_attributes = [ATTRIBUTES[i] for i in use_att_index]
        
        root = self.GetNode()
        root["id"] = self._root
        root["parent"] = self.UNDEFINE
        root["sample_num"] = len(data)
        root["data"] = data
        root["level"] = 1
        
        class_num = {"Iris-setosa":0, "Iris-versicolor":0, "Iris-virginica":0}
        for d in data:
            if d[-1] == "Iris-setosa": class_num["Iris-setosa"] += 1
            if d[-1] == "Iris-versicolor": class_num["Iris-versicolor"] += 1
            if d[-1] == "Iris-virginica": class_num["Iris-virginica"] += 1
        
        assert len(data) > 0, "data length <= 0"
        root["gini"] = 1 - pow(class_num["Iris-setosa"]/len(data), 2)\
                          - pow(class_num["Iris-versicolor"]/len(data), 2)\
                          - pow(class_num["Iris-virginica"]/len(data), 2)
                          
        self.AddToStack(root)
        
        self._deepest_level = self.SplitNode(self._root)
        
        RecordUpdate(self._deepest_level, self._node_num, self)
    
    def GetNode(self):
        node = { "id": None,
                 "split_attribute": None,
                 "thrashold": None,
                 "parent": None,
                 "left_child": None,
                 "right_child": None,
                 "gini": None,
                 "class": None,
                 "data": None,
                 "sample_num": None,
                 "level": None
                 }
        return node
    
    def AddToStack(self, node):
        self._node_stack.append(node)
        self._last_stack_index += 1
        node["id"] = self._last_stack_index
        self._node_num += 1
        return self._last_stack_index
    
    def GetLeftChild(self, node):
        index = node["left_child"]
        return self._node_stack[index]
    
    def GetRightChild(self, node):
        index = node["right_child"]
        return self._node_stack[index]
    
    def GetParent(self, node):
        index = node["parent"]
        return self._node_stack[index]
    
    def IsLeaf(self, node):
        return node["left_child"] == self.LEAF_NODE
    
    def SplitNode(self, nodeId):
        node = self._node_stack[nodeId]
        if node["sample_num"] <= self._min_sample\
                or node["gini"] <= self._min_gini\
                or node["level"] >= self._max_level:
            self.CalculateClass(node)
            node["left_child"] = self.LEAF_NODE
            node["right_child"] = self.LEAF_NODE
            return node["level"]
        else:
            data = node["data"]
            att_len = ATT_BAGGING_NUM
            min_gini = 100
            best_att = None
            best_thrashold = None
            best_split_index = 0
            best_part1_gini = 100
            best_part2_gini = 100
            
            for att_index in range(att_len):
                data.sort(key=lambda x: x[att_index])
                temp = [line[att_index] for line in data]
                thrashold = list(set(temp))
                for thr in thrashold:
                    gini, split_index, part1_gini, part2_gini = self.CalculateGini(data, att_index, thr)
                    
                    if gini < min_gini:
                        min_gini = gini
                        best_att = att_index
                        best_thrashold = thr
                        best_split_index = split_index
                        best_part1_gini = part1_gini
                        best_part2_gini = part2_gini
            
            leftNode = self.GetNode()
            leftNode["parent"] = node["id"] 
            leftNode["gini"] = best_part1_gini
            leftNode["data"] = data[:best_split_index]
            leftNode["sample_num"] = best_split_index
            leftNode["level"] = node["level"] + 1
            
            rightNode = self.GetNode()
            rightNode["parent"] = node["id"] 
            rightNode["gini"] = best_part2_gini
            rightNode["data"] = data[best_split_index:]
            rightNode["sample_num"] = len(data) - best_split_index
            rightNode["level"] = node["level"] + 1
            
            node["split_attribute"] = self._use_attributes[best_att]
            node["thrashold"] = best_thrashold
            
            self.AddToStack(leftNode)
            self.AddToStack(rightNode)
            node["left_child"] = leftNode["id"]
            node["right_child"] = rightNode["id"]
            
            return max(self.SplitNode(leftNode["id"]), self.SplitNode(rightNode["id"]))      
            
    def CalculateClass(self, node):
        data = node["data"]
        setosa_num = 0
        versicolor_num = 0
        virginica_num = 0
        for d in data:
            if d[-1] == "Iris-setosa": setosa_num += 1
            if d[-1] == "Iris-versicolor": versicolor_num += 1
            if d[-1] == "Iris-virginica": virginica_num += 1
        
        Max = max(setosa_num, versicolor_num, virginica_num)
        if Max == setosa_num: node["class"] = "Iris-setosa"
        elif Max == versicolor_num: node["class"] = "Iris-versicolor"
        else: node["class"] = "Iris-virginica"
    
    def CalculateGini(self, data, att_index, thrashold):
        all_num = len(data)
        part1_num = 0
        part2_num = 0
        split_index = 0
        
        part1_class = [0,0,0]
        part2_class = [0,0,0]
        
        for i in range(all_num):
            if data[i][att_index] > thrashold: 
                split_index = i
                break
            
        part1_num = split_index
        part2_num = all_num - part1_num
        
        for i in range(split_index):
            if data[i][-1] == "Iris-setosa": part1_class[0] += 1
            elif data[i][-1] == "Iris-versicolor": part1_class[1] += 1
            else: part1_class[2] += 1
        
        for i in range(split_index, all_num):
            if data[i][-1] == "Iris-setosa": part2_class[0] += 1
            elif data[i][-1] == "Iris-versicolor": part2_class[1] += 1
            else: part2_class[2] += 1
        
        if part1_num <= 0: 
            return 100, -1, -1, -1
        assert part1_num > 0, "part1_num <= 0"
        assert part2_num > 0, "part2_num <= 0"
        
        part1_gini = 1 - pow(part1_class[0]/part1_num, 2)\
                        - pow(part1_class[1]/part1_num, 2)\
                        - pow(part1_class[2]/part1_num, 2)
        
        part2_gini = 1 - pow(part2_class[0]/part2_num, 2)\
                        - pow(part2_class[1]/part2_num, 2)\
                        - pow(part2_class[2]/part2_num, 2)
        
        all_gini = (part1_num/all_num) * part1_gini + (part2_num/all_num) * part2_gini
        
        return all_gini, split_index, part1_gini, part2_gini
    
    def Predict(self, query):
        popIndex = list(set(range(4)) - set(self._use_att_index))
        
        [query.pop(idx) for idx in popIndex]
        
        return self.PredictInner(self._root, query)
        
    def PredictInner(self, nodeId, query):
        node = self._node_stack[nodeId]
        
        if self.IsLeaf(node):
            return node["class"]
        else:
            att_index = self._use_attributes.index(node["split_attribute"])
            
            if query[att_index] <= node["thrashold"]:
                return self.PredictInner(node["left_child"], query)
            else:
                return self.PredictInner(node["right_child"], query)

class Forest:
    def __init__(self):
        self._tree_num = TREE_NUM
        self._tree_stack = list()
        
    def BuildForest(self, train_data):
        for i in range(self._tree_num):
            data, use_att = Bagging(deepcopy(train_data))
            tree = Tree()
            tree.BuildTree(data, use_att)
            
            
            self._tree_stack.append(tree)
            
    def Predict(self, query):
        vote = {"Iris-setosa":0, "Iris-versicolor":0, "Iris-virginica":0}
        assert self._tree_stack != None, "self._tree_stack == None"
        for tree in self._tree_stack:
            one_vote = tree.Predict(deepcopy(query))
            vote[one_vote] += 1
        
        Max = max(vote["Iris-setosa"], vote["Iris-versicolor"], vote["Iris-virginica"])
        if Max == vote["Iris-setosa"]: result = "Iris-setosa"
        elif Max == vote["Iris-versicolor"]: result = "Iris-versicolor"
        else: result = "Iris-virginica"
        return result
    
    
"""
Function
"""
def preprocessing(data):
    for line in data:
        for i in range(4):
            line[i] = float(line[i])
    shuffle(data)
    
def TestTrainSplit(data):
    shuffle(data)
    train = data[:int(TRAIN_NUM)]
    test = data[int(TRAIN_NUM):]
    return test, train

def AttributeBagging(data):
    random_index = sample(range(0,4), ATT_REMOVE_NUM)
    [[line.pop(index) for line in data]for index in random_index]
    
    remain_index = list(set(range(4)) - set(random_index))
    return remain_index
    
def SampleBagging(data):
    return data[:int(SAM_NUM)]

def Bagging(data):
    shuffle(data)
    data = SampleBagging(data)
    remain_index = AttributeBagging(data)
    return data, remain_index

def Valid(forest, test):
    assert forest._tree_stack != list(), "Empty Forest"
    test_modify = deepcopy(test)
    ground_truth = [line.pop(-1) for line in test_modify]
    right_num = 0
    for query, truth in zip(test_modify, ground_truth):
        est = forest.Predict(query)
        
        if est == truth: right_num += 1
    accuracy = right_num/len(test)
    return accuracy

def ImportFile(filePath = FILE_PATH):
    global raw_data
    with open(filePath, newline='') as csvfile:
        _raw_data = list(csv.reader(csvfile))
        
        raw_data = _raw_data
        data = deepcopy(_raw_data)
    return data

def CrossValid(data):
    data_set = [ data[TEST_DATA_NUM*i : TEST_DATA_NUM *(i+1)] for i in range(CROSS_VALID_SPLIT_NUM-1)]
    data_set.append(data[TEST_DATA_NUM*(CROSS_VALID_SPLIT_NUM-1):]) 
    acc_sum = 0
    for i in range(CROSS_VALID_SPLIT_NUM):
        data_set_copy = deepcopy(data_set)
        
        test = data_set_copy[i]
        train = list()
        for j in range(CROSS_VALID_SPLIT_NUM):
            if j!=i: train += data_set_copy[j]
            
        forest = Forest()
        forest.BuildForest(train)
        acc_sum += Valid(forest, test)
    
    accuracy = acc_sum/CROSS_VALID_SPLIT_NUM
    return accuracy
    
def MultipleValid(data):
    acc_sum = 0
    for i in range(MULTI_VALID_TIME):
        data_copy = deepcopy(data)
        shuffle(data_copy)
        acc_sum += CrossValid(data_copy)
        print("Valid Process: "+str(i+1)+" / "+str(MULTI_VALID_TIME))
    return acc_sum/MULTI_VALID_TIME
    
    

"""
Main
"""
raw_data = None

data = ImportFile()
preprocessing(data)

accuracy = MultipleValid(data)
print()
print("Train Data Number:", DATA_NUM - TEST_DATA_NUM)
print("Test Data Number:", TEST_DATA_NUM)
print()
print("Accuracy:",accuracy)
print()

GetRecordResult()
RecordReset()

#RelationLevelAndAccuracy()
#RelationLGiniAndAccuracy()
#RelationTreeNumAndAccuracy()
#RelationSampleNumAndAccuracy()