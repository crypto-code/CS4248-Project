import os
import re
from nltk import word_tokenize
from nltk import pos_tag
from collections import Counter
import matplotlib.pyplot as plt
import pandas as pd
class node():
    def __init__(self,keyword):
        self.keyword = [keyword]
        self.child = []
        self.parent = []
    def insertChild(self,child):
        self.child.append(child)
    def insertParent(self,parent):
        self.parent.append(parent)
    def getChild(self):
        return self.child
    def getParent(self):
        return self.parent
    def getKeyword(self):
        return self.keyword
    def merge(self,toBeMerge):
        self.child += toBeMerge.getChild()
        self.parent += toBeMerge.getParent()
        self.keyword += toBeMerge.getKeyword()
directory = 'train2'
category = {"Material": [], "Task": [], "Process": []}
keywordType = {}
freq = Counter()
mode = 1
total = 0
processDoc = 0
taskDoc = 0
materialDoc = 0
moreThan5Total = 0
moreThan3Total = 0
singletonTotal = 0
relationship = {}
i = 0
surround = Counter()
surroundMat = Counter()
surroundPro = Counter()
surroundTas = Counter()
surroundDict = {"Process":surroundPro,"Material":surroundMat,"Task":surroundTas}
totalWord = 0
totalKeyword = 0
for filename in os.scandir(directory):
    if mode == 1:
        i+=1
        f = open(filename,'r', encoding='utf-8')
        contentList = f.read().split("\n")
        f.close()
        term = {}
        toBeAnalysed = []
        processAppeared = False
        taskAppeared = False
        materialAppeared = False
        for content in contentList:
            analyse = content.split('\t')
            if re.match("T\d*", analyse[0]):
                cat = analyse[1].split()[0]
                if not processAppeared and cat == "Process":
                    processDoc += 1
                    processAppeared = True
                if not taskAppeared and cat == "Task":
                    taskDoc += 1
                    taskAppeared = True
                if not materialAppeared and cat == "Material":
                    materialDoc += 1
                    materialAppeared = True
                keyword = analyse[2]
                totalKeyword += len(keyword.split(" "))
                term[analyse[0]] = keyword
                category[cat].append(keyword)
                keywordType[keyword] = cat
                if keyword in freq.keys():
                    freq[keyword] += 1
                else:
                    freq[keyword] = 1
                keywordSplit = keyword.split()
                if len(keywordSplit) == 1:
                    singletonTotal += 1
                if len(keywordSplit) >= 3:
                    moreThan3Total += 1
                if len(keywordSplit) >= 5:
                    moreThan5Total += 1
                total += 1
            else:
                if len(analyse) > 1:
                    toBeAnalysed.append(analyse)
                    temp = analyse[1].split()
                    typeOfRel = temp[0]
                    if typeOfRel == "Hyponym-of":
                        term1 = term[temp[1].split(":")[1]]
                        term2 = term[temp[2].split(":")[1]]
                    else:
                        term1 = term[temp[1]]
                        term2 = term[temp[2]]
                    #assign/create nodes
                    if term1 not in relationship.keys():
                        term1Node = node(term1)
                        relationship[term1] = term1Node
                    else:
                        term1Node = relationship[term1]
                    if term2 not in relationship.keys():
                        term2Node = node(term2)
                        relationship[term2] = term2Node
                    else:
                        term2Node = relationship[term2]
                    #check relationship
                    if typeOfRel == "Hyponym-of":
                        term1Node.insertParent(term2Node)
                        term2Node.insertChild(term1Node)
                    elif typeOfRel == "Synonym-of":
                        term1Node.merge(term2Node)
                        relationship[term2] = term1Node
                    #print("{} {} {}".format(typeOfRel,term[no1], term[no2]))                
        mode = 2
        f.close()
        continue
    if mode == 2:
        f = open(filename,'r', encoding='utf-8')
        text = f.read().split(" ")
        totalWord += len(text)
        mode = 3
        continue
    if mode == 3:
        mode = 1
        continue


def relate(a,b):
    if (a not in relationship.keys() or b not in relationship.keys()):
        return "'{}' and '{}' are not related".format(a,b)
    if (b in relationship[a].getKeyword()):
        return "'{}' and '{}' are synonyms".format(a,b)
    for child in relationship[a].getChild():
        if b in child.getKeyword():
            return "'{}' is a hyponym of '{}'".format(b,a)
    for parent in relationship[a].getParent():
        if b in parent.getKeyword():
            return "'{}' is a hyponym of '{}'".format(a,b)
    return "'{}' and '{}' are not related".format(a,b)
procPos = Counter()
Pos = Counter()
p = 0
o = 0
for word in category["Process"]:
    wordToken = word_tokenize(word)
    wordPos = pos_tag(wordToken)
    o += 1
    p += len(word.split(" "))
    tags = []
    for wor, tag in wordPos:
        tags.append(tag)
    tags = ' '.join(tags)
    if tags in procPos.keys():
        procPos[tags] += 1
    else:
        procPos[tags] = 1
    if tags in Pos.keys():
        Pos[tags] += 1
    else:
        Pos[tags] = 1

matPos = Counter()
for word in category["Material"]:
    wordToken = word_tokenize(word)
    wordPos = pos_tag(wordToken)
    tags = []
    o += 1
    p += len(word.split(" "))
    for wor, tag in wordPos:
        tags.append(tag)
    tags = ' '.join(tags)
    if tags in matPos.keys():
        matPos[tags] += 1
    else:
        matPos[tags] = 1
    if tags in Pos.keys():
        Pos[tags] += 1
    else:
        Pos[tags] = 1
        
taskPos = Counter()
for word in category["Task"]:
    wordToken = word_tokenize(word)
    wordPos = pos_tag(wordToken)
    tags = []
    o += 1
    p += len(word.split(" "))
    for wor, tag in wordPos:
        tags.append(tag)
    tags = ' '.join(tags)
    if tags in taskPos.keys():
        taskPos[tags] += 1
    else:
        taskPos[tags] = 1
    if tags in Pos.keys():
        Pos[tags] += 1
    else:
        Pos[tags] = 1
totalproc = 0
totaltask = 0
totalmat = 0
for key in freq.keys():
    if key in category["Process"]:
        totalproc += freq[key]
    if key in category["Task"]:
        totaltask += freq[key]
    if key in category["Material"]:
        totalmat += freq[key]
print(totalproc)
print(totaltask)
print(totalmat)
##print(procPos.most_common(5))
##print(matPos.most_common(5))
##print(taskPos.most_common(5))
##print(Pos.most_common(5))
##print("Top 10 most common keyphrases: {}".format([word for (word,number) in freq.most_common(10)]))
##classi = ["Total keyword appearances","Total unique keywords","Number of documents with PROCESS", "Number of documents with TASK","Number of documents with MATERIAL","Number of PROCESS keywords", "Number of TASK keywords", "Number of MATERIAL keywords", "Average PROCESS keywords in documents with PROCESS", "Average TASK keywords in documents with TASK",
##          "Average MATERIAL keywords in documents with MATERIAL","% keyword mentions, word length = 1", "% keyword mentions, word length >= 3", "% keyword mentions, word length >= 5"]
##value = [total, len(keywordType),processDoc,taskDoc,materialDoc,len(category['Process']),len(category['Task']),len(category['Material']),len(category['Process'])/processDoc,len(category['Task'])/taskDoc,
##            len(category['Material'])/materialDoc,singletonTotal/total,moreThan3Total/total,moreThan5Total/total]
##maxv = len("Average MATERIAL keywords in documents with MATERIAL")
##print(i)
##print("-"*(maxv+11))
##print("|Characteristics" + " "*(maxv+10-len("|Characteristics")) + "|")
##print("-"*(maxv+11))
##for i in range(len(classi)):
##    print("|" + classi[i] + " "*(maxv-len(classi[i])+1) + "|" + " "*(7 - len(str(round(value[i],5)))) + str(round(value[i],5)) + "|")
##print("-"*(maxv+11))
##print(totalKeyword/totalWord)
##print(p/o)
##print("Total keyword appearances: {}".format(total))
##print("Total unique keywords: {}".format(len(keywordType)))
##print("Average PROCESS keywords in documents with PROCESS: {}".format(len(category['Process'])/processDoc))
##print("|Average TASK keywords in documents with TASK|  {}".format(len(category['Task'])/taskDoc))
##print("|Average MATERIAL keywords in documents with MATERIAL| {}|".format(len(category['Material'])/materialDoc))
##print("% keyword mentions, word length = 1: {}".format(singletonTotal/total))
##print("% keyword mentions, word length >= 3: {}".format(moreThan3Total/total))
##print("% keyword mentions, word length >= 5: {}".format(moreThan5Total/total))
##print(relationship.keys())
##print(relate("10 strings of 24 modules per string","high efficiency system"))
##print(relate("system","high efficiency system"))
##print(relate("Least Mean Squares", "high efficiency system"))
##print(relate("line galloping","galloping"))
##print(relate("markedly strain sensitive materials","fibrin gels"))
