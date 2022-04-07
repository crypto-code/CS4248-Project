import pandas as pd
import numpy as np
from collections import Counter

with open('train_augmented_sentences.npy', 'rb') as f:
    sent = np.load(f, allow_pickle=True)
with open('train_augmented_annotations.npy', 'rb') as f:
    anno = np.load(f, allow_pickle=True)
print(sent[0])
print(anno[0])
process = Counter()
task = Counter()
material = Counter()
for i in range(len(sent)):
    prev = 0
    temp = ''
    for j in range(len(sent[i])):
        if prev == anno[i][j]:
            if j != len(sent[i]):
                temp += sent[i][j] + ' '
            else:
                if (anno[i][j] == 'Process'):
                    if temp[:-1] not in process:
                        process[temp[:-1]] = 1
                    else:
                        process[temp[:-1]] += 1
                if (anno[i][j] == 'Task'):
                    if temp[:-1] not in task:
                        task[temp[:-1]] = 1
                    else:
                        task[temp[:-1]] += 1
                if (anno[i][j] == 'Material'):
                    if temp[:-1] not in material:
                        material[temp[:-1]] = 1
                    else:
                        material[temp[:-1]] += 1
        else:
            if (anno[i][j-1] == 'Process'):
                if temp[:-1] not in process:
                    process[temp[:-1]] = 1
                else:
                    process[temp[:-1]] += 1
            if (anno[i][j-1] == 'Task'):
                if temp[:-1] not in task:
                    task[temp[:-1]] = 1
                else:
                    task[temp[:-1]] += 1
            if (anno[i][j-1] == 'Material'):
                if temp[:-1] not in material:
                    material[temp[:-1]] = 1
                else:
                    material[temp[:-1]] += 1
            temp = ''
        prev = anno[i][j]
task.pop('')
task.pop('.')
material.pop('')
material.pop('.')
process.pop('')
process.pop('.')
totalproc = 0
totaltask = 0
totalmat = 0
for key in process.keys():
    totalproc += process[key]
for key in material.keys():
    totalmat += material[key]
for key in task.keys():
    totaltask += task[key]
print(len(process))
print(len(task))
print(len(material))
print(totalproc)
print(totaltask)
print(totalmat)
print(len(anno))
