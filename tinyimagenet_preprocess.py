import sys
import os
import fileinput
import shutil

path = sys.argv[1]

# Download the zip file from: wget http://cs231n.stanford.edu/tiny-imagenet-200.zip
res = {}
for line in fileinput.input(f"{path}/val_annotations.txt"):
    filename, label = line.split("\t")[:2]
    if label not in res:
        res[label] = []
    res[label].append(filename)

print(res)

for label in res:
    if not os.path.exists(f"{path}/{label}/images"):
        os.makedirs(f"{path}/{label}/images")
    for filename in res[label]:
        shutil.copy(f"{path}/images/{filename}", f"{path}/{label}/images/{filename}")

