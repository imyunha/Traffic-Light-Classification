import os
from shutil import copyfile
for image in os.listdir("./"):
    for i in range(10):
        target = str(i) + image
        copyfile(image, target)
