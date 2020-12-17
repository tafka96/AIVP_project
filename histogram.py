import cv2
import numpy as np
from metrics import jaccard, accuracy

image_folder = 'data/images/'
mask_folder = 'data/masks/'

jaccards = []
accuracies = []
for i in range(1, 61):
    file = str(i).zfill(3)
    print(image_folder + file+ "_image.png")
    img = cv2.imread(image_folder + file+ "_image.png")
    mask = cv2.imread(mask_folder + file+"_mask.png")
    color = ('b', 'g', 'r')
    histr = cv2.calcHist([img], [1], None, [256], [0, 256])

    minis = 100+np.argmin(histr[100:110])
    new_img = (img[:, :, 1] < minis) * 255
    new_img = new_img.astype(np.uint8)


    acc = accuracy(new_img, mask[:,:,0])
    accuracies.append(acc)
    jacc = jaccard(new_img / 255, mask[:, :, 0] / 255)
    jaccards.append(jacc)

print("Average Accuracy: ", np.average(accuracies))
print("Average Jaccard Index: ", np.average(jaccards))