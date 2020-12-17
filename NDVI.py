import os
import cv2
import numpy as np
from metrics import jaccard, accuracy

image_folder = 'data/images/'
mask_folder = 'data/masks/'

if not os.path.exists('data/masks1/'):
    os.makedirs('data/masks1/')

accuracies = []
jaccards = []

for i in range(1, 61):
    file = str(i).zfill(3)
    print(image_folder + file + "_image.png")
    img = cv2.imread(image_folder + file + "_image.png")
    mask = cv2.imread(mask_folder + file + "_mask.png")

    NIR = np.float32(img[:, :, 1])
    red = np.float32(img[:, :, 0])

    up = NIR - red
    down = NIR + red

    true_mask = mask[:, :, 0]
    indexes = (NIR - red) / (NIR + red)

    pred_mask = (indexes < 330 / 1000) * 255
    acc = accuracy(pred_mask, true_mask)
    jacc = jaccard(pred_mask / 255, true_mask / 255)
    cv2.imwrite('data/masks1/' + file + '_mask.png', pred_mask)
    accuracies.append(acc)
    jaccards.append(jacc)
print("Average Accuracy: ", np.average(accuracies))
print("Average Jaccard Index: ", np.average(jaccards))