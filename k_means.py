import cv2
import numpy as np
from metrics import jaccard, accuracy

image_folder = 'data/images/'
mask_folder = 'data/masks/'

jaccards = []
accuracies = []

for i in range(1,61):
    file = str(i).zfill(3)
    print(image_folder + file+ "_image.png")
    img = cv2.imread(image_folder + file+ "_image.png")
    image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mask = cv2.imread(mask_folder + file+"_mask.png")

    pixel_values = image.reshape((-1, 3))
    pixel_values = np.float32(pixel_values)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)

    looking_for_green = True
    k=4
    while looking_for_green:
        print(k)
        compression, labels, (centers) = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
        differences = centers[:, 1] - centers[:, 0]
        k = k+1
        if np.max(differences>70):
            looking_for_green = False

    cluster = np.argmax(differences)
    centers = np.uint8(centers)
    labels = labels.flatten()
    segmented_image = centers[labels.flatten()]


    new_img = (labels != cluster)*255
    new_img = new_img.reshape(image.shape[0], image.shape[1])
    new_img = np.array(new_img.astype(np.uint8))

    acc = accuracy(new_img, mask[:,:,0])
    accuracies.append(acc)
    print(acc)
    jacc = jaccard(new_img/255, mask[:,:,0]/255)
    print(jacc)
    jaccards.append(jacc)
    new_img = new_img.astype(np.uint8)

print("Average Accuracy: ", np.average(accuracies))
print("Average Jaccard Index: ", np.average(jaccards))