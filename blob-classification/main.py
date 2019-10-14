import numpy as np
import math
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from matplotlib.pyplot import imread
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
# from sklearn.datasets import load_digits
# digits = load_digits()

labelledBlobs = np.load('../data/labelledBlobs.npy')

patchSize = 9

targets = labelledBlobs[:, 5]
data = np.zeros([targets.shape[0], int(math.pow(patchSize, 2) * 3)])

for i in range(0, targets.shape[0]):
    x = int(labelledBlobs[i, 0])
    y = int(labelledBlobs[i, 1])
    u = int(labelledBlobs[i, 3])
    v = int(labelledBlobs[i, 4])
    patchRadius = math.floor(patchSize / 2)
    minx = x - patchRadius
    maxx = x + patchRadius + 1
    miny = y - patchRadius
    maxy = y + patchRadius + 1
    image = imread('/Users/elliott/projects/rti-panoramic-webapp/raphael-2018/mipmap-normals_jpeg/mipmap-00-u' + "{:02d}".format(u) + '-v' + "{:02d}".format(v) + '.jpg')
    # image = rgb2gray(image)
    patch = image[minx:maxx, miny:maxy, :]
    patch = patch.flatten()
    data[i] = patch

# figureCount = 12
# plt.figure(figsize=(20,4))
# for index, (image, label) in enumerate(zip(data[0:figureCount], targets[0:figureCount])):
#     plt.subplot(4, figureCount/4, index + 1)
#     print(np.reshape(image, (patchSize,patchSize, 3)))
#     plt.imshow(np.reshape(image, (patchSize,patchSize, 3)) / 256)
#     plt.title('T: %i\n' % label, fontsize = 10)
# plt.show()

x_train, x_test, y_train, y_test = train_test_split(data, targets, test_size=0.5, random_state=0)

# print(y_train)

logisticRegr = LogisticRegression()
# logisticRegr.fit(x_train, y_train)
logisticRegr.fit(data, targets)

blobs = np.load('../data/blobs.npy')

# full_data = np.zeros([blobs.shape[0], int(math.pow(patchSize, 2) * 3)])
#
# image = None
# currU = -1
# currV = -1
# for i in range(0, blobs.shape[0]):
#     if i % 500 == 0:
#         print("{0:.2f}%".format(i / blobs.shape[0] * 100))
#         # print("{} of {}".format(i, blobs.shape[0]))
#     x = int(blobs[i, 0])
#     y = int(blobs[i, 1])
#     u = int(blobs[i, 3])
#     v = int(blobs[i, 4])
#     patchRadius = math.floor(patchSize / 2)
#     minx = x - patchRadius
#     maxx = x + patchRadius + 1
#     miny = y - patchRadius
#     maxy = y + patchRadius + 1
#     if(minx < 4 or maxx > 507 or miny < 4 or maxy > 507):
#         continue
#     if (currU != u and currV != v):
#         image = imread('/Users/elliott/projects/rti-panoramic-webapp/raphael-2018/mipmap-normals_jpeg/mipmap-00-u' + "{:02d}".format(u) + '-v' + "{:02d}".format(v) + '.jpg')
#         currU = u
#         currV = v
#     # image = rgb2gray(image)
#     patch = image[minx:maxx, miny:maxy, :]
#     # print(patch.shape)
#     patch = patch.flatten()
#     full_data[i] = patch
#
# np.save('../data/full_data.npy', full_data)

full_data = np.load("../data/full_data.npy")
print(full_data.shape)

#
# predictions = logisticRegr.predict(x_test)
predictions = logisticRegr.predict(full_data)
print(predictions)

predictedBlobs = np.array([])
for i in range(0, predictions.shape[0]):
    if predictions[i] == 1.0:
        predictedBlob = blobs[i]
        predictedBlobs = np.vstack([predictedBlobs, predictedBlob]) if predictedBlobs.size else predictedBlob
np.save('../data/predictedBlobs.npy', predictedBlobs)
print("predictedBlobs shape: {}".format(predictedBlobs.shape))
#
# for i in range(0, predictions.shape[0]):
#     if predictions[i] == 1.0:
#         image = data[i]
#         plt.imshow(np.reshape(image, (patchSize,patchSize, 3)) / 256)
#         plt.show()

# figureCount = 12
# plt.figure(figsize=(20,4))
# for index, (image, label) in enumerate(zip(data[0:figureCount], targets[0:figureCount])):
#     plt.subplot(4, figureCount/4, index + 1)
#     print(np.reshape(image, (patchSize,patchSize, 3)))
#     plt.imshow(np.reshape(image, (patchSize,patchSize, 3)) / 256)
#     plt.title('T: %i\n' % label, fontsize = 10)
# plt.show()

# score = logisticRegr.score(x_test, y_test)
score = logisticRegr.score(data, targets)
print(score)

np.save('../data/predictions.npy', predictions)
