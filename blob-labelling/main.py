from math import sqrt
from skimage import data
from skimage.feature import blob_dog, blob_log, blob_doh
from skimage.color import rgb2gray
from scipy import misc
import numpy as np
import random

# import matplotlib as mpl
# mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.pyplot import imread
from matplotlib.widgets import Button

blobs = np.array([])

blobs = np.load('../data/blobs.npy')

print(blobs.shape)

class Labelling(object):
    finished = False
    labelledBlobs = np.array([])
    def __init__(self):
        try:
            self.labelledBlobs = np.load('../data/labelledBlobs.npy')
        except FileNotFoundError:
            print("no file present")

    def callbackDone(self, event):
        print("done")
        self.finished = True
        np.save('../data/labelledBlobs.npy', self.labelledBlobs)
        plt.close();

    def callbackYes(self, event, blob):
        labelledBlob = np.zeros([1,6])
        labelledBlob[0, 0:5] = blob
        labelledBlob[0, 5] = 1
        print(labelledBlob)
        self.labelledBlobs = np.vstack([self.labelledBlobs, labelledBlob]) if self.labelledBlobs.size else labelledBlob
        plt.close();

    def callbackNo(self, event, blob):
        labelledBlob = np.zeros([1,6])
        labelledBlob[0, 0:5] = blob
        labelledBlob[0, 5] = 0
        print(labelledBlob)
        self.labelledBlobs = np.vstack([self.labelledBlobs, labelledBlob]) if self.labelledBlobs.size else labelledBlob
        plt.close();

labeller = Labelling()

while labeller.finished == False:
    randInt = random.randint(0, blobs.shape[0])
    blob = blobs[randInt]
    x = int(blob[0])
    y = int(blob[1])
    r = int(blob[2] * 8)
    u = int(blob[3])
    v = int(blob[4])
    fig, ax = plt.subplots()
    image = imread("/Users/elliott/projects/rti-panoramic-webapp/raphael-2018/mipmap-normals_jpeg/mipmap-00-u{}-v{}.jpg".format("{:02d}".format(u), "{:02d}".format(v)))
    minx = x - r
    maxx = x + r
    miny = y - r
    maxy = y + r
    if minx < 0 or maxx > 511 or miny < 0 or maxy > 511:
        plt.close()
        continue
    ax.imshow(image[minx:maxx, miny:maxy, :])
    ax.axis('off')

    axdone = plt.axes([0.59, 0.05, 0.1, 0.075])
    axyes = plt.axes([0.7, 0.05, 0.1, 0.075])
    axno = plt.axes([0.81, 0.05, 0.1, 0.075])
    bdone = Button(axdone, 'Done')
    bdone.on_clicked(labeller.callbackDone)
    byes = Button(axyes, 'Yes')
    byes.on_clicked(lambda x: labeller.callbackYes(x, blob))
    bno = Button(axno, 'No')
    bno.on_clicked(lambda x: labeller.callbackNo(x, blob))

    plt.show()
