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
#
# blobs = np.load('../data/blobs.npy')

# for u in range(0, 49):
#     for v in range(0, 38):
# for u in range(0, 49):
#     for v in range(0, 38):
u = 13
v = 11
print("detecting blobs in: u{}-v{}".format(u, v))
image = imread("/Users/elliott/projects/rti-panoramic-webapp/raphael-2018/mipmap-normals_jpeg/mipmap-00-u{}-v{}.jpg".format("{:02d}".format(u), "{:02d}".format(v)))
# image = data.hubble_deep_field()[0:500, 0:500]
image_gray = rgb2gray(image)

blobs_log = blob_log(image_gray, min_sigma=1, max_sigma=5, num_sigma=5, threshold=0.060)
blobs_log_curr_u_v = np.zeros([1,2])
blobs_log_curr_u_v[0][0] = u
blobs_log_curr_u_v[0][1] = v
blobs_log_curr_u_v = np.tile(blobs_log_curr_u_v, [blobs_log.shape[0], 1])

# Compute radii in the 3rd column.
blobs_log[:, 2] = blobs_log[:, 2] * sqrt(2)

blobs_log = np.concatenate((blobs_log, blobs_log_curr_u_v), axis=1)

# blobs = np.concatenate((blobs, blobs_log), axis=0)
blobs = np.vstack([blobs, blobs_log]) if blobs.size else blobs_log

##
blobs_list = [blobs_log]
colors = ['yellow']
titles = ['Laplacian of Gaussian']
sequence = zip(blobs_list, colors, titles)

fig, ax = plt.subplots(1, 1, figsize=(9, 3), sharex=True, sharey=True)
# ax = axes.ravel()
#
for idx, (blobs, color, title) in enumerate(sequence):
    # ax[idx].set_title(title)
    ax.imshow(image)
    for blob in blobs:
        y, x, r, U, V = blob
        c = plt.Circle((x, y), r, color=color, linewidth=2, fill=False)
        ax.add_patch(c)
    ax.set_axis_off()

plt.tight_layout()
plt.show()

# np.save('../data/blobs.npy', blobs)
