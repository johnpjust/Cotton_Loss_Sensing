import pandas as pd
import numpy as np
import cv2
from skimage.transform import (hough_line, hough_line_peaks,
                               probabilistic_hough_line)
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from matplotlib import cm

from sklearn.cluster import OPTICS, cluster_optics_dbscan

def get_R_theta(x1, y1, x2, y2):
    xd = x2-x1
    yd = y2-y1
    theta = np.arctan(xd/yd)
    n0 = np.sin(theta)
    n1 = np.cos(theta)
    M = np.array([[1,0,n0],[0,1,n1],[n0,n1,0]])
    b = np.array([[x1],[y1],[0]])
    sol = np.linalg.solve(M,b)
    return np.arctan(sol[1]/sol[0]), np.sqrt(sol[0]**2 + sol[1]**2)


img = cv2.imread(r'C:\Users\just\Desktop\Cotton\5_1464_1539113795.2610_561.png')
scale_percent = 20 # percent of original size
width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * scale_percent / 100)
dim = (width, height)
imgr = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
## HSV green stuff
hsv = cv2.cvtColor(imgr, cv2.COLOR_RGB2HSV)
lower_green = np.array([30, 25, 0])
upper_green = np.array([85, 255, 200])

mask_green = cv2.inRange(hsv, lower_green, upper_green)
res_green = cv2.bitwise_and(imgr, imgr, mask=mask_green)

#DB scan to eliminate noise

cv2.imshow('frame', imgr)
cv2.imshow('mask', mask_green)
cv2.imshow('res', res_green)


## HSV white stuff
lower_blue = np.array([0, 0, int(1.5*np.median(np.reshape(hsv,(-1,3)), 0)[2])])
upper_blue = np.array([20, 255, 255])
mask_blue1 = cv2.inRange(hsv, lower_blue, upper_blue)

lower_blue = np.array([150, 0, int(1.5*np.median(np.reshape(hsv,(-1,3)), 0)[2])])
upper_blue = np.array([180, 255, 255])
mask_blue2 = cv2.inRange(hsv, lower_blue, upper_blue)
mask_blue = mask_blue1 | mask_blue2
res_blue = cv2.bitwise_and(imgr, imgr, mask=abs(mask_blue-1))

cv2.imshow('frame', imgr)
cv2.imshow('mask', mask_blue)
cv2.imshow('res', res_blue)
# ################ kmeans for image #############
# Z = img.reshape((-1,3))
#
# # convert to np.float32
# Z = np.float32(Z)
#
# # define criteria, number of clusters(K) and apply kmeans()
# criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
# K = 8
# ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
#
# # Now convert back into uint8, and make original image
# center = np.uint8(center)
# res = center[label.flatten()]
# res2 = res.reshape((imgr.shape))
#
# cv2.imshow('res2',res2)


# Line finding using the Probabilistic Hough Transform
theta = 7*np.pi / 8 + np.arange(45) / 180 * np.pi
lines = probabilistic_hough_line(mask_green, threshold=15, line_length=40,
                                 line_gap=20, theta=theta)

# Generating figure 2
fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)
ax = axes.ravel()

ax[0].imshow(imgr, cmap=cm.gray)
ax[0].set_title('Input image')

ax[1].imshow(mask_green * 0)
mags = []
thetas = []
for line in lines:
    b, m = get_R_theta(*line[0], *line[1])
    mags.append(m)
    thetas.append(b)

    p0, p1 = line
    ax[1].plot((p0[0], p1[0]), (p0[1], p1[1]))
ax[1].set_xlim((0, imgr.shape[1]))
ax[1].set_ylim((imgr.shape[0], 0))
ax[1].set_title('Probabilistic Hough')

ax[2].scatter(mags, thetas)

for a in ax:
    a.set_axis_off()

plt.tight_layout()
plt.show()



X = np.transpose(np.squeeze(np.array([(mags - np.mean(mags))/np.std(mags),(thetas - np.mean(thetas))/np.std(thetas)])))
## dbscan
db = DBSCAN(eps=0.3).fit(X)

core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

# Black removed and is used for noise instead.
unique_labels = set(labels)
colors = [plt.cm.Spectral(each)
          for each in np.linspace(0, 1, len(unique_labels))]
plt.figure()
for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = [0, 0, 0, 1]

    class_member_mask = (labels == k)

    xy = X[class_member_mask & core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=14)

    xy = X[class_member_mask & ~core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=6)

plt.title('Estimated number of clusters: %d' % n_clusters_)
plt.show()


# ## OPTICS cluster to find rows
# clust = OPTICS(min_samples=5, xi=.05, min_cluster_size=.05, n_jobs=-1)
# clust.fit(X)
#
#
# fig = plt.figure()
# ax2 = fig.add_subplot(1, 1, 1)
# # OPTICS
# colors = ['g.', 'r.', 'b.', 'y.', 'c.', 'm', '#eeefff']
# for klass, color in zip(range(len(np.unique(clust.labels_)) - 1), colors):
#     Xk = X[clust.labels_ == klass]
#     ax2.plot(Xk[:, 0], Xk[:, 1], color, alpha=0.3)
# ax2.plot(X[clust.labels_ == -1, 0], X[clust.labels_ == -1, 1], 'k+', alpha=0.1)
# ax2.set_title('Automatic Clustering\nOPTICS')

## get group means
df = pd.DataFrame({'mags': mags, 'thetas': thetas, 'labels': labels})
lineavgs = df.groupby('labels').mean()[1:]