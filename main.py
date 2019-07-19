import numpy as np
import cv2
from skimage.transform import (hough_line, hough_line_peaks,
                               probabilistic_hough_line)
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from matplotlib import cm

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




# ## HSV white stuff
# hsv = cv2.cvtColor(imgr, cv2.COLOR_RGB2HSV)
# lower_green = np.array([30, 25, 0])
# upper_green = np.array([85, 255, 200])


################ kmeans for image #############
Z = img.reshape((-1,3))

# convert to np.float32
Z = np.float32(Z)

# define criteria, number of clusters(K) and apply kmeans()
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
K = 8
ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

# Now convert back into uint8, and make original image
center = np.uint8(center)
res = center[label.flatten()]
res2 = res.reshape((imgr.shape))

cv2.imshow('res2',res2)

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
