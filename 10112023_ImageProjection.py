import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread


img = imread('cameraman.tif')
pixelsize = [1.2, 1.2] # not assume 1mm per pixel

fig, ax = plt.subplots(1,2)
plt.axes(ax[0])
cls = ax[0].imshow(img, 'gray')

pntsx, pntsy = np.meshgrid(np.array([75, 125, 175])*pixelsize[0], np.array([100, 150])*pixelsize[1])
pnts = np.concatenate((pntsx.ravel()[np.newaxis,:], pntsy.ravel()[np.newaxis,:]), axis=0)

#clockwise 25 degree rotation about center of image
theta = -25 *np.pi/180
R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
center = (np.array(np.shape(img))-1)/2
t = - R @ center*pixelsize + center*pixelsize
T = np.eye(3)
T[0:2,0:2] = R
T[0:2,2] = t

iT = np.linalg.inv(T)

#apply forward transform to points
plt.plot(pnts[1,:]/pixelsize[1], pnts[0,:]/pixelsize[0], 'r.')
npnts_h = T @ np.concatenate((pnts, np.ones((1,6))), axis=0)
plt.plot(npnts_h[1,:]/pixelsize[1], npnts_h[0,:]/pixelsize[0], 'y*')

# use interpn for image interpolation
from scipy.interpolate import interpn
xs = np.linspace(0, pixelsize[0]*(np.shape(img)[0]-1), np.shape(img)[0])
ys = np.linspace(0, pixelsize[1]*(np.shape(img)[1]-1), np.shape(img)[1])
X,Y = np.meshgrid(xs, ys, indexing='ij')
V = np.concatenate((X.ravel()[np.newaxis,:], Y.ravel()[np.newaxis,:],
                    np.ones((1, np.size(img)))), axis=0)
# apply inverse transformation to project image
W = iT @ V
Z = W[0:2, :].T
imr = np.reshape(interpn((xs, ys), img, Z, 'linear', False, 0), np.shape(img))

plt.axes(ax[1])
cols = ax[1].imshow(imr, 'gray')
plt.plot(pnts[1,:]/pixelsize[1], pnts[0,:]/pixelsize[0], 'r.')
plt.plot(npnts_h[1,:]/pixelsize[1], npnts_h[0,:]/pixelsize[0], 'y*')

ngrid = np.array([512, 512])
center2 = (ngrid -1)/2
t = - R @ center*pixelsize + center2*pixelsize

T = np.eye(3)
T[0:2,0:2] = R
T[0:2,2] = t

iT = np.linalg.inv(T)

xt = np.linspace(0, pixelsize[0]*(ngrid[0]-1), ngrid[0])
yt = np.linspace(0, pixelsize[1]*(ngrid[1]-1), ngrid[1])
X,Y = np.meshgrid(xt, yt, indexing='ij')
V = np.concatenate((X.ravel()[np.newaxis,:], Y.ravel()[np.newaxis,:],
                    np.ones((1, ngrid[0]*ngrid[1]))), axis=0)
W = iT @ V
Z = W[0:2, :].T
imr = np.reshape(interpn((xs, ys), img, Z, 'linear', False, 0), ngrid)

fig, ax = plt.subplots(1,2)
plt.axes(ax[0])
ax[0].imshow(img, 'gray')
plt.plot(pnts[1,:]/pixelsize[1], pnts[0,:]/pixelsize[0], 'r.')
plt.plot(npnts_h[1,:]/pixelsize[1], npnts_h[0,:]/pixelsize[0], 'y*')

plt.axes(ax[1])
ax[1].imshow(imr, 'gray')
plt.plot(pnts[1,:]/pixelsize[1], pnts[0,:]/pixelsize[0], 'r.')
plt.plot(npnts_h[1,:]/pixelsize[1], npnts_h[0,:]/pixelsize[0], 'y*')



# change pixel resolution
ngrid = np.array([512, 512])
pixelsize2 = np.array([0.8, 0.8])
center2 = (ngrid -1)/2
t = - R @ center*pixelsize + center2*pixelsize2

T = np.eye(3)
T[0:2,0:2] = R
T[0:2,2] = t

iT = np.linalg.inv(T)

xt = np.linspace(0, pixelsize2[0]*(ngrid[0]-1), ngrid[0])
yt = np.linspace(0, pixelsize2[1]*(ngrid[1]-1), ngrid[1])
X,Y = np.meshgrid(xt, yt, indexing='ij')
V = np.concatenate((X.ravel()[np.newaxis,:], Y.ravel()[np.newaxis,:],
                    np.ones((1, ngrid[0]*ngrid[1]))), axis=0)
W = iT @ V
Z = W[0:2, :].T
imr = np.reshape(interpn((xs, ys), img, Z, 'linear', False, 0), ngrid)

fig, ax = plt.subplots(1,2)
plt.axes(ax[0])
ax[0].imshow(img, 'gray')
plt.plot(pnts[1,:]/pixelsize[1], pnts[0,:]/pixelsize[0], 'r.')
plt.plot(npnts_h[1,:]/pixelsize[1], npnts_h[0,:]/pixelsize[0], 'y*')

plt.axes(ax[1])
ax[1].imshow(imr, 'gray')
plt.plot(pnts[1,:]/pixelsize[1], pnts[0,:]/pixelsize[0], 'r.')
plt.plot(npnts_h[1,:]/pixelsize[1], npnts_h[0,:]/pixelsize[0], 'y*')


fig, ax = plt.subplots(1,2)
plt.axes(ax[0])
ax[0].imshow(img, 'gray')

pntsx, pntsy = np.meshgrid([75,125,175], np.array([100,150]))
pnts = np.concatenate((pntsx.ravel()[np.newaxis,:], pntsy.ravel()[np.newaxis,:]), axis=0)
plt.plot(pnts[1,:], pnts[0,:], 'r.')
pntsr = pnts + np.random.default_rng(0).normal(size=[2,6]) * 15
plt.plot(pntsr[1,:], pntsr[0,:], 'y*')


# Thin plate splines facilitate nonrigid transformation
from ThinPlateSpline import *
# try alpha=0, 1000; rbf='rlogsqrtr'(default) or 'r'
Dx, Dy = thinPlateSpline(pnts.T, pntsr.T, np.shape(img), alpha=1000, rbf='r')
fig2, ax2 = plt.subplots(1,2)
plt.axes(ax2[0])
cls = ax2[0].imshow(Dx, 'gray')
plt.colorbar(cls)
plt.title('Def_x')
plt.axes(ax2[1])
cls = ax2[1].imshow(Dy, 'gray')
plt.colorbar(cls)
plt.title('Def_y')

plt.figure(fig)
plt.axes(ax[1])
targetdim = np.shape(img)


xt = np.linspace(0, targetdim[0]-1, targetdim[0])
yt = np.linspace(0, targetdim[1]-1, targetdim[1])
X,Y = np.meshgrid(xt, yt, indexing='ij')
V = np.concatenate((X.ravel()[np.newaxis,:], Y.ravel()[np.newaxis,:]), axis=0).T
W = np.concatenate((Dx.ravel()[np.newaxis,:], Dy.ravel()[np.newaxis,:]), axis=0).T + V

imr = np.reshape(interpn((xt, yt), img, W, 'linear', False, 0), targetdim)
# Error here: (xs, ys) should have been (xt,yt)
# imr = np.reshape(interpn((xs, ys), img, W, 'linear', False, 0), targetdim)
ax[1].imshow(imr, 'gray')
plt.plot(pnts[1,:], pnts[0,:], 'r.')
plt.plot(pntsr[1,:], pntsr[0,:], 'y*')
plt.show()


