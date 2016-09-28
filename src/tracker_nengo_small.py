import nstbot
import numpy as np
# from nstbot.connection import Serial
import nengo

# small = False
small = True
vis = False
# Load weight
if small:
    cam_dim = 20
    weightMatrix_v = np.load('weights/connectivity_v_small.npy')
    weightMatrix_h = np.load('weights/connectivity_h_small.npy')
    weightMatrix_corner = np.load('weights/connectivity_c_small.npy')
else:
    cam_dim = 128
    weightMatrix_v = np.load('weights/connectivity_v.npy')
    weightMatrix_h = np.load('weights/connectivity_h.npy')
    weightMatrix_corner = np.load('weights/connectivity_c.npy')

n_features = np.size(weightMatrix_v, 0)
n_corners = 25

edvs = nstbot.RetinaBot()
edvs.connect(nstbot.Serial('/dev/ttyUSB0', baud=6000000))
edvs.retina(True)
# Either we use show image to visualize what is happing
edvs.keep_image()
if vis:
    edvs.show_image()
print np.sum(weightMatrix_v), np.sum(weightMatrix_h), np.sum(weightMatrix_corner+1)
print cam_dim

def stim_func(t):
    # print edvs.image
    if small:
        # print edvs.image[50:70, 50:70].flatten()
        return edvs.image[50:70, 50:70].flatten()
    else:
        return edvs.image.flatten()


model = nengo.Network()
with model:
    stim = nengo.Node(stim_func)
    # print np.size(stim)
    if small:
        featureMap_v = nengo.Ensemble(
            n_neurons=n_features, dimensions=cam_dim**2, encoders=weightMatrix_v)
        featureMap_h = nengo.Ensemble(
            n_neurons=n_features, dimensions=cam_dim**2, encoders=weightMatrix_h)
        cornerLayer = nengo.Ensemble(
            n_neurons=n_corners, dimensions=4, encoders=weightMatrix_corner+1)

        nengo.Connection(stim, featureMap_v)
        nengo.Connection(stim, featureMap_h)

    else:
        featureMap_v = nengo.Ensemble(
            n_neurons=n_features, dimensions=cam_dim**2, encoders=weightMatrix_v)
        featureMap_h = nengo.Ensemble(
            n_neurons=n_features, dimensions=cam_dim**2, encoders=weightMatrix_h)
        cornerLayer = nengo.Ensemble(
            n_neurons=n_corners, dimensions=n_features, encoders=weightMatrix_corner)

        nengo.Connection(stim, featureMap_v)
        nengo.Connection(stim, featureMap_h)
    nengo.Connection(featureMap_v.neurons, cornerLayer)
    nengo.Connection(featureMap_h.neurons, cornerLayer)
