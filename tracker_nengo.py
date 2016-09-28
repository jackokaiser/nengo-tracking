import nstbot
import numpy as np
# from nstbot.connection import Serial
import nengo
test = False
# Load weight
weightMatrix_v = np.load('weights/connectivity_v.npy')
weightMatrix_h = np.load('weights/connectivity_h.npy')
weightMatrix_corner = np.load('weights/connectivity_c.npy')

n_features = np.size(weightMatrix_v, 0)
n_corners = 25
cam_dim = 128
edvs = nstbot.RetinaBot()
edvs.connect(nstbot.Serial('/dev/ttyUSB0', baud=12000000))
edvs.retina(True)
# Either we use show image to visualize what is happing
# edvs.keep_image()
edvs.show_image()


def stim_func(t):
    # print edvs.image
    if test:
        return edvs.image[:10, : 10].flatten()
    else:
        return edvs.image.flatten()


model = nengo.Network()
with model:
    stim = nengo.Node(stim_func)
    if test:
        # featureMap_v = nengo.Ensemble(n_neurons=10, dimensions=10 * 10)
        # featureMap_h = nengo.Ensemble(n_neurons=10, dimensions=10 * 10)
        # nengo.Connection(stim, featureMap_v.neurons,
        #                  transform=weightMatrix_v[:10, :100])
        # nengo.Connection(stim, featureMap_h.neurons,
        #                  transform=weightMatrix_h[:10, :100])
        featureMap_v = nengo.Ensemble(n_neurons=10, dimensions=10**2)
        featureMap_h = nengo.Ensemble(n_neurons=10, dimensions=10**2)
        cornerLayer = nengo.Ensemble(
            n_neurons=n_corners, dimensions=10, encoders=weightMatrix_corner[:, :10])

        nengo.Connection(stim, featureMap_v.neurons, transform=weightMatrix_v[:10, :100])
        nengo.Connection(stim, featureMap_h.neurons, transform=weightMatrix_h[:10, :100])

    else:
        featureMap_v = nengo.Ensemble(
            n_neurons=n_features, dimensions=cam_dim**2, encoders=weightMatrix_v,
            intercepts=nengo.dists.Uniform(0, .5))
        featureMap_h = nengo.Ensemble(
            n_neurons=n_features, dimensions=cam_dim**2, encoders=weightMatrix_h,
            intercepts=nengo.dists.Uniform(0, .5))
        cornerLayer = nengo.Ensemble(
            n_neurons=n_corners, dimensions=n_features, encoders=weightMatrix_corner,
            intercepts=nengo.dists.Uniform(0, .5))

        nengo.Connection(stim, featureMap_v)
        nengo.Connection(stim, featureMap_h)
    nengo.Connection(featureMap_v.neurons, cornerLayer)
    nengo.Connection(featureMap_h.neurons, cornerLayer)
