import nstbot
import numpy as np
# from nstbot.connection import Serial
import nengo
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
prob = 0.0
elementsToDelete = int(np.floor(prob * n_features))


def stim_func(t):
    stim = edvs.image
    stim = np.flipud(np.fliplr(stim))
    stim = stim.flatten()
    ind_delete = np.random.rand(len(stim))
    stim[ind_delete <= 0.6] = 0
    return stim


model = nengo.Network()
with model:
    stim = nengo.Node(stim_func)
    ind_delete = []
    while len(ind_delete) < elementsToDelete:
        ind_delete = np.where(np.random.rand(n_features) < prob)[0]
    if len(ind_delete) > elementsToDelete:
        ind_delete = np.delete(ind_delete, np.arange(
            elementsToDelete, len(ind_delete)))

    featureMap_v = nengo.Network()
    for i in range(n_features):
        if i in ind_delete:
            continue
        w = weightMatrix_v[i, :]
        indices = np.where(w != 0)[0]
        with featureMap_v:
            e = nengo.Ensemble(n_neurons=1, dimensions=len(indices),
                               encoders=[w[indices]], intercepts=nengo.dists.Uniform(0, 0.5))
        c = nengo.Connection(stim[indices], e)
    featureMap_h = nengo.Network()
    for i in range(n_features):
        if i in ind_delete:
            continue
        w = weightMatrix_h[i, :]
        indices = np.where(w != 0)[0]
        with featureMap_h:
            e = nengo.Ensemble(n_neurons=1, dimensions=len(indices),
                               encoders=[w[indices]], intercepts=nengo.dists.Uniform(0, 0.5))
        nengo.Connection(stim[indices], e)
    cornerLayer = nengo.Ensemble(
        n_neurons=n_corners, dimensions=n_features,
        encoders=weightMatrix_corner, intercepts=nengo.dists.Uniform(0, 0.5))

    class DummySolver(nengo.solvers.Solver):
        def __init__(self, fixed):
            super(DummySolver, self).__init__(weights=False)
            self.fixed = fixed

        def __call__(self, A, Y, rng=None, E=None):
            return self.fixed, {}

    def dummy_func(x):
        return np.zeros(n_features)

    for i in range(n_features - elementsToDelete):
        nengo.Connection(featureMap_v.ensembles[i], cornerLayer,
                         function=dummy_func,
                         solver=DummySolver(np.eye(n_features)[i:i + 1, :]))
        nengo.Connection(featureMap_h.ensembles[i], cornerLayer,
                         function=dummy_func,
                         solver=DummySolver(np.eye(n_features)[i:i + 1, :]))

    output = nengo.Node(publisher.publish, size_in=n_corners)
    nengo.Connection(cornerLayer, output, function=(lambda x: np.zeros(n_corners)),
                     solver=DummySolver(np.eye(n_corners)), synapse=0.1)
    nengo.Connection(cornerLayer, cornerLayer.neurons, function=(lambda x: 0),
                     solver=DummySolver(np.ones((n_corners, 1))), synapse=0.2,
                     transform=np.ones((n_corners, 1)) * -10.)


    nengo.Connection(stim, featureMap_v)
    nengo.Connection(stim, featureMap_h)
    nengo.Connection(featureMap_v.neurons, cornerLayer)
    nengo.Connection(featureMap_h.neurons, cornerLayer)
