PKG = 'moritz_nengo_snn_object_tracking'
import roslib
roslib.load_manifest(PKG)
import ipdb
import nstbot
import numpy as np
import nengo
import rospy
from std_msgs.msg import Int64


class output_ros():
    def __init__(self):
        self._initialized = False

        # Init Node:
        self.name = "nengo_snn"
        rospy.init_node(self.name)

        # Init Publishers:
        self._pan_pub = rospy.Publisher("/head/pan", Int64, queue_size=1)
        self._tilt_pub = rospy.Publisher("/head/tilt", Int64, queue_size=1)

        # Done:
        self._initialized = True

    def publish(self, step, arr):
        if self._initialized:
            arr.reshape(5, 5)

            result = np.argmax(
                arr) / arr.shape[0] - 2, np.argmax(arr) % arr.shape[0] - 2

            # Publish results:
            self._pan_pub.publish(result[0])
            self._tilt_pub.publish(result[1])

            # Log:
            rospy.loginfo("Moving by " + str(result) +
                          ". Timestep: " + str(step))


publisher = output_ros()

# Load weight
weightMatrix_v = np.load('weights/connectivity_v.npy')
weightMatrix_h = np.load('weights/connectivity_h.npy')
weightMatrix_corner = np.load('weights/connectivity_c.npy')

n_features = np.size(weightMatrix_v, 0)
n_corners = 25

cam_dim = 128
edvs = nstbot.RetinaBot()
edvs.connect(nstbot.Serial('/dev/ttyUSB0', baud=4000000))
edvs.retina(True)
edvs.show_image()
prob = 0.3
elementsToDelete = int(np.floor(prob * n_features))


def stim_func(t):
    stim = edvs.image.flatten()
    ind_delete = np.random.rand(len(stim))
    stim[ind_delete < 0.5] = 0
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
                     solver=DummySolver(np.ones((1, n_corners))), synapse=0.1,
                     transform=np.ones((n_corners, 1)) * -0.1)

if __name__ == '__main__':
    import nengo_spinnaker
    import logging
    rec = False
    publisher = output_ros()
    logging.basicConfig(level=logging.DEBUG)
    if rec:
        with model:
            probes_v = [nengo.Probe(e.neurons) for e in featureMap_v.ensembles]
            probes_h = [nengo.Probe(e.neurons) for e in featureMap_h.ensembles]
            probe_c = nengo.Probe(cornerLayer.neurons)

    sim = nengo_spinnaker.Simulator(model)
    # sim = nengo.Simulator(model)

    print 'Starting simulation SpiN'
    while True:
        sim.run(20)
    if rec:
        data_v = np.hstack([sim.data[p] for p in probes_v])
        data_h = np.hstack([sim.data[p] for p in probes_h])
        data_c = sim.data[probe_c]

        np.save('./outputs/featureMap_v.npy', data_v)
        np.save('./outputs/featureMap_h.npy', data_h)
        np.save('./outputs/cornerLayer.npy', data_c)
        print 'Saving Done!'
