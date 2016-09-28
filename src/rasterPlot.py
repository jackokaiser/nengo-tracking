import nengo.utils.matplotlib as nengo_plt
import matplotlib.pyplot as plt
import numpy as np
import nengo

featureMap_h = np.load('./outputs/featureMap_h.npy')
cornerLayer = np.load('./outputs/cornerLayer.npy')

trange = np.arange(len(featureMap_h)) * 0.001
# print len(featureMap_h)
plt.figure(1)
plt.subplot(121)
nengo_plt.rasterplot(trange, featureMap_h)
plt.subplot(122)
nengo_plt.rasterplot(trange, cornerLayer)
print np.shape(cornerLayer)
# cornerLayer = nengo.synapses.Lowpass(tau=0.03).filt(cornerLayer, dt=0.001)
cornerLayer = np.reshape(cornerLayer.T, (np.sqrt(
    np.size(cornerLayer, 1)), np.sqrt(np.size(cornerLayer, 1)), -1))
# print cornerLayer[:, :, 19500:19999]
index = np.linspace(0, 19999, 9)
plt.figure(2)
for i in range(len(index)):
    # print index[i]
    plt.subplot(3, 3, i + 1)
    plt.imshow(cornerLayer[:, :, int(index[i])])
    # print cornerLayer[:, :, int(index[i])]
plt.show()
