import numpy as np
import matplotlib.pyplot as plt

featureMap_v = np.load('./weights/weightMatrix_v.npy')
featureMap_h = np.load('./weights/weightMatrix_h.npy')
targetFeatures = np.load('./weights/targetFeatures.npy')
targetCorners = np.load('./weights/targetCorners.npy')

fm_v = featureMap_v.flatten()
fm_h = featureMap_h.flatten()
tgF = targetFeatures.flatten()
print np.size(featureMap_h, 0)
if np.size(featureMap_h, 0) < 128:
    small = True
else:
    small = False

uniqueEntries_features = np.unique(tgF)
cornerLayer_dim = 25

connectivityFeatures_h = np.zeros([len(uniqueEntries_features), len(fm_v)])
connectivityFeatures_v = np.zeros([len(uniqueEntries_features), len(fm_v)])

connectivityCornerLayer = np.zeros(
    [cornerLayer_dim, len(uniqueEntries_features)])

for i in range(len(uniqueEntries_features)):
    cIndex = tgF == uniqueEntries_features[i]
    connectivityFeatures_h[i, cIndex] = fm_h[cIndex]
    connectivityFeatures_v[i, cIndex] = fm_v[cIndex]

unique_Entries_corner = np.unique(targetCorners)
unique_Entries_corner = np.delete(unique_Entries_corner, 0)
connectivityCorners = np.zeros([cornerLayer_dim, len(uniqueEntries_features)])
for i in range(len(unique_Entries_corner)):
    cIndex = targetCorners.flatten() == unique_Entries_corner[i]
    connectivityCorners[i, cIndex] = 1

plt.imshow(connectivityCorners)
plt.show()

print 'Feature Layer: ', np.shape(connectivityFeatures_h)
print 'Corner Layer: ', np.shape(connectivityCorners)

if small:
    np.save('./weights/connectivity_h_small.npy', connectivityFeatures_h)
    np.save('./weights/connectivity_v_small.npy', connectivityFeatures_v)
    np.save('./weights/connectivity_c_small.npy', connectivityCorners)
else:
    np.save('./weights/connectivity_h.npy', connectivityFeatures_h)
    np.save('./weights/connectivity_v.npy', connectivityFeatures_v)
    np.save('./weights/connectivity_c.npy', connectivityCorners)
print 'Saving Done!'
