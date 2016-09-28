import numpy as np
import matplotlib.pyplot as plt
import Image

# Flags
# visualize = False
visualize = True


def center_ind(x_dim, kernel_dim, stride=1, offset=0, return_params=False):
    hw = kernel_dim / 2
    if return_params:
        return np.arange(offset, x_dim - offset, kernel_dim - stride) + hw, hw
    else:
        return np.arange(offset, x_dim - offset, kernel_dim - stride) + hw


dim_x = 128
dim_y = 128
weightMatrix_h = np.zeros([dim_x, dim_y])
weightMatrix_v = np.zeros([dim_x, dim_y])
target_Features = np.zeros([dim_x, dim_y])

ind_Features, hw = center_ind(x_dim=dim_x, kernel_dim=10, stride=1, offset=0, return_params=True)
ind_Features = np.delete(ind_Features, np.where(ind_Features > dim_x))
ind_Corners, hw_corner = center_ind(x_dim=len(ind_Features), kernel_dim=3, stride=0, offset=0, return_params=True)

target_Corners = np.zeros([len(ind_Features), len(ind_Features)])

kernel_h = plt.imread('kernels/horiz_line_0.png')
kernel_v = plt.imread('kernels/vert_line_0.png')
counter = 0
for x in ind_Features:
    for y in ind_Features:
        weightMatrix_h[x - hw:x + hw, y - hw:y + hw] = kernel_h
        weightMatrix_v[x - hw:x + hw, y - hw:y + hw] = kernel_v
        target_Features[x - hw:x + hw, y - hw:y + hw] = counter
        counter += 1
        # weightMatrix_h[x, y] = 0
        # weightMatrix_v[x, y] = 0

counter = 1
for x in ind_Corners:
    for y in ind_Corners:
        target_Corners[x - hw_corner: x + hw_corner, y - hw_corner:y + hw_corner] = counter
        counter += 1


if visualize:
    plt.figure(1)
    plt.subplot(121)
    plt.imshow(weightMatrix_v)
    plt.subplot(122)
    plt.imshow(weightMatrix_h)
    plt.figure(2)
    plt.subplot(111)
    plt.imshow(target_Corners)
    plt.title('Connectivity Matrix FM to Cornerlayer')
    plt.show()

np.save('weights/weightMatrix_h', weightMatrix_h)
np.save('weights/weightMatrix_v', weightMatrix_v)
np.save('weights/targetFeatures.npy', target_Features)
np.save('weights/targetCorners.npy', target_Corners)
