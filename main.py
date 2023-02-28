import numpy as np
from scipy import ndimage
from matplotlib import pyplot as plt

data = np.load('chestmnist.npz')

image = data['train_images'][3]

plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.show()
inverted_image = np.invert(image)
plt.imshow(inverted_image, cmap='gray')
plt.title('Inverted Image')
plt.show()
blurred_image = ndimage.gaussian_filter(image, sigma=3)
plt.imshow(blurred_image, cmap='gray')
plt.title('Blurred Image')
plt.show()