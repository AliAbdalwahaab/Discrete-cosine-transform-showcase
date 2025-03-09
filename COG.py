import cv2
import numpy as np
import matplotlib.pyplot as plt

def dct_4_blocks(image):
    h, w = image.shape
    dct_blocks = []
    block_size = 4

    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            block = image[i:i+block_size, j:j+block_size]
            dct_block = cv2.dct(np.float32(block))
            dct_blocks.append(dct_block)

    return dct_blocks

def center_of_gravity(image):
    dct_blocks = dct_4_blocks(image)
    total_weight = 0
    x_weighted_sum = 0
    y_weighted_sum = 0
    block_size = 4

    for i, block in enumerate(dct_blocks):
        weight = np.sum(block)
        total_weight += weight
        x = (i % (image.shape[1] // block_size)) * block_size + block_size / 2
        y = (i // (image.shape[1] // block_size)) * block_size + block_size / 2
        x_weighted_sum += x * weight
        y_weighted_sum += y * weight

    center_x = x_weighted_sum / total_weight
    center_y = y_weighted_sum / total_weight

    return center_x, center_y


image_path = 'Your image here.png'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
center_x, center_y = center_of_gravity(image)
print(f"Center of Gravity: ({center_x}, {center_y})")
plt.imshow(image, cmap='gray')
plt.scatter(center_x, center_y, color='red')
plt.axvline(center_x, color='red', linestyle='--')
plt.axhline(center_y, color='red', linestyle='--')
plt.title('Center of Gravity')
plt.show()
