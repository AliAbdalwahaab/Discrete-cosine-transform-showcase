import numpy as np
import cv2
import matplotlib.pyplot as plt

def dct2(a):
    return cv2.dct(cv2.dct(a.T).T)

def idct2(a):
    return cv2.idct(cv2.idct(a.T).T)

img = cv2.imread('your image here.png', 0)

img = cv2.resize(img, (256, 256))

img = np.float32(img)
img = img - np.mean(img)

dct = dct2(img)

idct = idct2(dct)

zigzag = np.concatenate([np.diagonal(dct[::-1,:], i)[::(2*(i % 2)-1)] for i in range(1-dct.shape[0], dct.shape[0])])


zigzag = zigzag[:100]

plt.figure()

plt.subplot(141)
plt.imshow(img, cmap='gray')
plt.title('Original Image')

plt.subplot(142)
plt.imshow(dct, cmap='gray')
plt.title('DCT')

plt.subplot(144)
plt.plot(zigzag)
plt.title('Zigzag')
plt.xlabel('Index')
plt.ylabel('Value')

plt.subplot(143)
plt.plot(dct.flatten())
plt.title('DCT Vector')
plt.xlabel('Index')
plt.ylabel('Value')

plt.figure()
plt.imshow(np.log(abs(dct)), cmap='gray')
plt.title('DCT Coefficients')
plt.colorbar()

plt.show()

