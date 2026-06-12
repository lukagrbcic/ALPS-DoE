import numpy as np
from PIL import Image

img = Image.open("unitCell.jpeg").convert("L")   # load as grayscale
arr = np.array(img)
print(arr.min(), arr.max())
print("fraction white:", (arr == 255).mean())
bitmap = (arr < 128).astype(int)   # cross = 1
import matplotlib.pyplot as plt

plt.imshow(bitmap, cmap='gray_r')
#plt.axis('off')
plt.show()