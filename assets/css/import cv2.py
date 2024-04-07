import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load the image
image = cv2.imread('rockslide_image.jpg', cv2.IMREAD_GRAYSCALE)

# Apply image segmentation to discriminate rocks and soil
# Here, we use a simple thresholding technique
ret, segmented_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

# Add noise to the subset corresponding to rockslide
noise = np.random.normal(0, 20, segmented_image.shape).astype(np.uint8)
segmented_image += noise

# Apply median filter for periodic noise removal
filtered_image = cv2.medianBlur(segmented_image, 5)

# Display the final output
plt.figure(figsize=(10, 10))
plt.subplot(131), plt.imshow(image, cmap='gray'), plt.title('Original Image')
plt.subplot(132), plt.imshow(segmented_image, cmap='gray'), plt.title('Segmented Image with Noise')
plt.subplot(133), plt.imshow(filtered_image, cmap='gray'), plt.title('Filtered Image')
plt.show()
