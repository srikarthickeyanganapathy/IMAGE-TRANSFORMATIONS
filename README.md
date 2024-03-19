# IMAGE-TRANSFORMATIONS


## Aim:
To perform image transformation such as Translation, Scaling, Shearing, Reflection, Rotation and Cropping using OpenCV and Python.

## Software Required:
Anaconda - Python 3.7

## Algorithm:
### Step1:
Import numpy module as np and pandas as pd.

### Step2:
Assign the values to variables in the program.

### Step3:
Get the values from the user appropriately.
### Step4:
Continue the program by implementing the codes of required topics.
### Step5:
Thus the program is executed in google colab.
## Program:
```
Developed By: SRI KARTHICKEYAN GANAPATHY
Register Number:212222240102
```
### i)Image Translation
```PYTHON
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Read the input image
input_image = cv2.imread("cat.jpg")

# Convert from BGR to RGB so we can plot using matplotlib
input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)

# Disable x & y axis
plt.axis('off')

# Show the image
plt.imshow(input_image)
plt.show()

# Get the image shape
rows, cols, dim = input_image.shape

# Transformation matrix for translation
M = np.float32([[1, 0, 100],
                [0, 1, 200],
                [0, 0, 1]])  # Fixed the missing '0' and added correct dimensions

# Apply a perspective transformation to the image
translated_image = cv2.warpPerspective(input_image, M, (cols, rows))

# Disable x & y axis
plt.axis('off')

# Show the resulting image
plt.imshow(translated_image)
plt.show()

```
### ii) Image Scaling
```PYTHON
import cv2
import numpy as np
from matplotlib import pyplot as plt

# Function to display images in Colab
def show_image(image):
    plt.figure(figsize=(6, 6))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

# Load an image from URL or file path
image_url = 'nature.jpg'  # Replace with your image URL or file path
image = cv2.imread(image_url)

# Define scale factors
scale_x = 1.5  # Scaling factor along x-axis
scale_y = 1.5  # Scaling factor along y-axis

# Apply scaling to the image
scaled_image = cv2.resize(image, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)

# Display original and scaled images
print("Original Image:")
show_image(image)
print("Scaled Image:")
show_image(scaled_image)
```

### iii)Image shearing
```PYTHON
import cv2
import numpy as np
from matplotlib import pyplot as plt

# Function to display images in Colab
def show_image(image):
    plt.figure(figsize=(6, 6))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

# Load an image from URL or file path
image_url = '3nat.jpg'  # Replace with your image URL or file path
image = cv2.imread(image_url)

# Define shear parameters
shear_factor_x = 0.5  # Shear factor along x-axis
shear_factor_y = 0.2  # Shear factor along y-axis

# Define shear matrix
shear_matrix = np.float32([[1, shear_factor_x, 0], [shear_factor_y, 1, 0]])

# Apply shear to the image
sheared_image = cv2.warpAffine(image, shear_matrix, (image.shape[1], image.shape[0]))

# Display original and sheared images
print("Original Image:")
show_image(image)
print("Sheared Image:")
show_image(sheared_image)
```
### iv)Image Reflection
```PYTHON
import cv2
import numpy as np
from matplotlib import pyplot as plt

# Function to display images in Colab
def show_image(image):
    plt.figure(figsize=(6, 6))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

# Load an image from URL or file path
image_url = '4 nature.jpg'  # Replace with your image URL or file path
image = cv2.imread(image_url)

# Reflect the image horizontally
reflected_image_horizontal = cv2.flip(image, 1)

# Reflect the image vertically
reflected_image_vertical = cv2.flip(image, 0)

# Reflect the image both horizontally and vertically
reflected_image_both = cv2.flip(image, -1)

# Display original and reflected images
print("Original Image:")
show_image(image)
print("Reflected Horizontally:")
show_image(reflected_image_horizontal)
print("Reflected Vertically:")
show_image(reflected_image_vertical)
print("Reflected Both:")
show_image(reflected_image_both)
```
### v)Image Rotation
```PYTHON
import cv2
import numpy as np
from matplotlib import pyplot as plt

# Function to display images in Colab
def show_image(image):
    plt.figure(figsize=(6, 6))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

# Load an image from URL or file path
image_url = 'nat5.jpg'  # Replace with your image URL or file path
image = cv2.imread(image_url)

# Define rotation angle in degrees
angle = 45

# Get image height and width
height, width = image.shape[:2]

# Calculate rotation matrix
rotation_matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)

# Perform image rotation
rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))

# Display original and rotated images
print("Original Image:")
show_image(image)
print("Rotated Image:")
show_image(rotated_image)


```
### vi)Image Cropping
```PYTHON

import cv2
import numpy as np
from matplotlib import pyplot as plt

# Function to display images in Colab
def show_image(image):
    plt.figure(figsize=(6, 6))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

# Load an image from URL or file path
image_url = '6nat.jpg'  # Replace with your image URL or file path
image = cv2.imread(image_url)

# Define cropping coordinates (x, y, width, height)
x = 100  # Starting x-coordinate
y = 50   # Starting y-coordinate
width = 200  # Width of the cropped region
height = 150  # Height of the cropped region

# Perform image cropping
cropped_image = image[y:y+height, x:x+width]

# Display original and cropped images
print("Original Image:")
show_image(image)
print("Cropped Image:")
show_image(cropped_image)

```

## Output:
### i)Image Translation
![download](https://github.com/srikarthickeyanganapathy/IMAGE-TRANSFORMATIONS/assets/119393842/9078e74f-26d8-44de-9bef-d1e26438350c)
![download](https://github.com/srikarthickeyanganapathy/IMAGE-TRANSFORMATIONS/assets/119393842/24595835-0c94-4575-b366-bb5b54802e58)

### ii) Image Scaling
![download](https://github.com/srikarthickeyanganapathy/IMAGE-TRANSFORMATIONS/assets/119393842/611534cf-46ec-4f83-ae4d-6564765575eb)
![download](https://github.com/srikarthickeyanganapathy/IMAGE-TRANSFORMATIONS/assets/119393842/08680479-de10-4370-8720-de50cd0f2323)

### iii)Image shearing
![download](https://github.com/srikarthickeyanganapathy/IMAGE-TRANSFORMATIONS/assets/119393842/922fc9c1-f44b-4130-af07-17c911a1f1af)
![download](https://github.com/srikarthickeyanganapathy/IMAGE-TRANSFORMATIONS/assets/119393842/3f0b4922-e643-4b07-8c79-35aeff7ed3fe)

### iv)Image Reflection
![download](https://github.com/srikarthickeyanganapathy/IMAGE-TRANSFORMATIONS/assets/119393842/68fec26e-061b-4abb-b3eb-8e7f8e464f49)
![download](https://github.com/srikarthickeyanganapathy/IMAGE-TRANSFORMATIONS/assets/119393842/db395ac4-5cfc-4dbb-8ec6-abfaacb417e3)

### v)Image Rotation
![download](https://github.com/srikarthickeyanganapathy/IMAGE-TRANSFORMATIONS/assets/119393842/3dca92fb-5f6f-411b-91dd-b3b9b6f55204)
![download](https://github.com/srikarthickeyanganapathy/IMAGE-TRANSFORMATIONS/assets/119393842/bdf8c4bd-1b96-4918-9c8d-22e566ac9f4c)

### vi)Image Cropping
![download](https://github.com/srikarthickeyanganapathy/IMAGE-TRANSFORMATIONS/assets/119393842/7c3532f7-a5ac-41ec-956a-a9c049b70650)
![download](https://github.com/srikarthickeyanganapathy/IMAGE-TRANSFORMATIONS/assets/119393842/9a0ee1f5-c27d-408b-82b0-423ca38185b6)

## Result: 

Thus the different image transformations such as Translation, Scaling, Shearing, Reflection, Rotation and Cropping are done using OpenCV and python programming.
