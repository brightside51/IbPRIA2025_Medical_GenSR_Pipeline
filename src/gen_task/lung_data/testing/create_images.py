from PIL import Image

# Open the images
image1 = Image.open('images/slice5.png')
image2 = Image.open('images/slice24.png')
image3 = Image.open('images/slice44.png')
image4 = Image.open('images/slice55.png')
image5 = Image.open('images/slice74.png')
image6 = Image.open('images/slice83.png')

# Get dimensions of each image
width1, height1 = image1.size
width2, height2 = image2.size
width3, height3 = image3.size
width4, height4 = image4.size
width5, height5 = image5.size
width6, height6 = image6.size

# Calculate the total width and maximum height
total_width = width1 + width2 + width3 + width4 + width5 + width6
max_height = max(height1, height2, height3, height4, height5, height6)

# Create a new blank image with the appropriate dimensions
new_image = Image.new('RGB', (total_width, max_height))

# Paste each image into the new image
new_image.paste(image1, (0, 0))
new_image.paste(image2, (width1, 0))
new_image.paste(image3, (width1 + width2, 0))
new_image.paste(image4, (width1 + width2 + width3, 0))
new_image.paste(image5, (width1 + width2 + width3 + width4, 0))
new_image.paste(image6, (width1 + width2 + width3 + width4 + width5, 0))

# Save the new image
new_image.save('combined_image.jpg')