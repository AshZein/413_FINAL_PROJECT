import os
from PIL import Image

def count_non_standard_images(directory):
    count = 0
    max_width = 0
    max_height = 0
    for filename in os.listdir(directory):
        if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            try:
                with Image.open(os.path.join(directory, filename)) as img:
                    width, height = img.size
                    if width > max_width:
                        max_width = width
                    if height > max_height:
                        max_height = height
                    if (width, height) not in [(500, 375), (375, 500)]:

                        count += 1
            except Exception as e:
                print(f"Error opening image {filename}: {e}")
    print(max_width, max_height)
    return count

# Example usage
directory_path = '../flickr30k/flickr30k_images'
non_standard_count = count_non_standard_images(directory_path)
print(f"Number of non-standard images: {non_standard_count}")
