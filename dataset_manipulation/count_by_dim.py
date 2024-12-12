import os
from PIL import Image

def count_each_dim(directory):
    counts = {}

    for filename in os.listdir(directory):
        if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            try:
                with Image.open(os.path.join(directory, filename)) as img:
                    width, height = img.size
                    if (width, height) in counts:
                        counts[(width, height)] += 1
                    else:
                        counts[(width, height)] = 1
                    
            except Exception as e:
                print(f"Error opening image {filename}: {e}")

    return counts

# Example usage
directory_path = '../flickr30k/flickr30k_images'
count_dim = count_each_dim(directory_path)
count_dim = {k: v for k, v in sorted(count_dim.items(), key=lambda item: item[1])}
print(count_dim)
