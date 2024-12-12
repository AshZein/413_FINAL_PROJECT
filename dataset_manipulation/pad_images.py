import cv2
import os
import sys 
import threading 

import kagglehub

def pad_img(img_names, input_dir, output_dir, target_size=500,):
    # apply max width and height
    count_done = 0
    last = 0
    for filename in img_names:
        if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            try:
                img = cv2.imread(os.path.join(input_dir, filename))
                
                # Resize maintaining aspect ratio 
                h, w = img.shape[:2] 
                scale = target_size / max(h, w) 
                resized_img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
                
                # Calculate padding 
                pad_height = target_size - resized_img.shape[0] 
                pad_width = target_size - resized_img.shape[1] 
                top = pad_height // 2 
                bottom = pad_height - top 
                left = pad_width // 2 
                right = pad_width - left
                
                padded_img = cv2.copyMakeBorder(resized_img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
                padded_img = cv2.resize(padded_img, (224, 224))
                cv2.imwrite(os.path.join(output_dir, filename), padded_img)
                        
            except Exception as e:
                print(f"Error opening image {filename}: {e}")
        count_done += 1
        if count_done % 50 == 0:
            if int(count_done / len(img_names) * 100) != last:
                print(f"Progress: {int(count_done / len(img_names) * 100)}%")
                last = int(count_done / len(img_names) * 100)
if __name__ == "__main__":
    # Download latest version
    if len(sys.argv) == 1:
        path = kagglehub.dataset_download("eeshawn/flickr30k") 
        directory = path + "/flickr30k_images"
        padded_dir = path + "/flickr30k_images_padded"
        try:
            os.mkdir(padded_dir)
        except:
            pass
        print("Path to dataset files:", path)
        
    else:
        directory = sys.argv[1]
        padded_dir = directory + "_padded"
        try:
            os.mkdir(padded_dir)
        except:
            print(f"directory {padded_dir} already exists")
        
    try:
        filenames = os.listdir(directory)

        threads = []
        file_chunks = [] # list of list of filenames for each thread
        for t in range(8):
            filename_chunk = filenames[t * (len(filenames) // 4) : max(len(filenames), (t+1) * (len(filenames) // 4))]
            file_chunks.append(filename_chunk)
            
            thread = threading.Thread(target=pad_img, args=(filename_chunk, directory, padded_dir))
            threads.append(thread)
            thread.start()
            
        for thread in threads:
            thread.join()
            
        print("ALL DONE")
    except:
        print(f"No directory: {directory}")

