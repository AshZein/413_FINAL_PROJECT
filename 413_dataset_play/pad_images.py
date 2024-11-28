import cv2
from PIL import Image
import os
import threading    

threaded = False
def pad_img(img_names, input_dir, output_dir, target_size=500,):
    global threaded
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
                
                cv2.imwrite(os.path.join(output_dir, filename), padded_img)
                        
            except Exception as e:
                print(f"Error opening image {filename}: {e}")
        count_done += 1
        if count_done % 50 == 0:
            if not threaded:
                if int(count_done / len(img_names) * 100) != last:
                    print(f"Progress: {int(count_done / len(img_names) * 100)}%")
                    last = int(count_done / len(img_names) * 100)
            else:
                if int(count_done / len(img_names) * 100) != last:
                    print(f"Thread {threading.get_ident()}Progress: {int(count_done / len(img_names) * 100)}%")
                    last = int(count_done / len(img_names) * 100)
            
if __name__ == "__main__":
    directory = "../flickr30k/flickr30k_images"
    padded_dir = "../flickr30k/flickr30k_images_padded"
    
    filenames = os.listdir(directory)
    
    threaded = True
    threads = []
    file_chunks = [] # list of list of filenames for each thread
    for t in range(4):
        filename_chunk = filenames[t * (len(filenames) // 4) : max(len(filenames), (t+1) * (len(filenames) // 4))]
        file_chunks.append(filename_chunk)
        
        thread = threading.Thread(target=pad_img, args=(filename_chunk, directory, padded_dir))
        threads.append(thread)
        thread.start()
        
    for thread in threads:
        thread.join()
        
    print("ALL DONE")

