import cv2
import os
import threading
from tqdm import tqdm
from PIL import Image

def pad_image(img, target_size=500):
    h, w = img.shape[:2]
    scale = target_size / max(h, w)
    resized_img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    
    pad_height = target_size - resized_img.shape[0]
    pad_width = target_size - resized_img.shape[1]
    top = pad_height // 2
    bottom = pad_height - top
    left = pad_width // 2
    right = pad_width - left
    
    return cv2.copyMakeBorder(resized_img, top, bottom, left, right, 
                             cv2.BORDER_CONSTANT, value=[0, 0, 0])

def process_images(input_dir, output_dir, num_threads=8):
    os.makedirs(output_dir, exist_ok=True)
    filenames = [f for f in os.listdir(input_dir) 
                if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    def verify_image(img_path):
        try:
            img = Image.open(img_path)
            img.verify()
            return True
        except:
            return False
    
    def process_chunk(chunk):
        for filename in chunk:
            try:
                img_path = os.path.join(input_dir, filename)
                if not verify_image(img_path):
                    print(f"Corrupted image: {filename}")
                    continue
                    
                img = cv2.imread(img_path)
                if img is None:
                    print(f"Failed to read image: {filename}")
                    continue
                    
                padded_img = pad_image(img)
                cv2.imwrite(os.path.join(output_dir, filename), padded_img)
            except Exception as e:
                print(f"Error processing {filename}: {e}")
    
    # 分割文件列表
    chunk_size = len(filenames) // num_threads
    chunks = [filenames[i:i + chunk_size] for i in range(0, len(filenames), chunk_size)]
    
    threads = []
    for chunk in chunks:
        thread = threading.Thread(target=process_chunk, args=(chunk,))
        threads.append(thread)
        thread.start()
    
    for thread in threads:
        thread.join() 