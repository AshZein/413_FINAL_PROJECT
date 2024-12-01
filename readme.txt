# install requirements
pip install -r requirements.txt

# train model
python train.py \
    --image_dir flickr30k/images \
    --caption_file flickr30k/captions.txt \
    --save_dir checkpoints \
    --batch_size 32 \
    --epochs 30 \
    --learning_rate 0.001 \
    --embed_size 256 \
    --hidden_size 512


# inference
python inference.py \
    --image_path path/to/test_image.jpg \
    --model_path checkpoints/best_model.pth