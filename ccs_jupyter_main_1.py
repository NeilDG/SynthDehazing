import os
def main():
    os.system("python \"cyclegan_main.py\" --num_workers=10 --img_to_load=-1 --load_previous=0 --iteration=4 --server_config=2 --cuda_device=cuda:1 --batch_size=2048 --num_blocks=4 --net_config=3 --use_bce=0 --likeness_weight=1.0")
if __name__ == "__main__":
    main()