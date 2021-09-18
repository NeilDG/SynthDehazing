import os
def main():
    os.system("python \"cyclegan_main.py\" --num_workers=10 --img_to_load=-1 --load_previous=0 --iteration=2 --server_config=2 --cuda_device=cuda:1 --batch_size=1024 --num_blocks=10 --net_config=1 --use_bce=0")
if __name__ == "__main__":
    main()