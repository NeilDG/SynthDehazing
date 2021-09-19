import os
def main():
    os.system("python \"cyclegan_main.py\" --num_workers=10 --img_to_load=-1 --load_previous=0 --iteration=4 --server_config=2 --cuda_device=cuda:1 --batch_size=4096 --num_blocks=8 --net_config=2 --use_bce=0")
if __name__ == "__main__":
    main()