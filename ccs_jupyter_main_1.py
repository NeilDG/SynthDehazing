import os
import transmission_main
def main():
    os.system("python \"transmission_main.py\" --num_workers=10 --img_to_load=-1 --load_previous=0 --server_config=2 --cuda_device=cuda:0 --batch_size=512 --iteration=8")
    os.system("python \"transmission_main.py\" --num_workers=10 --img_to_load=-1 --load_previous=0 --server_config=2 --cuda_device=cuda:0 --batch_size=512 --iteration=9")

if __name__ == "__main__":
    main()