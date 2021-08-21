import os
import transmission_main
def main():
    os.system("python \"transmission_main.py\" --num_workers=10 --img_to_load=-1 --load_previous=0 --server_config=2 --cuda_device=cuda:1 --batch_size=256 --iteration=6 --t_min=0.1 --t_max=1.7 --a_min=0.1 --a_max=0.95")
    os.system("python \"transmission_main.py\" --num_workers=10 --img_to_load=-1 --load_previous=0 --server_config=2 --cuda_device=cuda:1 --batch_size=256 --iteration=7 --t_min=0.1 --t_min=1.9 --t_max=0.95 --a_min=0.1 --a_max=1.2")
    os.system("python \"airlight_main.py\" --num_workers=10 --img_to_load=-1 --load_previous=0 --server_config=2 --cuda_device=cuda:1 --batch_size=16384 --iteration=6 --t_min=0.1 --t_max=1.7 --a_min=0.1 --a_max=0.95")
    os.system("python \"airlight_main.py\" --num_workers=10 --img_to_load=-1 --load_previous=0 --server_config=2 --cuda_device=cuda:1 --batch_size=16384 --iteration=7 --t_min=0.1 --t_min=1.9 --t_max=0.95 --a_min=0.1 --a_max=1.2")

if __name__ == "__main__":
    main()