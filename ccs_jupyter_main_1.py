import os
def main():
    os.system("python \"transmission_main.py\" --num_workers=10 --img_to_load=-1 --load_previous=0 --iteration=4 --server_config=2 --batch_size=512 --t_min=0.6 --t_max=1.8 --a_min=0.7 --a_max=1.0")
    os.system("python \"airlight_main.py\" --num_workers=10 --img_to_load=-1 --load_previous=0 --iteration=4 --server_config=2 --batch_size=32768 --t_min=0.6 --t_max=1.8 --a_min=0.7 --a_max=1.0")

if __name__ == "__main__":
    main()