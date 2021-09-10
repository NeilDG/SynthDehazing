import os
def main():
    # os.system("python \"transmission_main.py\" --num_workers=12 --img_to_load=-1 --load_previous=1 --iteration=6 --batch_size=128 --patch_size=32 --t_min=0.6 --t_max=1.8 --a_min=0.1 --a_max=0.98")
    # os.system("python \"airlight_main.py\" --num_workers=12 --img_to_load=-1 --load_previous=0 --iteration=6 --batch_size=4096 --patch_size=32 --t_min=0.6 --t_max=1.8 --a_min=0.1 --a_max=0.98")
    os.system("python \"cyclegan_main.py\" --num_workers=12 --img_to_load=-1 --load_previous=0 --iteration=1 --batch_size=256 --num_blocks=6 --t_min=0.6 --t_max=1.8 --a_min=0.1 --a_max=0.98")
if __name__ == "__main__":
    main()