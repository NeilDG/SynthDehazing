import os
def main():
    os.system("python \"cyclegan_main.py\" --num_workers=12 --img_to_load=-1 --load_previous=0 --iteration=2 --batch_size=256 --num_blocks=10 --net_config=1 --use_bce=0")
    # os.system("python \"transmission_main.py\" --num_workers=12 --img_to_load=-1 --load_previous=1 --iteration=9 --batch_size=128 --t_min=0.3 --t_max=0.9 --a_min=0.35 --a_max=0.5")
    # os.system("python \"airlight_main.py\" --num_workers=12 --img_to_load=-1 --load_previous=0 --iteration=9 --batch_size=8192 --t_min=0.3 --t_max=0.9 --a_min=0.35 --a_max=0.5")
    # os.system("python \"transmission_main.py\" --num_workers=12 --img_to_load=-1 --load_previous=0 --iteration=10 --batch_size=128 --t_min=0.05 --t_max=0.9 --a_min=0.35 --a_max=0.5")
    # os.system("python \"airlight_main.py\" --num_workers=12 --img_to_load=-1 --load_previous=0 --iteration=10 --batch_size=8192 --t_min=0.05 --t_max=0.9 --a_min=0.35 --a_max=0.5")
if __name__ == "__main__":
    main()