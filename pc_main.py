import os
def main():
    os.system("python \"transmission_main.py\" --num_workers=12 --img_to_load=-1 --load_previous=0 --iteration=1 --batch_size=128 --patch_size=32 --a_min=0.1 --a_max=0.98")
    os.system("python \"airlight_main.py\" --num_workers=12 --img_to_load=-1 --load_previous=0 --iteration=1 --batch_size=16384 --patch_size=32 --a_min=0.1 --a_max=0.98")

if __name__ == "__main__":
    main()