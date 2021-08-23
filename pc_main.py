import os
import transmission_main
def main():
    # os.system("python \"transmission_main.py\" --num_workers=12 --img_to_load=-1 --load_previous=1 --iteration=97 --style_transfer_enabled=0")
    # os.system("python \"airlight_main.py\" --num_workers=12 --img_to_load=-1 --load_previous=1 --iteration=97 --style_transfer_enabled=0")
    # os.system("python \"transmission_main.py\" --num_workers=12 --img_to_load=-1 --load_previous=1 --iteration=98 --style_transfer_enabled=1 --unlit_enabled=0")
    # os.system("python \"airlight_main.py\" --num_workers=12 --img_to_load=-1 --load_previous=1 --iteration=98 --style_transfer_enabled=1")
    os.system("python \"transmission_main.py\" --num_workers=12 --img_to_load=-1 --load_previous=1 --iteration=99 --style_transfer_enabled=0 --unlit_enabled=0")
    os.system("python \"airlight_main.py\" --num_workers=12 --img_to_load=-1 --load_previous=0 --iteration=99 --style_transfer_enabled=0")

if __name__ == "__main__":
    main()