import os

def main():
    # os.system("python \"transmission_main.py\" --server_config=3 --num_workers=8 --img_to_load=-1 --load_previous=1 "
    #           "--version_name=transmission_albedo_estimator_v1.17 --iteration=13 --patch_size=64 --batch_size=128 --albedo_checkpt=\"checkpoint/albedo_transfer_v1.05_1.pt\" "
    #           "--t_min=0.6 --t_max=1.8 --a_min=0.7 --a_max=1.0 ")
    #
    # os.system("python \"transmission_main.py\" --server_config=3 --num_workers=8 --img_to_load=-1 --load_previous=1 "
    #           "--version_name=transmission_albedo_estimator_v1.17 --iteration=14 --patch_size=64 --batch_size=128 --albedo_checkpt=\"checkpoint/albedo_transfer_v1.05_1.pt\" "
    #           "--t_min=0.6 --t_max=1.8 --a_min=0.7 --a_max=1.0 ")

    os.system("python \"transmission_main.py\" --server_config=3 --num_workers=8 --img_to_load=-1 --load_previous=0 "
              "--version_name=transmission_albedo_estimator_v1.17 --iteration=15 --patch_size=64 --batch_size=256 --albedo_checkpt=\"checkpoint/albedo_transfer_v1.05_1.pt\" "
              "--t_min=0.6 --t_max=1.8 --a_min=0.7 --a_max=1.0 ")

    # os.system("python \"transmission_main.py\" --server_config=3 --num_workers=8 --img_to_load=-1 --load_previous=0 "
    #           "--version_name=transmission_albedo_estimator_v1.17 --iteration=16 --patch_size=64 --batch_size=128 --albedo_checkpt=\"checkpoint/albedo_transfer_v1.05_1.pt\" "
    #           "--t_min=0.6 --t_max=1.8 --a_min=0.7 --a_max=1.0 ")
    #
    # os.system("python \"transmission_main.py\" --server_config=3 --num_workers=8 --img_to_load=-1 --load_previous=0 "
    #           "--version_name=transmission_albedo_estimator_v1.17 --iteration=17 --patch_size=64 --batch_size=128 --albedo_checkpt=\"checkpoint/albedo_transfer_v1.05_1.pt\" "
    #           "--t_min=0.6 --t_max=1.8 --a_min=0.7 --a_max=1.0 ")

    os.system("python \"transmission_main.py\" --server_config=3 --num_workers=8 --img_to_load=-1 --load_previous=1 "
              "--version_name=transmission_albedo_estimator_v1.17 --iteration=18 --patch_size=64 --batch_size=128 --albedo_checkpt=\"checkpoint/albedo_transfer_v1.05_1.pt\" "
              "--t_min=0.6 --t_max=1.8 --a_min=0.7 --a_max=1.0 ")

    os.system("python \"transmission_main.py\" --server_config=3 --num_workers=8 --img_to_load=-1 --load_previous=0 "
              "--version_name=transmission_albedo_estimator_v1.17 --iteration=19 --patch_size=64 --batch_size=128 --albedo_checkpt=\"checkpoint/albedo_transfer_v1.05_1.pt\" "
              "--t_min=0.6 --t_max=1.8 --a_min=0.7 --a_max=1.0 ")

    os.system("python \"transmission_main.py\" --server_config=3 --num_workers=8 --img_to_load=-1 --load_previous=0 "
              "--version_name=transmission_albedo_estimator_v1.17 --iteration=20 --patch_size=64 --batch_size=128 --albedo_checkpt=\"checkpoint/albedo_transfer_v1.05_1.pt\" "
              "--t_min=0.6 --t_max=1.8 --a_min=0.7 --a_max=1.0 ")

    os.system("python \"transmission_main.py\" --server_config=3 --num_workers=8 --img_to_load=-1 --load_previous=0 "
              "--version_name=transmission_albedo_estimator_v1.17 --iteration=21 --batch_size=512 --albedo_checkpt=\"checkpoint/albedo_transfer_v1.05_1.pt\" "
              "--t_min=0.6 --t_max=1.8 --a_min=0.7 --a_max=1.0 ")

    os.system("python \"transmission_main.py\" --server_config=3 --num_workers=8 --img_to_load=-1 --load_previous=0 "
              "--version_name=transmission_albedo_estimator_v1.17 --iteration=22 --batch_size=512 --albedo_checkpt=\"checkpoint/albedo_transfer_v1.05_1.pt\" "
              "--t_min=0.6 --t_max=1.8 --a_min=0.7 --a_max=1.0 ")


if __name__ == "__main__":
    main()