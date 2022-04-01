import os

def main():
    # os.system("python \"transmission_main.py\" --server_config=3 --num_workers=8 --img_to_load=-1 --load_previous=1 "
    #           "--version_name=transmission_albedo_estimator_v1.17 --iteration=13 --patch_size=64 --batch_size=128 --albedo_checkpt=\"checkpoint/albedo_transfer_v1.05_1.pt\" "
    #           "--t_min=0.6 --t_max=1.8 --a_min=0.7 --a_max=1.0 ")
    #
    # os.system("python \"transmission_main.py\" --server_config=3 --num_workers=8 --img_to_load=-1 --load_previous=1 "
    #           "--version_name=transmission_albedo_estimator_v1.17 --iteration=14 --patch_size=64 --batch_size=128 --albedo_checkpt=\"checkpoint/albedo_transfer_v1.05_1.pt\" "
    #           "--t_min=0.6 --t_max=1.8 --a_min=0.7 --a_max=1.0 ")

    # os.system("python \"transmission_main.py\" --server_config=3 --num_workers=8 --img_to_load=-1 --load_previous=0 "
    #           "--version_name=transmission_albedo_estimator_v1.17 --iteration=15 --patch_size=64 --batch_size=128 --albedo_checkpt=\"checkpoint/albedo_transfer_v1.05_1.pt\" "
    #           "--t_min=0.6 --t_max=1.8 --a_min=0.7 --a_max=1.0 ")

    # os.system("python \"transmission_main.py\" --server_config=3 --num_workers=8 --img_to_load=-1 --load_previous=0 "
    #           "--version_name=transmission_albedo_estimator_v1.17 --iteration=16 --patch_size=64 --batch_size=128 --albedo_checkpt=\"checkpoint/albedo_transfer_v1.05_1.pt\" "
    #           "--t_min=0.6 --t_max=1.8 --a_min=0.7 --a_max=1.0 ")
    #
    # os.system("python \"transmission_main.py\" --server_config=3 --num_workers=8 --img_to_load=-1 --load_previous=0 "
    #           "--version_name=transmission_albedo_estimator_v1.17 --iteration=17 --patch_size=64 --batch_size=128 --albedo_checkpt=\"checkpoint/albedo_transfer_v1.05_1.pt\" "
    #           "--t_min=0.6 --t_max=1.8 --a_min=0.7 --a_max=1.0 ")
    #
    # os.system("python \"transmission_main.py\" --server_config=3 --num_workers=8 --img_to_load=-1 --load_previous=1 "
    #           "--version_name=transmission_albedo_estimator_v1.17 --iteration=18 --patch_size=64 --batch_size=128 --albedo_checkpt=\"checkpoint/albedo_transfer_v1.05_1.pt\" "
    #           "--t_min=0.6 --t_max=1.8 --a_min=0.7 --a_max=1.0 ")
    #
    # os.system("python \"transmission_main.py\" --server_config=3 --num_workers=8 --img_to_load=-1 --load_previous=0 "
    #           "--version_name=transmission_albedo_estimator_v1.17 --iteration=19 --patch_size=64 --batch_size=128 --albedo_checkpt=\"checkpoint/albedo_transfer_v1.05_1.pt\" "
    #           "--t_min=0.6 --t_max=1.8 --a_min=0.7 --a_max=1.0 ")
    #
    # os.system("python \"transmission_main.py\" --server_config=3 --num_workers=8 --img_to_load=-1 --load_previous=0 "
    #           "--version_name=transmission_albedo_estimator_v1.17 --iteration=20 --patch_size=64 --batch_size=128 --albedo_checkpt=\"checkpoint/albedo_transfer_v1.05_1.pt\" "
    #           "--t_min=0.6 --t_max=1.8 --a_min=0.7 --a_max=1.0 ")

    # os.system("python \"transmission_main.py\" --server_config=3 --num_workers=8 --img_to_load=-1 --load_previous=0 "
    #           "--version_name=transmission_albedo_estimator_v1.17 --iteration=21 --patch_size=64 --batch_size=128 --albedo_checkpt=\"checkpoint/albedo_transfer_v1.05_1.pt\" "
    #           "--t_min=0.6 --t_max=1.8 --a_min=0.7 --a_max=1.0 ")
    #
    # os.system("python \"transmission_main.py\" --server_config=3 --num_workers=8 --img_to_load=-1 --load_previous=0 "
    #           "--version_name=transmission_albedo_estimator_v1.17 --iteration=22 --patch_size=64 --batch_size=128 --albedo_checkpt=\"checkpoint/albedo_transfer_v1.05_1.pt\" "
    #           "--t_min=0.6 --t_max=1.8 --a_min=0.7 --a_max=1.0 ")
    #
    # os.system("python \"airlight_main.py\" --server_config=3 --num_workers=8 --img_to_load=-1 --load_previous=1 "
    #           "--version_name=airlight_estimator_v1.16 --iteration=7 --batch_size=8192 --albedo_checkpt=\"checkpoint/albedo_transfer_v1.05_1.pt\" "
    #           "--penalty_weight=10.0 --t_min=0.6 --t_max=1.8 --a_min=0.7 --a_max=1.0")

    os.system("python \"airlight_main.py\" --server_config=3 --num_workers=8 --img_to_load=-1 --load_previous=1 "
              "--version_name=airlight_estimator_v1.16 --iteration=8 --batch_size=8192 --albedo_checkpt=\"checkpoint/albedo_transfer_v1.05_1.pt\" "
              "--penalty_weight=50.0 --t_min=0.6 --t_max=1.8 --a_min=0.7 --a_max=1.0")

    os.system("python \"airlight_main.py\" --server_config=3 --num_workers=8 --img_to_load=-1 --load_previous=0 "
              "--version_name=airlight_estimator_v1.16 --iteration=9 --batch_size=8192 --albedo_checkpt=\"checkpoint/albedo_transfer_v1.05_1.pt\" "
              "--penalty_weight=100.0 --t_min=0.6 --t_max=1.8 --a_min=0.7 --a_max=1.0")

    os.system("python \"airlight_main.py\" --server_config=3 --num_workers=8 --img_to_load=-1 --load_previous=0 "
              "--version_name=airlight_estimator_v1.16 --iteration=10 --batch_size=8192 --albedo_checkpt=\"checkpoint/albedo_transfer_v1.05_1.pt\" "
              "--penalty_weight=200.0 --t_min=0.6 --t_max=1.8 --a_min=0.7 --a_max=1.0")

    os.system("python \"airlight_main.py\" --server_config=3 --num_workers=8 --img_to_load=-1 --load_previous=0 "
              "--version_name=airlight_estimator_v1.16 --iteration=11 --batch_size=8192 --albedo_checkpt=\"checkpoint/albedo_transfer_v1.05_1.pt\" "
              "--penalty_weight=500.0 --t_min=0.6 --t_max=1.8 --a_min=0.7 --a_max=1.0")


if __name__ == "__main__":
    main()