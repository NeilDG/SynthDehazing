import os
def main():
    # os.system("python \"albedo_main.py\" --num_workers=12 --img_to_load=-1 --load_previous=1 --iteration=1")
    # os.system("python \"cyclegan_main.py\" --num_workers=12 --img_to_load=-1 --load_previous=1 --iteration=2 --batch_size=512 --num_blocks=4 --net_config=3 --use_bce=1 --likeness_weight=1.0")
    # os.system("python \"cyclegan_main.py\" --num_workers=12 --img_to_load=-1 --load_previous=1 --iteration=2 --batch_size=512 --num_blocks=6 --net_config=1 --use_bce=0 --likeness_weight=1.0")
    #os.system("python \"transmission_main.py\" --num_workers=12 --img_to_load=-1 --load_previous=0 --iteration=6 --batch_size=256 --t_min=0.6 --t_max=1.8 --a_min=0.7 --a_max=1.0 --albedo_checkpt=\"checkpoint/albedo_transfer_v1.05_1.pt\"")
    # os.system("python \"airlight_main.py\" --num_workers=12 --img_to_load=-1 --load_previous=1 --iteration=6 --batch_size=8192 --t_min=0.6 --t_max=1.8 --a_min=0.7 --a_max=1.0 --albedo_checkpt=\"checkpoint/albedo_transfer_v1.05_1.pt\"")
    # os.system("python \"transmission_main.py\" --num_workers=12 --img_to_load=-1 --load_previous=1 --iteration=12 --batch_size=128 --t_min=0.1 --t_max=1.8 --a_min=0.35 --a_max=0.5")
    # os.system("python \"airlight_main.py\" --num_workers=12 --img_to_load=-1 --load_previous=1 --iteration=12 --batch_size=8192 --t_min=0.1 --t_max=1.8 --a_min=0.35 --a_max=0.5")
    # os.system("python \"dehaze_main.py\" --num_workers=12 --img_to_load=-1 --load_previous=1 --iteration=6 --batch_size=256 --t_min=0.6 --t_max=1.8 --a_min=0.7 --a_max=1.0 "
    #           "--version_name=\"dehazer_v2.10\" --iteration=6 "
    #           "--albedo_checkpt=\"checkpoint/albedo_transfer_v1.05_1.pt\"")

    # os.system("python \"benchmarker_ots.py\"")

    # os.system("python \"airlight_main.py\" --num_workers=12 --img_to_load=-1 --load_previous=1 "
    #           "--version_name=airlight_estimator_v1.16 --iteration=7 --batch_size=8192 --albedo_checkpt=\"checkpoint/albedo_transfer_v1.05_1.pt\" "
    #           "--penalty_weight=10.0 --t_min=0.6 --t_max=1.8 --a_min=0.7 --a_max=1.0")

    # os.system("python \"airlight_main.py\" --num_workers=12 --img_to_load=-1 --load_previous=0 "
    #           "--version_name=airlight_estimator_v1.16 --iteration=8 --batch_size=8192 --albedo_checkpt=\"checkpoint/albedo_transfer_v1.05_1.pt\" "
    #           "--penalty_weight=50.0 --t_min=0.6 --t_max=1.8 --a_min=0.7 --a_max=1.0")
    #
    # os.system("python \"airlight_main.py\" --num_workers=12 --img_to_load=-1 --load_previous=0 "
    #           "--version_name=airlight_estimator_v1.16 --iteration=9 --batch_size=8192 --albedo_checkpt=\"checkpoint/albedo_transfer_v1.05_1.pt\" "
    #           "--penalty_weight=100.0 --t_min=0.6 --t_max=1.8 --a_min=0.7 --a_max=1.0")

    # os.system("python \"transmission_main.py\" --num_workers=12 --server_config=4 --img_to_load=-1 --load_previous=1 "
    #           "--version_name=transmission_albedo_estimator_v1.22 --iteration=13 --unlit_enabled=1 --patch_size=32 --batch_size=256 --albedo_checkpt=\"checkpoint/albedo_transfer_v1.05_1.pt\" "
    #           "--t_min=0.6 --t_max=1.8 --a_min=0.7 --a_max=1.0 ")

    # os.system("python \"transmission_main.py\" --num_workers=12 --server_config=4 --img_to_load=-1 --load_previous=1 "
    #           "--version_name=transmission_albedo_estimator_v1.22 --iteration=14 --unlit_enabled=1 --patch_size=32 --batch_size=256 --albedo_checkpt=\"checkpoint/albedo_transfer_v1.05_1.pt\" "
    #           "--t_min=0.6 --t_max=1.8 --a_min=0.7 --a_max=1.0 ")
    #
    # os.system("python \"transmission_main.py\" --num_workers=12 --server_config=4 --img_to_load=-1 --load_previous=0 "
    #           "--version_name=transmission_albedo_estimator_v1.22 --iteration=15 --unlit_enabled=1 --patch_size=32 --batch_size=256 --albedo_checkpt=\"checkpoint/albedo_transfer_v1.05_1.pt\" "
    #           "--t_min=0.6 --t_max=1.8 --a_min=0.7 --a_max=1.0 ")

    # os.system("python \"transmission_main.py\" --num_workers=12 --server_config=4 --img_to_load=-1 --load_previous=1 "
    #           "--version_name=transmission_albedo_estimator_v1.22 --iteration=16 --unlit_enabled=1 --patch_size=32 --batch_size=256 --albedo_checkpt=\"checkpoint/albedo_transfer_v1.05_1.pt\" "
    #           "--t_min=0.6 --t_max=1.8 --a_min=0.7 --a_max=1.0 ")

    os.system("python \"transmission_main.py\" --num_workers=12 --server_config=4 --img_to_load=-1 --load_previous=1 "
              "--version_name=transmission_albedo_estimator_v1.22 --iteration=17 --unlit_enabled=1 --patch_size=32 --batch_size=256 --albedo_checkpt=\"checkpoint/albedo_transfer_v1.05_1.pt\" "
              "--t_min=0.6 --t_max=1.8 --a_min=0.7 --a_max=1.0 ")
    #
    # os.system("shutdown /s /t 1")

if __name__ == "__main__":
    main()