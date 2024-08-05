from pathlib import Path
from pymodal_surgical.apps.force_estimator.force_estimator import ForceEstimator
from pymodal_surgical.apps.force_estimator.stats import Statistics
import numpy as np
import cv2
from scipy import signal
import matplotlib.pyplot as plt


experiment_dir = Path("/home/md21/heart_experiments")
sensor_data_dir = Path("experiments")



if __name__ == "__main__":
    
    plot_force = True
    plot_dict = {}
    pixels = None
    K = 16
    axis = 1
    environments = ["phantom", "real"]

    statistics_config = ["cross_correlation", "rmse", "mae", "mape", "cosine"]
    complete_stat_dict = {"phantom": {key: [] for key in statistics_config}, "real": {key: [] for key in statistics_config}}
    
    statistics = Statistics(statistics_config)

    directories = [experiment_dir/"phantom", experiment_dir/"real"]
    sensor_readings_dirs = [sensor_data_dir/"phantom", sensor_data_dir/"real"]


    for type_idx, dir in enumerate(directories):
        mask_dir = dir/"mask.png"
        video_dir = dir/"processed"
        if type_idx == 0:
            mode_shape_video = video_dir/"20240501_042319.mp4"
        else:
            mode_shape_video = video_dir/"20240503_094013.mp4"
        
        mode_shape_config = {
            "video_path": str(mode_shape_video),
            "K": K,
            "start": 0,
            "end": 300,
            "video_type": "mono",
            "batch_size": 2,
            "masking": {
                "enabled": True,
                "mask": str(mask_dir)
            },
            "filtering": {
                "enabled": True,
                "size": 17,
                "sigma": 2.0
            },

            "save_mode_shapes": True
        }

        env = environments[type_idx]
        sensor_readings_files = [file for file in sorted(sensor_readings_dirs[type_idx].rglob("*.npy"))]

        for file_idx, video_file in enumerate(sorted(video_dir.glob("*.mp4"))):

            if type_idx == 1:
                if file_idx == 0:
                    continue
                else:
                    file_idx -= 1

            print(f"Processing video: {video_file}")

            force_video_config = {
                "video_type": "mono",
                "video_path": str(video_file),
                "start": 0,
                "end": 0,
                "pixels": pixels,
            }

            estimator = ForceEstimator(
                mode_shape_config=mode_shape_config,
                force_estimation_config=force_video_config
            )

            fps = estimator.fps

            if force_video_config["end"] == 0 or force_video_config["end"] > len(estimator.force_video_reader):
                force_video_config["end"] = len(estimator.force_video_reader)

            plotting_force = np.zeros((force_video_config["end"] - force_video_config["start"], 2))

            for i in range(force_video_config["start"], force_video_config["end"]):
                idx = i - force_video_config["start"]

                tmp_force = estimator.calculate_force(force_video_config["start"], i, simplify_force=plot_force)
                
                if idx == 0 and K == 4:
                    pixels = estimator.pixels

                if plot_force:
                    plotting_force[idx] = tmp_force / K
                # print(f"Force at frame {i}: {tmp_force}")

            if estimator.save_force:
                estimator.force_video_writer.release()
                cv2.destroyAllWindows()
            
            real_data = np.load(sensor_readings_files[file_idx])
            resampled_real = signal.resample(real_data[:], plotting_force.shape[0])
            normalized_plotting_force = (plotting_force - plotting_force.mean(axis=0)) / plotting_force.std(axis=0)
            resampled_real = (resampled_real - resampled_real.mean(axis=0)) / resampled_real.std(axis=0)

            # plt.plot(-resampled_real[:, 2])
            # plt.plot(normalized_plotting_force[:, 0])
            # plt.legend(["Sensor data", "Prediction"])
            # plt.show()
            stats = statistics(real_data, plotting_force, axis)
            print(stats)

            for key in stats.keys():
                print(f"Saving {key}")
                current_stat = complete_stat_dict[env][key]
                print(f"Current value is {current_stat}")
                print(f"The result for the current stat is {stats[key]}")
                current_stat.append(stats[key])
                complete_stat_dict[env][key] = current_stat
        

        # plt.rcParams.update({'font.size': 22, 'font.family': 'serif', 'font.serif': 'Times New Roman'})
        # fig, axs = plt.subplots(2, 1, figsize=(1920 / dpi, 1080 / dpi), dpi = dpi, sharex=True)
        # fig.supxlabel("Time (s)", fontweight="bold")
        # fig.supylabel("Force", fontweight="bold")
        # time = (1/estimator.fps) * np.arange(force_video_config["start"], force_video_config["end"])
        
    
        print(f"---------------- {env.capitalize()} results --------------------")
        
        for key in complete_stat_dict[env].keys():
            stat = complete_stat_dict[env][key] 
            print(f"* {key.capitalize()}")
            print(f"\tStat: {stat}")
            print(f"\tMean: {np.mean(stat)}")
            print(f"\tStandard deviation: {np.std(stat)}")
            print(f"\tMax: {np.max(stat)}")
            print(f"\tMin: {np.min(stat)}")
