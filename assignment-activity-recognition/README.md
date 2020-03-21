# Activity Recognition 

The data/dataset.csv is downloaded from Kaggle, and has been analyzed and visualized on [Kaggle](https://www.kaggle.com/vmalyi/run-or-walk-data-analysis-and-visualization), considered as a high quality dataset for run-walk activity recognition.

The assignment we need to do is more than that, we need to classify standing, walking or running. And we need to clean the data first, in order to be used for training.

## environment

* pandas
* sklearn
* matplotlib (for visualization)

## read data from the log file and save the corresponding label
In the data collecting stage, we first stand for a while to save the log only for standing; then stop the logging on the app, and start walking while starting to save another log file only for waling; then do the same for running. In this way we have three log files with the ground truth labels. 

In the `data` folder, the `sensorLogr_` indicates it is for running, the `sensorLogs_` indicates it is for standing, the `sensorLogw_` indicates it is for walking.

run `read_data.py` with `sys.argv[1]` as the file path and `sys.arvg[2]` as the activity (s/w/r), as in standing/walking/running.

Then you have three `.csv` file with `time, activity, acc_x, acc_y, acc_z, gyr_x, gyr_y, gyr_z`.

To be mentioned, the log file saved by the sensor fusion app, saves the data in such a way, for every time stamp, it saves either acc or gyro, not both. So in the raw data, for the same time stamp, we cannot have both acc and gyro. So in `read_data.py`, I simply copy the acc/gyro at the last stamp as the acc/gyro for the curent time stamp, since the difference between the two adjunt time stamps are relatively small, I think it is reasonable.

## build the dataset and train with random forrest

Run `activity-recognition.ipynb` cell by cell, allows you build the dataset for standing/walking/running activity recognition, and train with random forest in `sklearn`, with a high prediction accuracy on the test set.

