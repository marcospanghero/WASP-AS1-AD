import csv
import sys
"""
Here we assume for each log file, it is for one activity
"""

file_path = sys.argv[1] # the path of the log

activity_flag = sys.argv[2]

# standing 0; walking 1; running 2
if activity_flag=="s":
    acti = 0
elif activity_flag=="w":
    acti = 1
elif activity_flag=="r":
    acti = 2
else:
    print("error activity_flag: ", activity_flag)

with open(file_path, newline="") as csvfile: 
    reader = csv.reader(csvfile, delimiter=' ') 
    logg = [] 
    for row in reader: 
        logg.append(row)


last_acc_x, last_acc_y, last_acc_z = 0,0,0
last_gyr_x, last_gyr_y, last_gyr_z = 0,0,0
log_dic = {}
for logg_ in logg:
    time,name,x,y,z = logg_[0].split("\t")
    time_ = int(time)
    x_, y_, z_ = float(x), float(y), float(z)
    tmp_log = {"ACC":[], "GYR":[]}
    if name=="ACC":
        last_acc_x, last_acc_y, last_acc_z = x_,y_,z_
        tmp_log["ACC"].append((x_,y_,z_))
        tmp_log["GYR"].append((last_gyr_x, last_gyr_y, last_gyr_z))
    elif name=="GYR":
        last_gyr_x, last_gyr_y, last_gyr_z = x_,y_,z_

        tmp_log["GYR"].append((x_,y_,z_))
        tmp_log["ACC"].append((last_acc_x, last_acc_y, last_acc_z))
    log_dic[time_] = tmp_log

#print(log_dic)

with open("dataset_"+activity_flag+".csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    writer.writerow(["time","activity", "acceleration_x", "acceleration_y", "acceleration_z", "gyro_x", "gyro_y", "gyro_z"])
    for key,value in log_dic.items():
        writer.writerow([key,acti,value["ACC"][0][0], value["ACC"][0][1], value["ACC"][0][2], value["GYR"][0][0], value["GYR"][0][2], value["GYR"][0][2]])
