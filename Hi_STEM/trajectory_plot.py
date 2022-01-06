# -*- coding: UTF-8 -*-

from collections import defaultdict
from matplotlib import pyplot as plt


class CheckIn:
    def __init__(self, user_id, time, latitude, longitude, location_id):
        self.user_id = user_id
        self.time = time
        self.latitude = float(latitude)
        self.longitude = float(longitude)
        self.location_id = location_id

    def string(self):
        return " ".join([self.user_id, self.time, self.latitude, self.longitude, self.location_id])


def read_data():

    trajectory_set = defaultdict(list)

    with open("Gowalla_totalCheckins.txt", "r") as f:
        for line in f:
            user_id, time, latitude, longitude, location_id = line.split()
            if user_id == '10':
                break
            check_in = CheckIn(user_id, time, latitude, longitude, location_id)
            # print(check_in.string())
            trajectory_set[user_id].append(check_in)

    return trajectory_set


def trajectory_plot():

    trajectory_set = read_data()
    for user_id in trajectory_set.keys():
        trajectory = trajectory_set[user_id]
        lon, lat = [], []
        for check_in in trajectory:
            lon.append(check_in.longitude)
            lat.append(check_in.latitude)
            # plt.plot(check_in.longitude, check_in.latitude, '*')
            # print(check_in.longitude, check_in.latitude)
        plt.plot(lon, lat, '*')
    plt.show()


if __name__ == "__main__":
    # trajectory_set = read_data()
    # print(trajectory_set["0"])
    trajectory_plot()
