import numpy as np
import matplotlib.pyplot as plt

class LimitsObjects2D:
    __x_min = 0
    __x_max = 0
    __z_min = 0
    __z_max = 0
    __axis_x_min = 0
    __axis_x_max = 0
    __axis_z_min = 0
    __axis_z_max = 0

    def set_x_min(self, value):
        self.__x_min = value

    def set_x_max(self, value):
        self.__x_max = value

    def set_z_min(self, value):
        self.__z_min = value

    def set_z_max(self, value):
        self.__z_max = value

    def calc_axis(self):
        border_ratio = 1.2
        x_center = (self.__x_max + self.__x_min) / 2
        z_center = (self.__z_max + self.__z_min) / 2
        x_length = (self.__x_max - self.__x_min)
        z_length = (self.__z_max - self.__z_min)
        global_length = np.max([x_length, z_length]) * border_ratio
        self.__axis_x_min = x_center - global_length / 2
        self.__axis_x_max = x_center + global_length / 2
        self.__axis_z_min = z_center - global_length / 2
        self.__axis_z_max = z_center + global_length / 2

    def get_axis_xz(self):
        return [self.__axis_x_min, self.__axis_x_max,
                self.__axis_z_min, self.__axis_z_max]

class LimitsObjects:
    __x_min = 0
    __x_max = 0
    __y_min = 0
    __y_max = 0
    __z_min = 0
    __z_max = 0
    __axis_x_min = 0
    __axis_x_max = 0
    __axis_y_min = 0
    __axis_y_max = 0
    __axis_z_min = 0
    __axis_z_max = 0

    def set_x_min(self, value):
        self.__x_min = value

    def set_x_max(self, value):
        self.__x_max = value

    def set_y_min(self, value):
        self.__y_min = value

    def set_y_max(self, value):
        self.__y_max = value

    def set_z_min(self, value):
        self.__z_min = value

    def set_z_max(self, value):
        self.__z_max = value

    def calc_axis(self):
        border_ratio = 1.2
        x_center = (self.__x_max + self.__x_min) / 2
        y_center = (self.__y_max + self.__y_min) / 2
        z_center = (self.__z_max + self.__z_min) / 2
        x_length = (self.__x_max - self.__x_min)
        y_length = (self.__y_max - self.__y_min)
        z_length = (self.__z_max - self.__z_min)
        global_length = np.max([x_length, y_length, z_length]) * border_ratio
        self.__axis_x_min = x_center - global_length / 2
        self.__axis_x_max = x_center + global_length / 2
        self.__axis_y_min = y_center - global_length / 2
        self.__axis_y_max = y_center + global_length / 2
        self.__axis_z_min = z_center - global_length / 2
        self.__axis_z_max = z_center + global_length / 2

    def get_axis_xz(self):
        return [self.__axis_x_min, self.__axis_x_max,
                self.__axis_z_min, self.__axis_z_max]

    def get_axis_xy(self):
        return [self.__axis_x_min, self.__axis_x_max,
                self.__axis_y_min, self.__axis_y_max]
