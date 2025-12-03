# Copyright 2024 CrackNuts. All rights reserved.

import numpy as np
import zarr
from preprocessing_basic import PPBasic
import matplotlib.pyplot as plt

class stat(PPBasic):
    """
    计算均值、方差等的类
    """

    def __init__(self, input_path=None, output_path=None, sample_range=(0, None), **kwargs):
        super().__init__(input_path=input_path, output_path=output_path, **kwargs)
        self.sample_range = sample_range

    def calc_mean(self):
        mean = self.t[:, self.sample_range[0]:self.sample_range[1]].mean(axis=0)
        mean = mean.compute()
        return mean

    def calc_var(self):
        var = self.t[:, self.sample_range[0]:self.sample_range[1]].var(axis=0)
        var = var.compute()
        return var

if __name__ == "__main__":
    # 示例用法
    calc = stat(input_path='E:\\codes\\Acquisition\\dataset\\HD.zarr')
    calc.auto_out_filename()
    mean = calc.calc_mean()
    plt.plot(mean)
    plt.show()
