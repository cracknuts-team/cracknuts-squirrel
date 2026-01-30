import numpy as np


# BASE_TRACE GENERATE TEST...

def generate_linear_curve(length, key_points):
    x_nodes = np.array(list(key_points.keys()))
    y_nodes = np.array(list(key_points.values()))

    # 全局 x
    x = np.arange(length)

    # 线性插值
    y = np.interp(x, x_nodes, y_nodes)

    return y


# def generate_sub_wave(sub_length, sub_key_points):
#     """
#     生成一个周期小波形（可以插值到 sub_length 长度）
#
#     参数：
#         sub_length (int): 小波形长度
#         sub_key_points (dict): 小波形关键点，相对位置索引和值，例如 {0:0, 5:1, 10:0}
#
#     返回：
#         np.ndarray: 小波形数组
#     """
#     x_nodes = np.array(list(sub_key_points.keys()))
#     y_nodes = np.array(list(sub_key_points.values()))
#     x = np.arange(sub_length)
#     # 用三次样条生成平滑小波形
#     cs = CubicSpline(x_nodes, y_nodes, bc_type='periodic')  # 周期性
#     y = cs(x)
#     return y


def generate_sub_wave(sub_length, sub_key_points):
    """生成小波形（线性插值）"""
    x_nodes = np.array(list(sub_key_points.keys()))
    y_nodes = np.array(list(sub_key_points.values()))
    x = np.arange(sub_length)
    y = np.interp(x, x_nodes, y_nodes)
    return y


def add_repeated_sub_wave(main_curve, sub_wave):
    """
    将小波形重复叠加到主曲线上

    参数：
        main_curve (np.ndarray): 主曲线
        sub_wave (np.ndarray): 单周期小波形
    返回:
        np.ndarray: 叠加后的曲线
    """
    main_length = len(main_curve)
    sub_length = len(sub_wave)

    # 重复小波形，使长度 >= 主曲线长度
    repeats = int(np.ceil(main_length / sub_length))
    repeated_wave = np.tile(sub_wave, repeats)[:main_length]

    return main_curve + repeated_wave


def add_interval_sub_waves(main_curve, interval_sub_configs):
    """
    在主曲线的不同区间叠加不同的小波形

    参数：
        main_curve (np.ndarray): 主曲线
        interval_sub_configs (list of dict): 每个区间配置
            每个 dict 包含：
                'start' (int): 区间起始索引
                'end' (int): 区间结束索引
                'sub_length' (int): 小波形长度
                'sub_key_points' (dict): 小波形关键点
    返回：
        np.ndarray: 叠加后的曲线
    """
    curve = main_curve.copy()
    for cfg in interval_sub_configs:
        start = cfg['start']
        end = cfg['end']
        sub_length = cfg['sub_length']
        sub_key_points = cfg['sub_key_points']

        interval_len = end - start
        sub_wave = generate_sub_wave(sub_length, sub_key_points)

        # 重复小波形覆盖整个区间
        repeats = int(np.ceil(interval_len / sub_length))
        repeated_wave = np.tile(sub_wave, repeats)[:interval_len]

        # 叠加到主曲线
        curve[start:end] += repeated_wave

    return curve


def add_global_noise(curve, noise_std=1.0, seed=None):
    """在曲线上加高斯噪声"""
    if seed is not None:
        np.random.seed(seed)
    noise = np.random.normal(0, noise_std, size=len(curve))
    return curve + noise


def __main__():
    length = 20000
    main_key_points = {0: 0, 500: -10, 1000: 10, 2000: 0, 13000:0, 16000: -20, 17000:0}
    main_curve = generate_linear_curve(length, main_key_points)

    # 小波形：长度 10，关键点 0->1->0
    # sub_wave = generate_sub_wave(20, {0: 0, 5: 10, 10: 0, 15: 100, 20: 0})
    # 区间小波形配置
    interval_sub_configs = [
        {'start': 0, 'end': 13000, 'sub_length': 20, 'sub_key_points': {0: 0, 5: 10, 10: 0, 13: 100, 18:80, 20: 0}},
        {'start': 13000, 'end': 16000, 'sub_length': 20, 'sub_key_points': {0: 10, 5: 13, 10: 10, 15: 80, 20: 10}},
        {'start': 16000, 'end': 20000, 'sub_length': 20, 'sub_key_points': {0: 0, 5: 10, 10: 0, 15: 100, 20: 0}}
    ]

    # curve_with_subs = add_interval_sub_waves(main_curve, interval_sub_configs)
    curve_with_subs = add_interval_sub_waves(main_curve, interval_sub_configs)

    # 叠加
    # curve_with_sub  = add_repeated_sub_wave(main_curve, sub_wave)

    final_curve = add_global_noise(curve_with_subs, noise_std=5.5, seed=42)

    # 可视化
    import matplotlib.pyplot as plt

    plt.plot(final_curve)
    plt.title("Linear Interpolated Curve")
    plt.show()


if __name__ == "__main__":
    __main__()

