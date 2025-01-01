import numpy as np
from scipy import stats as scipy_stats

def calculate_statistics(data, dist_type, params):
    """統計量を計算する関数"""
    stats_dict = {}
    
    # 基本統計量の計算
    stats_dict["平均"] = np.mean(data)
    stats_dict["分散"] = np.var(data)
    stats_dict["標準偏差"] = np.std(data)
    stats_dict["歪度"] = scipy_stats.skew(data)
    stats_dict["尖度"] = scipy_stats.kurtosis(data)
    
    # 理論値の計算（主要な分布について）
    if dist_type == "正規分布":
        stats_dict["理論平均"] = params["mean"]
        stats_dict["理論分散"] = params["std_dev"]**2
        stats_dict["理論歪度"] = 0
        stats_dict["理論尖度"] = 0
    elif dist_type == "一様分布":
        a, b = params["low"], params["high"]
        stats_dict["理論平均"] = (a + b) / 2
        stats_dict["理論分散"] = (b - a)**2 / 12
        stats_dict["理論歪度"] = 0
        stats_dict["理論尖度"] = -6/5

    return stats_dict

def fit_distribution(data, dist_type):
    """分布の適合度を評価する関数"""
    if dist_type == "正規分布":
        statistic, p_value = scipy_stats.normaltest(data)
    elif dist_type == "一様分布":
        statistic, p_value = scipy_stats.kstest(data, 'uniform')
    else:
        return None, None
    
    return statistic, p_value