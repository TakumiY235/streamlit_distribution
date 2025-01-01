import streamlit as st
import numpy as np
from scipy.stats import norm, uniform, expon, binom, poisson, beta, t, f, chi2, gamma, weibull_min, geom, nbinom

def get_distribution_params(dist_type):
    """分布のパラメータを取得する関数"""
    num_points = 1000
    x = np.linspace(-5, 10, num_points)
    params = {}
    
    # パラメータの説明を表示するヘルパー関数
    def show_param_description(param_name, description):
        with st.expander(f"{param_name}の説明"):
            st.write(description)
    
    if dist_type == "正規分布":
        show_param_description("平均 (μ)", "分布の中心位置を決定します。データの重心となる値です。")
        params["mean"] = st.slider("平均 (μ)", -5.0, 5.0, 0.0, 0.1, 
            help="分布の中心位置を調整します")
            
        show_param_description("標準偏差 (σ)", "分布の広がりを制御します。大きいほど分布が広がります。")
        params["std_dev"] = st.slider("標準偏差 (σ)", 0.1, 5.0, 1.0, 0.1,
            help="分布の広がりを調整します")
        
        data = np.random.normal(params["mean"], params["std_dev"], size=num_points)
        pdf = norm.pdf(x, params["mean"], params["std_dev"])
        
        # 統計量の追加表示
        st.write(f"68%のデータが {params['mean']-params['std_dev']:.2f} から {params['mean']+params['std_dev']:.2f} の範囲に")
        st.write(f"95%のデータが {params['mean']-2*params['std_dev']:.2f} から {params['mean']+2*params['std_dev']:.2f} の範囲に")
        
    elif dist_type == "一様分布":
        show_param_description("最小値 (a)", "分布の下限値を設定します。この値未満の確率は0になります。")
        params["low"] = st.slider("最小値 (a)", -5.0, 5.0, 0.0, 0.1,
            help="分布の下限を設定")
            
        show_param_description("最大値 (b)", "分布の上限値を設定します。この値を超える確率は0になります。")
        params["high"] = st.slider("最大値 (b)", params["low"], 5.0, 1.0, 0.1,
            help="分布の上限を設定")
            
        if params["high"] <= params["low"]:
            st.error("最大値は最小値より大きい値を設定してください")
            return None, None, None, params
            
        data = np.random.uniform(params["low"], params["high"], size=num_points)
        pdf = uniform.pdf(x, params["low"], params["high"] - params["low"])
        
        # 統計量の追加表示
        st.write(f"平均値: {(params['low'] + params['high'])/2:.2f}")
        st.write(f"分散: {((params['high'] - params['low'])**2)/12:.2f}")
        
    elif dist_type == "指数分布":
        show_param_description("尺度 (λ)", "分布の減衰率を制御します。大きいほど早く減衰します。")
        params["scale"] = st.slider("尺度 (λ)", 0.1, 5.0, 1.0, 0.1,
            help="分布の減衰率を調整")
            
        data = np.random.exponential(params["scale"], size=num_points)
        pdf = expon.pdf(x, scale=params["scale"])
        
        # 統計量の追加表示
        st.write(f"平均値: {params['scale']:.2f}")
        st.write(f"分散: {params['scale']**2:.2f}")
        
    elif dist_type == "ベータ分布":
        show_param_description("α", "分布の左側の形状を制御します。大きいほど左側が急峻になります。")
        params["alpha"] = st.slider("α", 0.1, 10.0, 2.0, 0.1,
            help="左側の形状を調整")
            
        show_param_description("β", "分布の右側の形状を制御します。大きいほど右側が急峻になります。")
        params["beta"] = st.slider("β", 0.1, 10.0, 2.0, 0.1,
            help="右側の形状を調整")
            
        data = beta.rvs(params["alpha"], params["beta"], size=num_points)
        pdf = beta.pdf(x, params["alpha"], params["beta"])
        
        mean = params["alpha"] / (params["alpha"] + params["beta"])
        var = (params["alpha"] * params["beta"]) / ((params["alpha"] + params["beta"])**2 * (params["alpha"] + params["beta"] + 1))
        st.write(f"平均値: {mean:.2f}")
        st.write(f"分散: {var:.2f}")

    elif dist_type == "二項分布":
        show_param_description("試行回数 (n)", "実験や観測を行う総回数です。正の整数値を指定します。")
        params["n"] = st.slider("試行回数 (n)", 1, 100, 10, 1,
            help="実験の総試行回数を設定")
            
        show_param_description("成功確率 (p)", "各試行での成功確率です。0から1の値を指定します。")
        params["p"] = st.slider("成功確率 (p)", 0.0, 1.0, 0.5, 0.01,
            help="各試行での成功確率を設定")
            
        data = binom.rvs(params["n"], params["p"], size=num_points)
        k = np.arange(0, params["n"] + 1)
        pdf = binom.pmf(k, params["n"], params["p"])
        
        mean = params["n"] * params["p"]
        var = params["n"] * params["p"] * (1 - params["p"])
        st.write(f"平均値: {mean:.2f}")
        st.write(f"分散: {var:.2f}")
        
    elif dist_type == "ポアソン分布":
        show_param_description("平均 (λ)", "単位時間あたりの平均発生回数です。正の実数値を指定します。")
        params["mu"] = st.slider("平均 (λ)", 0.1, 10.0, 3.0, 0.1,
            help="平均発生回数を設定")
            
        data = poisson.rvs(params["mu"], size=num_points)
        pdf = poisson.pmf(np.arange(0, 21), params["mu"])
        
        st.write(f"平均値: {params['mu']:.2f}")
        st.write(f"分散: {params['mu']:.2f}")
        
    elif dist_type == "t分布":
        show_param_description("自由度 (ν)", "分布の形状を制御するパラメータです。大きいほど正規分布に近づきます。")
        params["df"] = st.slider("自由度 (ν)", 1, 100, 10, 1,
            help="分布の形状を調整")
            
        data = t.rvs(params["df"], size=num_points)
        pdf = t.pdf(x, params["df"])
        
        if params["df"] > 1:
            st.write("平均値: 0")
            if params["df"] > 2:
                variance = params["df"] / (params["df"] - 2)
                st.write(f"分散: {variance:.2f}")
            else:
                st.write("分散: 未定義")
        else:
            st.write("平均値と分散は未定義")

    elif dist_type == "F分布":
        show_param_description("分子の自由度 (d₁)", "分布の形状を制御する1つ目のパラメータです。")
        params["dfn"] = st.slider("分子の自由度 (d₁)", 1, 100, 10, 1,
            help="分布の形状を調整")
            
        show_param_description("分母の自由度 (d₂)", "分布の形状を制御する2つ目のパラメータです。")
        params["dfd"] = st.slider("分母の自由度 (d₂)", 1, 100, 10, 1,
            help="分布の裾を調整")
            
        data = f.rvs(params["dfn"], params["dfd"], size=num_points)
        pdf = f.pdf(x, params["dfn"], params["dfd"])
        
        if params["dfd"] > 2:
            mean = params["dfd"] / (params["dfd"] - 2)
            if params["dfd"] > 4:
                variance = (2 * params["dfd"]**2 * (params["dfn"] + params["dfd"] - 2)) / \
                          (params["dfn"] * (params["dfd"] - 2)**2 * (params["dfd"] - 4))
                st.write(f"分散: {variance:.2f}")
            else:
                st.write("分散: 未定義")
            st.write(f"平均値: {mean:.2f}")
        else:
            st.write("平均値と分散は未定義")

    elif dist_type == "カイ二乗分布":
        show_param_description("自由度 (k)", "分布の形状を制御するパラメータです。")
        params["df"] = st.slider("自由度 (k)", 1, 100, 10, 1,
            help="分布の形状を調整")
            
        data = chi2.rvs(params["df"], size=num_points)
        pdf = chi2.pdf(x, params["df"])
        
        st.write(f"平均値: {params['df']:.2f}")
        st.write(f"分散: {2 * params['df']:.2f}")

    elif dist_type == "ガンマ分布":
        show_param_description("形状母数 (k)", "分布の形状を制御します。")
        params["k"] = st.slider("形状母数 (k)", 0.1, 10.0, 1.0, 0.1,
            help="分布の形状を調整")
            
        show_param_description("尺度母数 (θ)", "分布の広がりを制御します。")
        params["theta"] = st.slider("尺度母数 (θ)", 0.1, 10.0, 1.0, 0.1,
            help="分布の広がりを調整")
            
        data = gamma.rvs(a=params["k"], scale=params["theta"], size=num_points)
        pdf = gamma.pdf(x, a=params["k"], scale=params["theta"])
        
        mean = params["k"] * params["theta"]
        var = params["k"] * params["theta"]**2
        st.write(f"平均値: {mean:.2f}")
        st.write(f"分散: {var:.2f}")

    elif dist_type == "ワイブル分布":
        show_param_description("形状母数 (k)", "分布の形状を制御します。k=1で指数分布になります。")
        params["c"] = st.slider("形状母数 (k)", 0.1, 10.0, 1.0, 0.1,
            help="分布の形状を調整")
            
        show_param_description("尺度母数 (λ)", "分布の広がりを制御します。")
        params["scale"] = st.slider("尺度母数 (λ)", 0.1, 10.0, 1.0, 0.1,
            help="分布の広がりを調整")
            
        data = weibull_min.rvs(params["c"], scale=params["scale"], size=num_points)
        pdf = weibull_min.pdf(x, params["c"], scale=params["scale"])
        
        from scipy.special import gamma as gamma_func
        mean = params["scale"] * gamma_func(1 + 1/params["c"])
        var = params["scale"]**2 * (gamma_func(1 + 2/params["c"]) - (gamma_func(1 + 1/params["c"]))**2)
        st.write(f"平均値: {mean:.2f}")
        st.write(f"分散: {var:.2f}")

    elif dist_type == "幾何分布":
        show_param_description("成功確率 (p)", "各試行での成功確率です。")
        params["p"] = st.slider("成功確率 (p)", 0.01, 1.0, 0.5, 0.01,
            help="成功確率を設定")
            
        data = geom.rvs(params["p"], size=num_points)
        pdf = geom.pmf(np.arange(1, 11), params["p"])
        
        mean = 1 / params["p"]
        var = (1 - params["p"]) / (params["p"]**2)
        st.write(f"平均値: {mean:.2f}")
        st.write(f"分散: {var:.2f}")

    elif dist_type == "負の二項分布":
        show_param_description("成功回数 (r)", "目標とする成功回数です。")
        params["n"] = st.slider("成功回数 (r)", 1, 100, 10, 1,
            help="目標成功回数を設定")
            
        show_param_description("成功確率 (p)", "各試行での成功確率です。")
        params["p"] = st.slider("成功確率 (p)", 0.01, 1.0, 0.5, 0.01,
            help="成功確率を設定")
            
        data = nbinom.rvs(params["n"], params["p"], size=num_points)
        pdf = nbinom.pmf(np.arange(0, 21), params["n"], params["p"])
        
        mean = params["n"] * (1 - params["p"]) / params["p"]
        var = params["n"] * (1 - params["p"]) / (params["p"]**2)
        st.write(f"平均値: {mean:.2f}")
        st.write(f"分散: {var:.2f}")

    elif dist_type == "多項分布":
        # カテゴリ数の設定
        show_param_description("カテゴリ数", "分類するカテゴリの数です（2から6まで設定可能）。")
        num_categories = st.slider("カテゴリ数", 2, 6, 3, 1,
            help="分類するカテゴリの数を設定")
            
        show_param_description("試行回数 (n)", "実験や観測を行う総回数です。")
        params["n"] = st.slider("試行回数 (n)", 1, 1000, 100, 1,
            help="実験の総試行回数を設定")
            
        # 各カテゴリの確率を設定
        probs = []
        col1, col2 = st.columns(2)
        with col1:
            st.write("各カテゴリの確率")
            st.write("（最後のカテゴリは自動的に計算されます）")
            remaining_prob = 1.0
            for i in range(num_categories-1):
                max_prob = min(1.0, remaining_prob)
                p = st.slider(f"カテゴリ{i+1}の確率", 0.0, max_prob, max_prob/2, 0.01,
                    key=f"prob_{i}")
                probs.append(p)
                remaining_prob -= p
            probs.append(remaining_prob)
        
        with col2:
            st.write("確率の合計: 1.0")
            for i, p in enumerate(probs):
                st.write(f"カテゴリ{i+1}: {p:.2f}")
            
        # データ生成
        params["probs"] = probs
        data = np.random.multinomial(params["n"], probs, size=num_points)
        pdf = np.mean(data, axis=0) / params["n"]
        
        # 期待値の表示
        st.write("### 期待値（理論値）")
        for i, p in enumerate(probs):
            expected = params["n"] * p
            st.write(f"カテゴリ{i+1}の期待出現回数: {expected:.1f}")
        
        # 分散共分散の表示
        st.write("### 分散共分散")
        for i in range(num_categories):
            if i == num_categories - 1:
                continue  # 最後のカテゴリは他から決定されるので省略
            var = params["n"] * probs[i] * (1 - probs[i])
            st.write(f"カテゴリ{i+1}の分散: {var:.1f}")
            
        # パラメータの保存
        params["num_categories"] = num_categories
        
        return data, pdf, np.arange(num_categories), params
    
    else:
        st.error(f"未実装の分布タイプです: {dist_type}")
        return None, None, x, params


    return data, pdf, x, params

    