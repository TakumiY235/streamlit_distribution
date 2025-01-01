import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, uniform, expon, gamma

def plot_distribution(data, pdf, x, dist_type, dist_info):
    """分布をプロットする関数"""

    plt.rcParams['font.family'] = ['MS Gothic', 'Yu Gothic']

    graph1, graph2 = st.columns(2)
    
    # サンプル分布のプロット
    graph1.write("標本分布")
    df_data = pd.DataFrame(data, columns=["value"])
    graph1.line_chart(df_data)
    
    # 確率密度関数のプロット
    graph2.write("確率密度関数")
    if pdf is not None:
        if dist_info["type"] == "discrete":
            x_discrete = np.arange(len(pdf))
            df_pdf = pd.DataFrame({'pdf': pdf}, index=x_discrete)
        elif dist_info["type"] == "non_negative":
            pdf_filtered = pdf[x >= 0]
            x_filtered = x[x >= 0]
            df_pdf = pd.DataFrame({'pdf': pdf_filtered}, index=x_filtered)
        else:
            df_pdf = pd.DataFrame({'pdf': pdf}, index=x)
        graph2.line_chart(df_pdf)

    if dist_info["type"] == "discrete" and dist_type == "多項分布":
        # 多項分布用の棒グラフ表示
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # サンプルデータの表示
        sample_means = np.mean(data, axis=0)
        ax1.bar(range(len(sample_means)), sample_means)
        ax1.set_title("サンプル分布")
        ax1.set_xlabel("カテゴリ")
        ax1.set_ylabel("頻度")
        
        # 理論確率の表示
        ax2.bar(range(len(pdf)), pdf)
        ax2.set_title("理論確率")
        ax2.set_xlabel("カテゴリ")
        ax2.set_ylabel("確率")
        
        st.pyplot(fig)

def plot_parameter_effect(dist_type, base_params):
    """パラメータの影響を示すグラフを表示する関数"""
    st.write("### パラメータの影響")
    
    x = np.linspace(-5, 10, 1000)
    fig, ax = plt.subplots(figsize=(10, 6))

    if dist_type == "正規分布":
        # 平均の影響
        means = [-2, 0, 2]
        for mu in means:
            pdf = norm.pdf(x, mu, base_params["std_dev"])
            ax.plot(x, pdf, label=f'μ={mu}')
        plt.title("平均値(μ)の影響")
        
        # 標準偏差の影響を別のグラフで
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        stds = [0.5, 1, 2]
        for sigma in stds:
            pdf = norm.pdf(x, base_params["mean"], sigma)
            ax2.plot(x, pdf, label=f'σ={sigma}')
        plt.title("標準偏差(σ)の影響")
        
    elif dist_type == "ガンマ分布":
        # 形状母数の影響
        ks = [0.5, 1, 2, 5]
        for k in ks:
            pdf = gamma.pdf(x, a=k, scale=base_params["theta"])
            ax.plot(x, pdf, label=f'k={k}')
        plt.title("形状母数(k)の影響")
        
    # 共通の設定
    ax.set_xlabel('x')
    ax.set_ylabel('確率密度')
    ax.legend()
    ax.grid(True)
    
    # グラフの表示
    st.pyplot(fig)
    if dist_type == "正規分布":
        st.pyplot(fig2)

def plot_multiple_distributions():
    """複数の分布を重ね合わせて表示する関数"""
    st.write("### 分布の比較")
    
    # 比較したい分布を選択
    selected_distributions = st.multiselect(
        "比較する分布を選択してください",
        ["正規分布", "一様分布", "指数分布", "ガンマ分布"],
        default=["正規分布"]
    )
    
    if selected_distributions:
        x = np.linspace(-5, 10, 1000)
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for dist in selected_distributions:
            if dist == "正規分布":
                pdf = norm.pdf(x, 0, 1)
                ax.plot(x, pdf, label='正規分布(μ=0, σ=1)')
            elif dist == "一様分布":
                pdf = uniform.pdf(x, -1, 2)
                ax.plot(x, pdf, label='一様分布(a=-1, b=1)')
            elif dist == "指数分布":
                pdf = expon.pdf(x, scale=1)
                ax.plot(x, pdf, label='指数分布(λ=1)')
            elif dist == "ガンマ分布":
                pdf = gamma.pdf(x, a=2, scale=1)
                ax.plot(x, pdf, label='ガンマ分布(k=2, θ=1)')
        
        ax.set_xlabel('x')
        ax.set_ylabel('確率密度')
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)
