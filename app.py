import streamlit as st
from config.distributions import DISTRIBUTIONS
from utils.statistics import calculate_statistics, fit_distribution
from utils.plotting import plot_distribution, plot_multiple_distributions, plot_parameter_effect
from utils.parameters import get_distribution_params

def main():
    """メイン関数"""
    st.title("統計と分布の可視化アプリ")
    st.write("このアプリでは、様々な分布をインタラクティブに可視化できます。")

    # タブを作成
    tab1, tab2, tab3 = st.tabs(["単一分布", "分布の比較", "パラメータ関係"])

    with tab1:
        # 分布の種類を選択
        distribution_type = st.selectbox("分布の種類を選択してください", list(DISTRIBUTIONS.keys()))
        
        # LaTeX数式の表示
        st.latex(DISTRIBUTIONS[distribution_type]["latex"])
        
        # パラメータの取得と分布の計算
        data, pdf, x, params = get_distribution_params(distribution_type)
        
        # グラフの描画
        plot_distribution(data, pdf, x, distribution_type, DISTRIBUTIONS[distribution_type])
        
        # 統計量の計算と表示
        stats = calculate_statistics(data, distribution_type, params)
        
        # 適合度の評価
        statistic, p_value = fit_distribution(data, distribution_type)
        if statistic is not None:
            st.write(f"適合度検定: 統計量 = {statistic:.4f}, p値 = {p_value:.4f}")
        
        # 分布の説明
        with st.expander("分布の説明"):
            st.write(DISTRIBUTIONS[distribution_type]["description"])

    with tab2:
        # 複数分布の比較
        plot_multiple_distributions()

    with tab3:
        # パラメータと形状の関係の説明
        st.write("### パラメータと分布の形状の関係")
        selected_dist = st.selectbox(
            "分布を選択してください",
            ["正規分布", "ガンマ分布"],
            key="param_effect"
        )
        
        # 基準となるパラメータの設定
        if selected_dist == "正規分布":
            base_params = {"mean": 0, "std_dev": 1}
        elif selected_dist == "ガンマ分布":
            base_params = {"k": 2, "theta": 1}
            
        plot_parameter_effect(selected_dist, base_params)

if __name__ == "__main__":
    main()