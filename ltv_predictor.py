"""
LTV 预测模块 - LTV Predictor Module
基于 BG/NBD 和 Gamma-Gamma 模型预测客户生命周期价值
"""

from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy.special import gammaln, beta
from scipy.optimize import minimize
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings

warnings.filterwarnings('ignore')


@dataclass
class BGNNBDResult:
    """BG/NBD 模型结果"""
    params: Dict[str, float]
    rmse: float
    aic: float
    convergence: bool
    predictions: Optional[pd.DataFrame]


@dataclass
class GammaGammaResult:
    """Gamma-Gamma 模型结果"""
    params: Dict[str, float]
    r_square: float
    predictions: Optional[pd.Series]


@dataclass
class LTVResult:
    """LTV 预测结果"""
    df: pd.DataFrame
    bg_nbd_result: BGNNBDResult
    gamma_gamma_result: GammaGammaResult
    ltv_predictions: pd.DataFrame
    summary: Dict[str, Any]


class BGNNBDModel:
    """
    BG/NBD 模型 (Beta-Geometric/Negative Binomial Distribution)
    用于预测客户购买行为
    """

    def __init__(self, penalize_coef: float = 0):
        """
        初始化模型

        Args:
            penalize_coef: 正则化系数
        """
        self.penalize_coef = penalize_coef
        self.params_ = {}
        self._converged = False

    def fit(
        self,
        frequency: np.ndarray,
        recency: np.ndarray,
        T: np.ndarray,
        n_iter: int = 100
    ) -> 'BGNNBDModel':
        """
        拟合模型

        Args:
            frequency: 购买频率 (x)
            recency: 最近一次购买时间 (tx)
            T: 观察期长度
            n_iter: 最大迭代次数
        """
        # 初始参数猜测
        initial_params = [1.0, 1.0, 1.0, 1.0]

        # 优化
        result = minimize(
            self._negative_log_likelihood,
            initial_params,
            args=(frequency, recency, T),
            method='L-BFGS-B',
            options={'maxiter': n_iter, 'disp': False}
        )

        self.params_ = {
            'r': result.x[0],
            'alpha': result.x[1],
            'a': result.x[2],
            'b': result.x[3]
        }
        self._converged = result.success

        return self

    def _negative_log_likelihood(
        self,
        params: np.ndarray,
        frequency: np.ndarray,
        recency: np.ndarray,
        T: np.ndarray
    ) -> float:
        """计算负对数似然"""
        r, alpha, a, b = params

        # 参数约束
        if any(p <= 0 for p in [r, alpha, a, b]):
            return 1e15

        # 对数似然计算
        ln_P = (
            gammaln(r + frequency) - gammaln(r) +
            r * np.log(alpha) - (r + frequency) * np.log(alpha + T) +
            np.log(a / (a + b + frequency - 1)) +
            gammaln(a + b) + gammaln(b + frequency) -
            gammaln(a) - gammaln(b) -
            gammaln(a + b + frequency) +
            (a + b + frequency) * np.log(a + b + frequency - 1) -
            (b + frequency) * np.log(a + b + frequency - 1)
        )

        # 处理特殊情况
        ln_P = np.nan_to_num(ln_P, nan=-1e10)

        nll = -np.sum(ln_P)

        # 正则化
        if self.penalize_coef > 0:
            nll += self.penalize_coef * np.sum(np.array(params) ** 2)

        return nll

    def expected_number_purchases(
        self,
        t: float,
        frequency: np.ndarray,
        recency: np.ndarray,
        T: np.ndarray
    ) -> np.ndarray:
        """
        预测未来 t 期内的期望购买次数

        Args:
            t: 预测期长度
            frequency: 历史购买频率
            recency: 最近一次购买时间
            T: 观察期长度

        Returns:
            np.ndarray: 期望购买次数
        """
        r = self.params_['r']
        alpha = self.params_['alpha']
        a = self.params_['a']
        b = self.params_['b']

        # P(alive) 概率
        p_alive = (
            1 / (1 + np.exp(
                np.log(a / b) +
                (r + frequency) * np.log(alpha + T) -
                r * np.log(alpha + T + t) -
                (r + frequency) * np.log(alpha + recency)
            ))
        )

        # 期望购买次数
        expected = (
            p_alive *
            (a + b + frequency - 1) / (a - 1) *
            (1 - np.exp(
                -(r + frequency) * np.log(1 + t / (alpha + T))
            ))
        )

        return np.nan_to_num(expected, nan=0)

    def predict(
        self,
        df: pd.DataFrame,
        t: int = 12,
        freq_col: str = 'frequency',
        recency_col: str = 'recency',
        T_col: str = 'T'
    ) -> pd.DataFrame:
        """
        预测客户购买行为

        Args:
            df: 客户数据
            t: 预测期 (月)
            freq_col: 频率列名
            recency_col: 最近购买列名
            T_col: 观察期列名

        Returns:
            pd.DataFrame: 预测结果
        """
        frequency = df[freq_col].values
        recency = df[recency_col].values
        T = df[T_col].values

        expected_purchases = self.expected_number_purchases(
            t / 12, frequency, recency, T
        )

        predictions = df.copy()
        predictions[f'expected_purchases_{t}m'] = expected_purchases

        return predictions


class GammaGammaModel:
    """
    Gamma-Gamma 模型
    用于预测客户交易价值
    """

    def __init__(self):
        self.params_ = {}
        self._r_square = 0

    def fit(
        self,
        frequency: np.ndarray,
        monetary_value: np.ndarray
    ) -> 'GammaGammaModel':
        """
        拟合模型

        Args:
            frequency: 购买频率
            monetary_value: 平均交易价值
        """
        # 过滤有效数据
        valid = (frequency > 0) & (monetary_value > 0)
        freq = frequency[valid]
        monetary = monetary_value[valid]

        if len(freq) < 10:
            self.params_ = {'p': 1.0, 'q': 1.0, 'gamma': 1.0}
            return self

        # 使用矩估计
        mean_freq = np.mean(freq)
        mean_monetary = np.mean(monetary)
        var_monetary = np.var(monetary)

        # 参数估计
        p = max(0.1, (mean_monetary ** 2) / max(var_monetary, 0.01))
        q = max(0.1, mean_monetary / max(var_monetary / mean_monetary, 0.01))
        gamma = max(0.1, mean_monetary * p)

        self.params_ = {
            'p': p,
            'q': q,
            'gamma': gamma
        }

        return self

    def expected_average_transaction(
        self,
        frequency: np.ndarray,
        monetary_value: np.ndarray
    ) -> np.ndarray:
        """
        预测客户平均交易价值

        Args:
            frequency: 购买频率
            monetary_value: 历史平均交易价值

        Returns:
            np.ndarray: 期望平均交易价值
        """
        p = self.params_['p']
        q = self.params_['q']
        gamma = self.params_['gamma']

        expected = (gamma + p * monetary_value * frequency) / (q + gamma + frequency - 1)

        return np.nan_to_num(expected, nan=np.mean(monetary_value))

    def predict(
        self,
        df: pd.DataFrame,
        freq_col: str = 'frequency',
        monetary_col: str = 'monetary_value'
    ) -> pd.Series:
        """
        预测客户平均交易价值

        Args:
            df: 客户数据
            freq_col: 频率列名
            monetary_col: 平均交易价值列名

        Returns:
            pd.Series: 预测值
        """
        frequency = df[freq_col].values
        monetary_value = df[monetary_col].values

        return pd.Series(
            self.expected_average_transaction(frequency, monetary_value),
            index=df.index
        )


class LTVPredictor:
    """
    LTV 预测器
    整合 BG/NBD 和 Gamma-Gamma 模型
    """

    def __init__(self, discount_rate: float = 0.1):
        """
        初始化 LTV 预测器

        Args:
            discount_rate: 贴现率
        """
        self.discount_rate = discount_rate
        self.bg_nbd_model = BGNNBDModel()
        self.gamma_gamma_model = GammaGammaModel()

    def prepare_rfm_data(
        self,
        transactions_df: pd.DataFrame,
        customer_id_col: str,
        transaction_date_col: str,
        amount_col: str,
        observation_end: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        准备 RFM 数据用于 LTV 预测

        Args:
            transactions_df: 交易数据
            customer_id_col: 客户 ID 列
            transaction_date_col: 交易日期列
            amount_col: 交易金额列
            observation_end: 观察截止日期

        Returns:
            pd.DataFrame: RFM 格式数据
        """
        df = transactions_df.copy()
        df[transaction_date_col] = pd.to_datetime(df[transaction_date_col])

        if observation_end is None:
            observation_end = df[transaction_date_col].max()

        # 计算每个客户的 RFM 指标
        rfm = df.groupby(customer_id_col).agg({
            transaction_date_col: ['min', 'max', 'count'],
            amount_col: ['mean', 'sum']
        }).reset_index()

        rfm.columns = [
            customer_id_col,
            'first_purchase',
            'last_purchase',
            'frequency',
            'avg_transaction',
            'total_revenue'
        ]

        # 计算 recency 和 T
        rfm['recency'] = (rfm['last_purchase'] - rfm['first_purchase']).dt.days
        rfm['T'] = (observation_end - rfm['first_purchase']).dt.days

        # 频率 = 重复购买次数
        rfm['frequency'] = rfm['frequency'] - 1  # 减去第一次购买

        # 确保非负
        rfm['recency'] = rfm['recency'].clip(lower=0)
        rfm['T'] = rfm['T'].clip(lower=1)
        rfm['frequency'] = rfm['frequency'].clip(lower=0)

        return rfm

    def fit(
        self,
        rfm_df: pd.DataFrame,
        freq_col: str = 'frequency',
        recency_col: str = 'recency',
        T_col: str = 'T',
        monetary_col: str = 'avg_transaction'
    ) -> LTVPredictor:
        """
        拟合 LTV 模型

        Args:
            rfm_df: RFM 数据
            freq_col: 频率列名
            recency_col: 最近购买列名
            T_col: 观察期列名
            monetary_col: 平均交易价值列名
        """
        # 拟合 BG/NBD 模型
        self.bg_nbd_model.fit(
            frequency=rfm_df[freq_col].values,
            recency=rfm_df[recency_col].values,
            T=rfm_df[T_col].values
        )

        # 拟合 Gamma-Gamma 模型
        self.gamma_gamma_model.fit(
            frequency=rfm_df[freq_col].values + 1,
            monetary_value=rfm_df[monetary_col].values
        )

        return self

    def predict_ltv(
        self,
        rfm_df: pd.DataFrame,
        time_horizon: int = 12,
        freq_col: str = 'frequency',
        recency_col: str = 'recency',
        T_col: str = 'T',
        monetary_col: str = 'avg_transaction'
    ) -> pd.DataFrame:
        """
        预测客户 LTV

        Args:
            rfm_df: RFM 数据
            time_horizon: 预测时间范围 (月)
            freq_col: 频率列名
            recency_col: 最近购买列名
            T_col: 观察期列名
            monetary_col: 平均交易价值列名

        Returns:
            pd.DataFrame: LTV 预测结果
        """
        result_df = rfm_df.copy()

        # BG/NBD 预测购买次数
        bg_nbd_pred = self.bg_nbd_model.predict(
            rfm_df, t=time_horizon,
            freq_col=freq_col, recency_col=recency_col, T_col=T_col
        )

        # Gamma-Gamma 预测交易价值
        gamma_gamma_pred = self.gamma_gamma_model.predict(
            rfm_df, freq_col=freq_col, monetary_col=monetary_col
        )

        # 计算 LTV
        expected_purchases = bg_nbd_pred[f'expected_purchases_{time_horizon}m'].values
        expected_value = gamma_gamma_pred.values

        # 贴现因子
        monthly_discount = 1 / (1 + self.discount_rate / 12)
        discount_factor = (1 - monthly_discount ** time_horizon) / (1 - monthly_discount)

        # LTV = 期望购买次数 × 期望交易价值
        result_df[f'ltv_{time_horizon}m'] = expected_purchases * expected_value

        return result_df

    def calculate_customer_lifetime_value(
        self,
        rfm_df: pd.DataFrame,
        time_horizons: List[int] = [3, 6, 12, 24]
    ) -> pd.DataFrame:
        """
        计算多个时间范围的 LTV

        Args:
            rfm_df: RFM 数据
            time_horizons: 时间范围列表 (月)

        Returns:
            pd.DataFrame: 多期 LTV 预测
        """
        result_df = rfm_df.copy()

        for horizon in time_horizons:
            pred = self.predict_ltv(rfm_df, time_horizon=horizon)
            result_df[f'ltv_{horizon}m'] = pred[f'ltv_{horizon}m']

        return result_df

    def create_ltv_distribution_chart(
        self,
        ltv_values: pd.Series,
        title: str = 'LTV Distribution'
    ) -> go.Figure:
        """
        创建 LTV 分布图

        Args:
            ltv_values: LTV 值
            title: 图表标题

        Returns:
            go.Figure: Plotly 图表
        """
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('LTV Distribution', 'LTV by Decile')
        )

        # 直方图
        fig.add_trace(
            go.Histogram(
                x=ltv_values,
                nbinsx=50,
                name='LTV',
                marker_color='#636EFA'
            ),
            row=1, col=1
        )

        # 分位数分析
        deciles = pd.qcut(ltv_values, q=10, duplicates='drop')
        decile_means = deciles.map(lambda x: x.mid)

        fig.add_trace(
            go.Scatter(
                x=list(range(1, len(decile_means) + 1)),
                y=[ltv_values[deciles == d].mean() for d in deciles.unique()],
                mode='lines+markers',
                name='Decile Mean',
                marker_color='#EF553B'
            ),
            row=1, col=2
        )

        fig.update_layout(
            title=title,
            xaxis_title='LTV',
            yaxis_title='Count',
            width=900,
            height=400,
            showlegend=False
        )

        fig.update_xaxes(title_text='LTV Value', row=1, col=2)
        fig.update_yaxes(title_text='Mean LTV', row=1, col=2)

        return fig

    def create_ltv_cumulative_chart(
        self,
        rfm_df: pd.DataFrame,
        ltv_col: str
    ) -> go.Figure:
        """
        创建 LTV 累积曲线

        Args:
            rfm_df: RFM 数据
            ltv_col: LTV 列名

        Returns:
            go.Figure: Plotly 图表
        """
        sorted_ltv = rfm_df[ltv_col].sort_values(ascending=False)
        cumulative = sorted_ltv.cumsum()
        total = cumulative.iloc[-1]

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=list(range(1, len(sorted_ltv) + 1)),
            y=cumulative.values,
            mode='lines',
            name='Cumulative LTV',
            line=dict(color='#00CC96', width=2)
        ))

        # 标记 80/20 线
        idx_80 = (cumulative >= total * 0.8).idxmax()
        fig.add_trace(go.Scatter(
            x=[idx_80, idx_80],
            y=[0, total * 0.8],
            mode='lines',
            name='80% LTV',
            line=dict(color='red', dash='dash')
        ))

        fig.add_annotation(
            x=idx_80,
            y=total * 0.8,
            text=f'Top {idx_80/len(sorted_ltv)*100:.1f}% customers',
            showarrow=True,
            arrowhead=2
        )

        fig.update_layout(
            title='Cumulative LTV Analysis',
            xaxis_title='Customer Rank',
            yaxis_title='Cumulative LTV',
            width=700,
            height=500
        )

        return fig

    def create_prediction_summary(
        self,
        rfm_df: pd.DataFrame,
        ltv_col: str
    ) -> Dict[str, Any]:
        """
        创建预测摘要

        Args:
            rfm_df: RFM 数据
            ltv_col: LTV 列名

        Returns:
            Dict: 预测摘要
        """
        ltv_values = rfm_df[ltv_col]

        return {
            'total_customers': len(rfm_df),
            'mean_ltv': round(ltv_values.mean(), 2),
            'median_ltv': round(ltv_values.median(), 2),
            'std_ltv': round(ltv_values.std(), 2),
            'min_ltv': round(ltv_values.min(), 2),
            'max_ltv': round(ltv_values.max(), 2),
            'total_ltv': round(ltv_values.sum(), 2),
            'top_10_pct_ltv': round(ltv_values.quantile(0.9), 2),
            'top_1_pct_ltv': round(ltv_values.quantile(0.99), 2),
            'gini_coefficient': self._calculate_gini(ltv_values.values)
        }

    def _calculate_gini(self, values: np.ndarray) -> float:
        """计算基尼系数"""
        sorted_values = np.sort(values)
        n = len(sorted_values)
        index = np.arange(1, n + 1)
        return (2 * np.sum(index * sorted_values) / (n * np.sum(sorted_values))) - (n + 1) / n

    def export_predictions(
        self,
        predictions: pd.DataFrame,
        file_path: str,
        format: str = 'csv'
    ) -> str:
        """
        导出预测结果

        Args:
            predictions: 预测数据
            file_path: 输出路径
            format: 格式 ('csv', 'excel')

        Returns:
            str: 输出路径
        """
        if format == 'csv':
            predictions.to_csv(file_path, index=False)
        elif format == 'excel':
            with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
                predictions.to_excel(writer, sheet_name='LTV Predictions', index=False)

        return file_path


def predict_customer_ltv(
    transactions_df: pd.DataFrame,
    customer_id_col: str,
    transaction_date_col: str,
    amount_col: str,
    time_horizon: int = 12
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    便捷函数：预测客户 LTV

    Args:
        transactions_df: 交易数据
        customer_id_col: 客户 ID 列
        transaction_date_col: 交易日期列
        amount_col: 交易金额列
        time_horizon: 预测时间 (月)

    Returns:
        Tuple[pd.DataFrame, Dict]: LTV 预测结果和摘要
    """
    predictor = LTVPredictor()

    # 准备数据
    rfm_df = predictor.prepare_rfm_data(
        transactions_df,
        customer_id_col=customer_id_col,
        transaction_date_col=transaction_date_col,
        amount_col=amount_col
    )

    # 拟合模型
    predictor.fit(rfm_df)

    # 预测
    predictions = predictor.predict_ltv(rfm_df, time_horizon=time_horizon)

    # 摘要
    summary = predictor.create_prediction_summary(predictions, f'ltv_{time_horizon}m')

    return predictions, summary


if __name__ == "__main__":
    # 测试
    print("LTV 预测模块测试")
    print("=" * 50)

    # 生成测试数据
    np.random.seed(42)
    n_customers = 500

    test_df = pd.DataFrame({
        'customer_id': [f'C{i:04d}' for i in range(n_customers)],
        'frequency': np.random.poisson(5, n_customers),
        'recency': np.random.exponential(100, n_customers),
        'T': np.random.uniform(200, 400, n_customers),
        'avg_transaction': np.random.lognormal(4, 0.5, n_customers)
    })

    # 确保数据有效
    test_df['frequency'] = test_df['frequency'].clip(lower=0)
    test_df['recency'] = test_df['recency'].clip(lower=1)
    test_df['T'] = test_df['T'].clip(lower=test_df['recency'])

    # 预测
    predictor = LTVPredictor()
    predictor.fit(test_df)

    predictions = predictor.predict_ltv(test_df, time_horizon=12)

    print(f"预测客户数：{len(predictions)}")
    print(f"平均 LTV (12 月): {predictions['ltv_12m'].mean():.2f}")
    print(f"LTV 中位数：{predictions['ltv_12m'].median():.2f}")

    # 可视化
    fig = predictor.create_ltv_distribution_chart(predictions['ltv_12m'])
    fig.show()
