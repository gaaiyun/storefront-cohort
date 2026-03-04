"""
RFM 分析模块 - RFM Analyzer Module
提供客户 RFM (Recency, Frequency, Monetary) 分析功能
包括 3 维度评分、客户分群、RFM 矩阵可视化
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


@dataclass
class RFMResult:
    """RFM 分析结果"""
    df: pd.DataFrame
    r_scores: Dict[int, str]
    f_scores: Dict[int, str]
    m_scores: Dict[int, str]
    segments: Dict[str, Dict]
    summary: Dict[str, Any]


class RFMAnalyzer:
    """
    RFM 分析器
    计算客户 RFM 分数并进行分群
    """

    # 默认分段标签
    DEFAULT_LABELS = {
        5: ['Very High', 'High', 'Medium', 'Low', 'Very Low'],
        4: ['Top 25%', '25-50%', '50-75%', 'Bottom 25%'],
        3: ['High', 'Medium', 'Low']
    }

    # 客户分群定义
    SEGMENT_DEFINITIONS = {
        'Champions': {
            'description': '最近购买、高频消费、高金额客户',
            'r_range': (4, 5),
            'f_range': (4, 5),
            'm_range': (4, 5),
            'marketing_focus': '奖励计划、早期访问新产品'
        },
        'Loyal Customers': {
            'description': '经常购买的忠实客户',
            'r_range': (3, 5),
            'f_range': (3, 5),
            'm_range': (3, 5),
            'marketing_focus': '提升客单价、交叉销售'
        },
        'Potential Loyalists': {
            'description': '近期购买且消费金额高的潜力客户',
            'r_range': (4, 5),
            'f_range': (1, 3),
            'm_range': (3, 5),
            'marketing_focus': '会员计划、品牌培养'
        },
        'New Customers': {
            'description': '最近首次购买的客户',
            'r_range': (4, 5),
            'f_range': (1, 2),
            'm_range': (1, 5),
            'marketing_focus': '欢迎系列、品牌介绍'
        },
        'Promising': {
            'description': '最近购买但频率和金额较低',
            'r_range': (3, 4),
            'f_range': (1, 2),
            'm_range': (1, 2),
            'marketing_focus': '建立品牌认知、入门优惠'
        },
        'Need Attention': {
            'description': '高于平均的RFM 分数但最近未购买',
            'r_range': (2, 3),
            'f_range': (2, 4),
            'm_range': (2, 4),
            'marketing_focus': '限时优惠、个性化推荐'
        },
        'About to Sleep': {
            'description': '即将进入休眠状态的客户',
            'r_range': (2, 3),
            'f_range': (1, 2),
            'm_range': (1, 2),
            'marketing_focus': '唤醒活动、特别优惠'
        },
        'At Risk': {
            'description': '曾经高价值但许久未购买',
            'r_range': (1, 2),
            'f_range': (3, 5),
            'm_range': (3, 5),
            'marketing_focus': '强力召回、个性化优惠'
        },
        'Cant Lose Them': {
            'description': '无法挽回的高价值客户',
            'r_range': (1, 2),
            'f_range': (4, 5),
            'm_range': (4, 5),
            'marketing_focus': '赢回活动、深度调研'
        },
        'Hibernating': {
            'description': '休眠客户',
            'r_range': (1, 2),
            'f_range': (1, 2),
            'm_range': (1, 2),
            'marketing_focus': '低价唤醒、清仓促销'
        },
        'Lost': {
            'description': '已流失客户',
            'r_range': (1, 1),
            'f_range': (1, 2),
            'm_range': (1, 2),
            'marketing_focus': '低成本触达、调研原因'
        }
    }

    def __init__(
        self,
        n_segments: int = 5,
        score_method: str = 'quantile',
        reference_date: Optional[datetime] = None
    ):
        """
        初始化 RFM 分析器

        Args:
            n_segments: 分段数量 (3/4/5)
            score_method: 评分方法 ('quantile', 'equal_width', 'custom')
            reference_date: 参考日期 (用于计算最近一次购买)
        """
        self.n_segments = n_segments
        self.score_method = score_method
        self.reference_date = reference_date or datetime.now()
        self._labels = self.DEFAULT_LABELS.get(n_segments, self.DEFAULT_LABELS[5])

    def calculate_rfm(
        self,
        df: pd.DataFrame,
        customer_id_col: str,
        transaction_date_col: str,
        amount_col: str,
        r_direction: str = 'desc',
        f_direction: str = 'asc',
        m_direction: str = 'asc'
    ) -> RFMResult:
        """
        计算 RFM 分数

        Args:
            df: 交易数据 DataFrame
            customer_id_col: 客户 ID 列名
            transaction_date_col: 交易日期列名
            amount_col: 交易金额列名
            r_direction: R 值评分方向 ('desc'=越近分数越高，'asc'=越远分数越高)
            f_direction: F 值评分方向
            m_direction: M 值评分方向

        Returns:
            RFMResult: RFM 分析结果
        """
        df = df.copy()
        df[transaction_date_col] = pd.to_datetime(df[transaction_date_col])

        # 计算 RFM 原始值
        rfm = df.groupby(customer_id_col).agg({
            transaction_date_col: lambda x: (self.reference_date - x.max()).days,
            amount_col: ['count', 'sum']
        }).reset_index()

        # 扁平化列名
        rfm.columns = [customer_id_col, 'recency', 'frequency', 'monetary']

        # 计算 RFM 分数
        rfm['r_score'] = self._calculate_score(
            rfm['recency'], direction=r_direction
        )
        rfm['f_score'] = self._calculate_score(
            rfm['frequency'], direction=f_direction
        )
        rfm['m_score'] = self._calculate_score(
            rfm['monetary'], direction=m_direction
        )

        # 计算综合分数
        rfm['rfm_score'] = (
            rfm['r_score'].astype(int) * 100 +
            rfm['f_score'].astype(int) * 10 +
            rfm['m_score'].astype(int)
        )

        # 客户分群
        rfm['segment'] = rfm.apply(self._assign_segment, axis=1)

        # 分数标签
        r_scores = {i: self._labels[i-1] for i in range(1, self.n_segments + 1)}
        f_scores = r_scores.copy()
        m_scores = r_scores.copy()

        # 摘要统计
        summary = self._calculate_summary(rfm)

        return RFMResult(
            df=rfm,
            r_scores=r_scores,
            f_scores=f_scores,
            m_scores=m_scores,
            segments=self.SEGMENT_DEFINITIONS,
            summary=summary
        )

    def _calculate_score(
        self,
        values: pd.Series,
        direction: str = 'asc'
    ) -> pd.Series:
        """
        计算单个维度的分数

        Args:
            values: 原始值 Series
            direction: 方向 ('asc'=值越大分数越高，'desc'=值越小分数越高)

        Returns:
            pd.Series: 分数 (1-5)
        """
        if self.score_method == 'quantile':
            # 分位数分箱
            scores = pd.qcut(
                values.rank(method='first'),
                q=self.n_segments,
                labels=False
            ) + 1

            if direction == 'desc':
                scores = self.n_segments - scores

        elif self.score_method == 'equal_width':
            # 等距分箱
            scores = pd.cut(
                values,
                bins=self.n_segments,
                labels=False
            ) + 1

            if direction == 'desc':
                scores = self.n_segments - scores

        else:
            # 自定义方法：基于排名
            ranks = values.rank(method='average')
            score_range = ranks.max() - ranks.min()
            if score_range == 0:
                scores = pd.Series([3] * len(values))
            else:
                normalized = (ranks - ranks.min()) / score_range * (self.n_segments - 1) + 1
                scores = normalized.round().astype(int)

        return scores.astype(int)

    def _assign_segment(self, row: pd.Series) -> str:
        """
        分配客户分群

        Args:
            row: RFM 数据行

        Returns:
            str: 分群名称
        """
        r, f, m = row['r_score'], row['f_score'], row['m_score']

        for segment_name, definition in self.SEGMENT_DEFINITIONS.items():
            r_range = definition['r_range']
            f_range = definition['f_range']
            m_range = definition['m_range']

            if (r_range[0] <= r <= r_range[1] and
                f_range[0] <= f <= f_range[1] and
                m_range[0] <= m <= m_range[1]):
                return segment_name

        return 'Others'

    def _calculate_summary(self, rfm: pd.DataFrame) -> Dict[str, Any]:
        """
        计算摘要统计

        Args:
            rfm: RFM 数据

        Returns:
            Dict: 摘要统计
        """
        segment_summary = rfm.groupby('segment').agg({
            customer_id_col := 'customer_id' if 'customer_id' in rfm.columns else rfm.columns[0]: 'count',
            'recency': 'mean',
            'frequency': 'mean',
            'monetary': 'mean'
        }).round(2)

        segment_summary.columns = ['count', 'avg_recency', 'avg_frequency', 'avg_monetary']
        segment_summary['percentage'] = (segment_summary['count'] / len(rfm) * 100).round(2)

        return {
            'total_customers': len(rfm),
            'segment_distribution': segment_summary.to_dict('index'),
            'rfm_correlation': rfm[['r_score', 'f_score', 'm_score']].corr().round(3).to_dict(),
            'score_distribution': {
                'r': rfm['r_score'].value_counts().sort_index().to_dict(),
                'f': rfm['f_score'].value_counts().sort_index().to_dict(),
                'm': rfm['m_score'].value_counts().sort_index().to_dict()
            }
        }

    def create_rfm_matrix(self, rfm: pd.DataFrame, title: str = "RFM Matrix") -> go.Figure:
        """
        创建 RFM 矩阵热力图

        Args:
            rfm: RFM 数据
            title: 图表标题

        Returns:
            go.Figure: Plotly 图表
        """
        # 创建透视表
        pivot = rfm.groupby(['r_score', 'f_score']).agg({
            'monetary': 'mean',
            rfm.columns[0]: 'count'
        }).reset_index()

        pivot_table = pivot.pivot(index='r_score', columns='f_score', values='monetary')

        # 创建热力图
        fig = go.Figure(data=go.Heatmap(
            z=pivot_table.values,
            x=pivot_table.columns.astype(str),
            y=pivot_table.index.astype(str),
            colorscale='RdYlGn',
            showscale=True,
            text=np.round(pivot_table.values, 1),
            texttemplate="%{text}",
            textfont={"size": 12},
            colorbar=dict(title="平均金额")
        ))

        fig.update_layout(
            title=title,
            xaxis_title="Frequency Score",
            yaxis_title="Recency Score",
            yaxis=dict(autorange='reversed'),
            width=600,
            height=500
        )

        return fig

    def create_segment_chart(self, rfm: pd.DataFrame) -> go.Figure:
        """
        创建客户分群饼图

        Args:
            rfm: RFM 数据

        Returns:
            go.Figure: Plotly 图表
        """
        segment_counts = rfm['segment'].value_counts()

        # 只展示 top 10 分群
        if len(segment_counts) > 10:
            top_10 = segment_counts.head(10)
            other = segment_counts[10:].sum()
            if other > 0:
                top_10['Others'] = other
            segment_counts = top_10

        fig = px.pie(
            values=segment_counts.values,
            names=segment_counts.index,
            title='Customer Segment Distribution',
            hole=0.3,
            color_discrete_sequence=px.colors.qualitative.Set3
        )

        fig.update_traces(
            textposition='inside',
            textinfo='percent+label',
            textfont_size=12
        )

        fig.update_layout(
            width=600,
            height=500,
            showlegend=True,
            legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1)
        )

        return fig

    def create_rfm_distribution(self, rfm: pd.DataFrame) -> go.Figure:
        """
        创建 RFM 分数分布图

        Args:
            rfm: RFM 数据

        Returns:
            go.Figure: Plotly 图表
        """
        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=('Recency Distribution', 'Frequency Distribution', 'Monetary Distribution'),
            specs=[{'type': 'bar'}, {'type': 'bar'}, {'type': 'bar'}]
        )

        # R 分布
        r_dist = rfm['r_score'].value_counts().sort_index()
        fig.add_trace(
            go.Bar(
                x=r_dist.index.astype(str),
                y=r_dist.values,
                name='R',
                marker_color='#636EFA'
            ),
            row=1, col=1
        )

        # F 分布
        f_dist = rfm['f_score'].value_counts().sort_index()
        fig.add_trace(
            go.Bar(
                x=f_dist.index.astype(str),
                y=f_dist.values,
                name='F',
                marker_color='#EF553B'
            ),
            row=1, col=2
        )

        # M 分布
        m_dist = rfm['m_score'].value_counts().sort_index()
        fig.add_trace(
            go.Bar(
                x=m_dist.index.astype(str),
                y=m_dist.values,
                name='M',
                marker_color='#00CC96'
            ),
            row=1, col=3
        )

        fig.update_layout(
            title='RFM Score Distribution',
            showlegend=False,
            width=900,
            height=400
        )

        fig.update_xaxes(title_text='Score', row=1, col=1)
        fig.update_xaxes(title_text='Score', row=1, col=2)
        fig.update_xaxes(title_text='Score', row=1, col=3)
        fig.update_yaxes(title_text='Count', row=1, col=1)

        return fig

    def create_scatter_3d(self, rfm: pd.DataFrame, opacity: float = 0.6) -> go.Figure:
        """
        创建 3D 散点图

        Args:
            rfm: RFM 数据
            opacity: 透明度

        Returns:
            go.Figure: Plotly 3D 图表
        """
        # 采样以避免过多数据点
        if len(rfm) > 2000:
            sample = rfm.sample(2000, random_state=42)
        else:
            sample = rfm

        fig = px.scatter_3d(
            sample,
            x='r_score',
            y='f_score',
            z='m_score',
            color='segment',
            size='monetary',
            color_discrete_sequence=px.colors.qualitative.Set3,
            opacity=opacity
        )

        fig.update_layout(
            title='3D RFM Visualization',
            scene=dict(
                xaxis_title='Recency Score',
                yaxis_title='Frequency Score',
                zaxis_title='Monetary Score'
            ),
            width=800,
            height=700,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )

        return fig

    def get_segment_strategies(self, segment: str) -> Dict[str, str]:
        """
        获取特定分群的营销策略

        Args:
            segment: 分群名称

        Returns:
            Dict: 策略信息
        """
        if segment in self.SEGMENT_DEFINITIONS:
            return self.SEGMENT_DEFINITIONS[segment]
        return {
            'description': '未分类客户',
            'marketing_focus': '进一步分析确定策略'
        }

    def export_rfm_results(
        self,
        rfm: pd.DataFrame,
        file_path: str,
        format: str = 'csv'
    ) -> str:
        """
        导出 RFM 结果

        Args:
            rfm: RFM 数据
            file_path: 输出文件路径
            format: 输出格式 ('csv', 'excel')

        Returns:
            str: 输出文件路径
        """
        if format == 'csv':
            rfm.to_csv(file_path, index=False)
        elif format == 'excel':
            with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
                rfm.to_excel(writer, sheet_name='RFM Analysis', index=False)

                # 添加分群摘要
                segment_summary = rfm.groupby('segment').agg({
                    rfm.columns[0]: 'count',
                    'recency': 'mean',
                    'frequency': 'mean',
                    'monetary': 'mean'
                }).round(2)
                segment_summary.to_excel(writer, sheet_name='Segment Summary')

        return file_path


def calculate_rfm(
    df: pd.DataFrame,
    customer_id_col: str,
    transaction_date_col: str,
    amount_col: str,
    reference_date: Optional[datetime] = None,
    n_segments: int = 5
) -> RFMResult:
    """
    便捷函数：计算 RFM 分数

    Args:
        df: 交易数据
        customer_id_col: 客户 ID 列
        transaction_date_col: 交易日期列
        amount_col: 交易金额列
        reference_date: 参考日期
        n_segments: 分段数量

    Returns:
        RFMResult: RFM 分析结果
    """
    analyzer = RFMAnalyzer(n_segments=n_segments, reference_date=reference_date)
    return analyzer.calculate_rfm(
        df,
        customer_id_col=customer_id_col,
        transaction_date_col=transaction_date_col,
        amount_col=amount_col
    )


if __name__ == "__main__":
    # 测试
    print("RFM 分析模块测试")
    print("=" * 50)

    # 生成测试数据
    np.random.seed(42)
    n = 1000

    test_df = pd.DataFrame({
        'customer_id': [f"C{i:04d}" for i in range(n)] * 5,
        'transaction_date': pd.date_range('2024-01-01', periods=n * 5, freq='D'),
        'amount': np.random.exponential(100, n * 5)
    })

    # 打乱顺序
    test_df = test_df.sample(frac=1, random_state=42).reset_index(drop=True)

    # 分析
    analyzer = RFMAnalyzer()
    result = analyzer.calculate_rfm(
        test_df,
        customer_id_col='customer_id',
        transaction_date_col='transaction_date',
        amount_col='amount'
    )

    print(f"总客户数：{result.summary['total_customers']}")
    print(f"分群数量：{len(result.df['segment'].unique())}")
    print(f"\n前 5 大分群:")
    print(result.df['segment'].value_counts().head(5))

    # 可视化
    fig = analyzer.create_segment_chart(result.df)
    fig.show()
