"""
营销顾问模块 - Marketing Advisor Module
基于客户分群和预测结果的智能营销建议引擎
"""

from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta


@dataclass
class MarketingRecommendation:
    """营销建议"""
    segment: str
    strategy: str
    tactics: List[str]
    expected_roi: float
    priority: str
    budget_allocation: float
    kpi_targets: Dict[str, float]
    timeline: str


@dataclass
class CampaignPlan:
    """营销活动计划"""
    campaign_name: str
    target_segment: str
    recommendations: List[MarketingRecommendation]
    total_budget: float
    expected_revenue: float
    expected_roi: float
    timeline: Dict[str, str]
    kpis: Dict[str, float]


class MarketingAdvisor:
    """
    营销顾问引擎
    提供基于数据的营销策略和建议
    """

    # 分群策略库
    SEGMENT_STRATEGIES = {
        'Champions': {
            'strategy': 'VIP 忠诚计划',
            'description': '奖励高价值客户，培养品牌大使',
            'tactics': [
                '专属 VIP 折扣和提前购买权',
                '个性化产品推荐',
                '生日和周年特别礼遇',
                '邀请参加独家活动',
                '推荐奖励计划'
            ],
            'channels': ['Email', 'SMS', '专属客服'],
            'budget_weight': 0.25,
            'expected_roi': 3.5,
            'kpi_targets': {
                'retention_rate': 0.95,
                'repeat_purchase_rate': 0.80,
                'avg_order_value_lift': 0.15,
                'referral_rate': 0.25
            }
        },
        'Loyal Customers': {
            'strategy': '提升客单价',
            'description': '鼓励忠诚客户增加消费金额',
            'tactics': [
                '满额免运费',
                '买多优惠 (Bundle Deals)',
                '跨品类推荐',
                '会员升级激励',
                '定期订阅优惠'
            ],
            'channels': ['Email', 'App Push', 'Website'],
            'budget_weight': 0.20,
            'expected_roi': 2.8,
            'kpi_targets': {
                'retention_rate': 0.85,
                'repeat_purchase_rate': 0.70,
                'avg_order_value_lift': 0.20,
                'cross_sell_rate': 0.30
            }
        },
        'Potential Loyalists': {
            'strategy': '培养忠诚度',
            'description': '将潜力客户转化为忠诚客户',
            'tactics': [
                '新客专享持续优惠',
                '会员积分奖励',
                '品牌故事和内容营销',
                '社交媒体互动活动',
                '限时exclusive 优惠'
            ],
            'channels': ['Email', 'Social Media', 'Content Marketing'],
            'budget_weight': 0.15,
            'expected_roi': 2.5,
            'kpi_targets': {
                'retention_rate': 0.75,
                'repeat_purchase_rate': 0.50,
                'engagement_rate': 0.40,
                'membership_conversion': 0.35
            }
        },
        'New Customers': {
            'strategy': '欢迎与引导',
            'description': '建立良好第一印象，促进二次购买',
            'tactics': [
                '欢迎系列邮件 (3-5 封)',
                '首单后 follow-up 优惠',
                '产品使用指南',
                '新用户专属折扣码',
                '满意度调研邀请'
            ],
            'channels': ['Email', 'SMS', 'In-App Message'],
            'budget_weight': 0.15,
            'expected_roi': 2.0,
            'kpi_targets': {
                'second_purchase_rate': 0.40,
                'activation_rate': 0.60,
                'email_open_rate': 0.35,
                'nps_score': 50
            }
        },
        'Promising': {
            'strategy': '建立关系',
            'description': '增加互动，培养购买习惯',
            'tactics': [
                '个性化产品推荐',
                '浏览放弃提醒',
                '购物车挽回优惠',
                '限时闪购活动',
                '社交证明展示'
            ],
            'channels': ['Email', 'Retargeting Ads', 'Push Notification'],
            'budget_weight': 0.10,
            'expected_roi': 1.8,
            'kpi_targets': {
                'engagement_rate': 0.30,
                'cart_recovery_rate': 0.25,
                'purchase_frequency_lift': 0.20
            }
        },
        'Need Attention': {
            'strategy': '激活唤醒',
            'description': '防止客户流失，重新建立联系',
            'tactics': [
                '限时回归优惠',
                '"我们想念你"主题营销',
                '新品通知',
                '个性化召回邮件',
                '专属客服联系'
            ],
            'channels': ['Email', 'SMS', 'Direct Mail'],
            'budget_weight': 0.10,
            'expected_roi': 1.5,
            'kpi_targets': {
                'reactivation_rate': 0.20,
                'response_rate': 0.15,
                'churn_prevention_rate': 0.30
            }
        },
        'At Risk': {
            'strategy': '紧急挽留',
            'description': '高强度干预防止流失',
            'tactics': [
                '大额折扣优惠',
                '一对一客服联系',
                '满意度调研和问题解决',
                '特别礼遇邀请',
                '限时独家优惠'
            ],
            'channels': ['Phone', 'Email', 'SMS', 'Direct Mail'],
            'budget_weight': 0.15,
            'expected_roi': 1.2,
            'kpi_targets': {
                'retention_rate': 0.40,
                'response_rate': 0.25,
                'complaint_resolution': 0.80
            }
        },
        'Cant Lose Them': {
            'strategy': '全力赢回',
            'description': '针对高价值流失客户的最后努力',
            'tactics': [
                '最高级别优惠',
                '高层管理人员亲自联系',
                '定制化解决方案',
                '无条件退货/换货保证',
                '终身优惠承诺'
            ],
            'channels': ['Phone', 'Direct Mail', 'In-Person'],
            'budget_weight': 0.10,
            'expected_roi': 1.0,
            'kpi_targets': {
                'winback_rate': 0.25,
                'retention_6m': 0.50
            }
        },
        'Hibernating': {
            'strategy': '低价唤醒',
            'description': '用低成本方式尝试唤醒休眠客户',
            'tactics': [
                '清仓促销通知',
                '超低价特价商品',
                '免费样品/试用',
                '大规模促销邀请',
                '会员日通知'
            ],
            'channels': ['Email', 'SMS'],
            'budget_weight': 0.05,
            'expected_roi': 0.8,
            'kpi_targets': {
                'reactivation_rate': 0.10,
                'click_rate': 0.08
            }
        },
        'Lost': {
            'strategy': '低成本维护',
            'description': '保持最低限度联系，等待机会',
            'tactics': [
                '节日祝福邮件',
                '品牌新闻通讯',
                '年度大清仓',
                '重新注册邀请'
            ],
            'channels': ['Email'],
            'budget_weight': 0.02,
            'expected_roi': 0.3,
            'kpi_targets': {
                'unsubscribe_rate': 0.05,
                'minimal_engagement': 0.02
            }
        }
    }

    def __init__(
        self,
        total_budget: Optional[float] = None,
        currency: str = 'CNY'
    ):
        """
        初始化营销顾问

        Args:
            total_budget: 总营销预算
            currency: 货币单位
        """
        self.total_budget = total_budget
        self.currency = currency
        self._segment_data = None
        self._recommendations = {}

    def analyze_segments(
        self,
        rfm_df: pd.DataFrame,
        segment_col: str = 'segment',
        value_col: str = 'monetary',
        frequency_col: str = 'frequency',
        recency_col: str = 'recency'
    ) -> Dict[str, Any]:
        """
        分析客户分群

        Args:
            rfm_df: RFM 分析结果
            segment_col: 分群列名
            value_col: 价值列名
            frequency_col: 频率列名
            recency_col: 最近度列名

        Returns:
            Dict: 分群分析结果
        """
        self._segment_data = rfm_df.groupby(segment_col).agg({
            value_col: ['mean', 'sum', 'count'],
            frequency_col: 'mean',
            recency_col: 'mean'
        }).round(2)

        self._segment_data.columns = [
            'avg_value', 'total_value', 'customer_count',
            'avg_frequency', 'avg_recency'
        ]

        # 计算百分比
        total_customers = len(rfm_df)
        self._segment_data['customer_pct'] = (
            self._segment_data['customer_count'] / total_customers * 100
        ).round(2)

        total_value = rfm_df[value_col].sum()
        self._segment_data['value_pct'] = (
            self._segment_data['total_value'] / total_value * 100
        ).round(2)

        # 计算每客户价值
        self._segment_data['value_per_customer'] = (
            self._segment_data['avg_value'] * self._segment_data['avg_frequency']
        ).round(2)

        return self._segment_data.to_dict('index')

    def generate_recommendations(
        self,
        segment_analysis: Optional[Dict[str, Any]] = None
    ) -> Dict[str, MarketingRecommendation]:
        """
        生成营销建议

        Args:
            segment_analysis: 分群分析结果

        Returns:
            Dict: 各分群的营销建议
        """
        if segment_analysis is None and self._segment_data is None:
            raise ValueError("请先提供分群数据")

        if segment_analysis:
            self._segment_data = pd.DataFrame(segment_analysis).T

        recommendations = {}

        for segment in self._segment_data.index:
            if segment in self.SEGMENT_STRATEGIES:
                strategy_info = self.SEGMENT_STRATEGIES[segment]
                segment_info = self._segment_data.loc[segment]

                # 计算预算分配
                if self.total_budget:
                    budget = self.total_budget * strategy_info['budget_weight']
                else:
                    budget = segment_info['total_value'] * 0.1  # 默认 10%

                # 计算预期收益
                expected_revenue = budget * strategy_info['expected_roi']

                # 确定优先级
                if segment_info['customer_pct'] > 20 or segment_info['value_pct'] > 30:
                    priority = 'High'
                elif segment_info['customer_pct'] > 10 or segment_info['value_pct'] > 15:
                    priority = 'Medium'
                else:
                    priority = 'Low'

                recommendations[segment] = MarketingRecommendation(
                    segment=segment,
                    strategy=strategy_info['strategy'],
                    tactics=strategy_info['tactics'],
                    expected_roi=strategy_info['expected_roi'],
                    priority=priority,
                    budget_allocation=budget,
                    kpi_targets=strategy_info['kpi_targets'],
                    timeline=self._get_timeline(segment)
                )

        self._recommendations = recommendations
        return recommendations

    def _get_timeline(self, segment: str) -> str:
        """获取建议执行时间线"""
        high_priority = ['Champions', 'At Risk', 'Cant Lose Them']
        medium_priority = ['Loyal Customers', 'Need Attention', 'New Customers']

        if segment in high_priority:
            return '立即执行 (1-2 周内)'
        elif segment in medium_priority:
            return '近期执行 (2-4 周内)'
        else:
            return '常规执行 (1-3 个月内)'

    def create_campaign_plan(
        self,
        budget: Optional[float] = None,
        campaign_name: str = 'Q1 Customer Marketing Campaign'
    ) -> CampaignPlan:
        """
        创建营销活动计划

        Args:
            budget: 活动预算
            campaign_name: 活动名称

        Returns:
            CampaignPlan: 营销活动计划
        """
        if not self._recommendations:
            self.generate_recommendations()

        if budget:
            self.total_budget = budget

        # 计算总预算和预期收益
        total_budget = sum(
            r.budget_allocation for r in self._recommendations.values()
        )
        total_revenue = sum(
            r.budget_allocation * r.expected_roi
            for r in self._recommendations.values()
        )

        # 计算整体 ROI
        overall_roi = total_revenue / total_budget if total_budget > 0 else 0

        # 整合 KPI
        aggregate_kpis = {}
        for rec in self._recommendations.values():
            for kpi, target in rec.kpi_targets.items():
                if kpi not in aggregate_kpis:
                    aggregate_kpis[kpi] = []
                aggregate_kpis[kpi].append(target)

        final_kpis = {
            k: np.mean(v) for k, v in aggregate_kpis.items()
        }

        return CampaignPlan(
            campaign_name=campaign_name,
            target_segment='All Segments',
            recommendations=list(self._recommendations.values()),
            total_budget=total_budget,
            expected_revenue=total_revenue,
            expected_roi=overall_roi,
            timeline={
                'start_date': datetime.now().strftime('%Y-%m-%d'),
                'end_date': (datetime.now() + timedelta(days=90)).strftime('%Y-%m-%d'),
                'review_date': (datetime.now() + timedelta(days=30)).strftime('%Y-%m-%d')
            },
            kpis=final_kpis
        )

    def calculate_roi_estimation(
        self,
        segment: str,
        investment: float,
        time_horizon: int = 12
    ) -> Dict[str, float]:
        """
        计算 ROI 估算

        Args:
            segment: 客户分群
            investment: 投资金额
            time_horizon: 时间范围 (月)

        Returns:
            Dict: ROI 估算结果
        """
        if segment not in self.SEGMENT_STRATEGIES:
            return {'error': f'未知分群：{segment}'}

        strategy = self.SEGMENT_STRATEGIES[segment]
        base_roi = strategy['expected_roi']

        # 考虑规模效应
        if investment > 100000:
            scale_factor = 0.9  # 大额投资 ROI 略降
        elif investment > 50000:
            scale_factor = 0.95
        else:
            scale_factor = 1.0

        adjusted_roi = base_roi * scale_factor

        # 计算预期收益
        expected_return = investment * adjusted_roi
        net_profit = expected_return - investment

        # 月度分解
        monthly_roi = (1 + adjusted_roi) ** (1 / time_horizon) - 1

        return {
            'base_roi': base_roi,
            'adjusted_roi': adjusted_roi,
            'investment': investment,
            'expected_return': expected_return,
            'net_profit': net_profit,
            'monthly_roi': monthly_roi,
            'payback_period': investment / (expected_return / time_horizon)
        }

    def plot_segment_attractiveness(
        self,
        segment_analysis: Optional[Dict[str, Any]] = None,
        title: str = 'Segment Attractiveness Matrix'
    ) -> go.Figure:
        """
        绘制分群吸引力矩阵

        Args:
            segment_analysis: 分群分析结果
            title: 图表标题

        Returns:
            go.Figure: Plotly 图表
        """
        if segment_analysis is None and self._segment_data is None:
            raise ValueError("请先提供分群数据")

        if segment_analysis:
            self._segment_data = pd.DataFrame(segment_analysis).T

        # 准备数据
        plot_df = self._segment_data.reset_index()
        plot_df.columns = ['segment'] + list(self._segment_data.columns)

        # 归一化
        plot_df['norm_value'] = (
            plot_df['value_pct'] - plot_df['value_pct'].min()
        ) / (plot_df['value_pct'].max() - plot_df['value_pct'].min())

        plot_df['norm_size'] = (
            plot_df['customer_pct'] - plot_df['customer_pct'].min()
        ) / (plot_df['customer_pct'].max() - plot_df['customer_pct'].min())

        fig = go.Figure()

        # 添加象限线
        fig.add_shape(
            type='line', x0=0.5, y0=0, x1=0.5, y1=1,
            line=dict(color='gray', dash='dash')
        )
        fig.add_shape(
            type='line', x0=0, y0=0.5, x1=1, y1=0.5,
            line=dict(color='gray', dash='dash')
        )

        # 添加象限标签
        fig.add_annotation(x=0.25, y=0.75, text='高价值<br>小群体', showarrow=False)
        fig.add_annotation(x=0.75, y=0.75, text='高价值<br>大群体', showarrow=False)
        fig.add_annotation(x=0.25, y=0.25, text='低价值<br>小群体', showarrow=False)
        fig.add_annotation(x=0.75, y=0.25, text='低价值<br>大群体', showarrow=False)

        # 添加散点
        for _, row in plot_df.iterrows():
            fig.add_trace(go.Scatter(
                x=[row['norm_size'] * 100],
                y=[row['norm_value'] * 100],
                mode='markers+text',
                name=row['segment'],
                marker=dict(
                    size=row['customer_pct'] * 2,
                    color=row['norm_value'],
                    colorscale='Viridis',
                    showscale=False,
                    line=dict(width=2, color='white')
                ),
                text=[row['segment']],
                textposition='top center'
            ))

        fig.update_layout(
            title=title,
            xaxis_title='Customer Size (%)',
            yaxis_title='Value Contribution (%)',
            width=700,
            height=600,
            showlegend=False
        )

        return fig

    def plot_budget_allocation(
        self,
        recommendations: Optional[Dict[str, MarketingRecommendation]] = None,
        title: str = 'Budget Allocation by Segment'
    ) -> go.Figure:
        """
        绘制预算分配图

        Args:
            recommendations: 营销建议
            title: 图表标题

        Returns:
            go.Figure: Plotly 图表
        """
        if recommendations is None:
            recommendations = self._recommendations

        if not recommendations:
            raise ValueError("请先生成营销建议")

        budget_data = [
            {'segment': seg, 'budget': rec.budget_allocation, 'roi': rec.expected_roi}
            for seg, rec in recommendations.items()
        ]
        budget_df = pd.DataFrame(budget_data)
        budget_df = budget_df.sort_values('budget', ascending=True)

        fig = px.bar(
            budget_df,
            y='segment',
            x='budget',
            orientation='h',
            title=title,
            color='roi',
            color_continuous_scale='RdYlGn',
            labels={'budget': 'Budget Allocation', 'roi': 'Expected ROI'}
        )

        fig.update_layout(
            width=700,
            height=100 * len(budget_df),
            coloraxis_colorbar=dict(title='ROI')
        )

        return fig

    def plot_strategy_roadmap(
        self,
        recommendations: Optional[Dict[str, MarketingRecommendation]] = None,
        title: str = 'Marketing Strategy Roadmap'
    ) -> go.Figure:
        """
        绘制策略路线图

        Args:
            recommendations: 营销建议
            title: 图表标题

        Returns:
            go.Figure: Plotly 图表
        """
        if recommendations is None:
            recommendations = self._recommendations

        if not recommendations:
            raise ValueError("请先生成营销建议")

        # 准备数据
        roadmap_data = []
        priority_order = {'High': 0, 'Medium': 1, 'Low': 2}

        for seg, rec in recommendations.items():
            roadmap_data.append({
                'segment': seg,
                'priority': rec.priority,
                'priority_num': priority_order[rec.priority],
                'strategy': rec.strategy,
                'roi': rec.expected_roi,
                'tactics_count': len(rec.tactics)
            })

        roadmap_df = pd.DataFrame(roadmap_data)
        roadmap_df = roadmap_df.sort_values(['priority_num', 'roi'], ascending=[True, False])

        fig = px.scatter(
            roadmap_df,
            x='tactics_count',
            y='roi',
            size='tactics_count',
            color='priority',
            hover_data=['segment', 'strategy'],
            text='segment',
            title=title,
            color_discrete_map={'High': '#EF553B', 'Medium': '#FFA15A', 'Low': '#00CC96'},
            size_max=15
        )

        fig.update_traces(textposition='top center')
        fig.update_layout(
            xaxis_title='Number of Tactics',
            yaxis_title='Expected ROI',
            width=800,
            height=500
        )

        return fig

    def export_campaign_plan(
        self,
        plan: CampaignPlan,
        file_path: str,
        format: str = 'excel'
    ) -> str:
        """
        导出营销活动计划

        Args:
            plan: 营销活动计划
            file_path: 输出路径
            format: 格式 ('excel', 'csv')

        Returns:
            str: 输出路径
        """
        if format == 'excel':
            with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
                # 计划摘要
                summary_data = {
                    'Campaign Name': [plan.campaign_name],
                    'Total Budget': [plan.total_budget],
                    'Expected Revenue': [plan.expected_revenue],
                    'Expected ROI': [plan.expected_roi],
                    'Start Date': [plan.timeline['start_date']],
                    'End Date': [plan.timeline['end_date']]
                }
                pd.DataFrame(summary_data).to_excel(
                    writer, sheet_name='Summary', index=False
                )

                # 分群建议
                rec_data = []
                for rec in plan.recommendations:
                    rec_data.append({
                        'Segment': rec.segment,
                        'Strategy': rec.strategy,
                        'Budget': rec.budget_allocation,
                        'Expected ROI': rec.expected_roi,
                        'Priority': rec.priority,
                        'Timeline': rec.timeline,
                        'Tactics': ' | '.join(rec.tactics)
                    })
                pd.DataFrame(rec_data).to_excel(
                    writer, sheet_name='Recommendations', index=False
                )

                # KPI 目标
                kpi_df = pd.DataFrame(list(plan.kpis.items()), columns=['KPI', 'Target'])
                kpi_df.to_excel(writer, sheet_name='KPI Targets', index=False)

        elif format == 'csv':
            rec_data = []
            for rec in plan.recommendations:
                rec_data.append({
                    'segment': rec.segment,
                    'strategy': rec.strategy,
                    'budget': rec.budget_allocation,
                    'expected_roi': rec.expected_roi,
                    'priority': rec.priority
                })
            pd.DataFrame(rec_data).to_csv(file_path, index=False)

        return file_path

    def get_personalized_tactics(
        self,
        customer_profile: Dict[str, Any],
        segment: str
    ) -> List[Dict[str, Any]]:
        """
        获取个性化战术建议

        Args:
            customer_profile: 客户画像
            segment: 客户分群

        Returns:
            List[Dict]: 个性化战术列表
        """
        if segment not in self.SEGMENT_STRATEGIES:
            return []

        strategy = self.SEGMENT_STRATEGIES[segment]
        tactics = strategy['tactics']

        # 基于客户画像个性化
        personalized = []
        for tactic in tactics:
            personalized_tactic = {
                'tactic': tactic,
                'channel': strategy['channels'][0] if strategy['channels'] else 'Email',
                'priority': 'High' if customer_profile.get('value', 0) > 1000 else 'Medium',
                'message_template': self._get_message_template(tactic, customer_profile)
            }
            personalized.append(personalized_tactic)

        return personalized

    def _get_message_template(
        self,
        tactic: str,
        customer_profile: Dict[str, Any]
    ) -> str:
        """获取消息模板"""
        name = customer_profile.get('name', 'Valued Customer')

        if '欢迎' in tactic:
            return f"亲爱的{name}，欢迎加入我们！为您准备专属优惠..."
        elif 'VIP' in tactic:
            return f"尊敬的{name}，作为我们的 VIP 客户，您享有专属特权..."
        elif '唤醒' in tactic or '回归' in tactic:
            return f"{name}，我们想念您！回来享受特别优惠..."
        else:
            return f"亲爱的{name}，为您精选..."


def generate_marketing_plan(
    rfm_df: pd.DataFrame,
    budget: float = 100000,
    segment_col: str = 'segment'
) -> Tuple[MarketingAdvisor, CampaignPlan]:
    """
    便捷函数：生成营销计划

    Args:
        rfm_df: RFM 分析结果
        budget: 营销预算
        segment_col: 分群列名

    Returns:
        Tuple: (营销顾问，营销活动计划)
    """
    advisor = MarketingAdvisor(total_budget=budget)
    advisor.analyze_segments(rfm_df, segment_col=segment_col)
    advisor.generate_recommendations()
    plan = advisor.create_campaign_plan(budget=budget)
    return advisor, plan


if __name__ == "__main__":
    # 测试
    print("营销顾问模块测试")
    print("=" * 50)

    # 生成测试数据
    np.random.seed(42)
    segments = ['Champions', 'Loyal Customers', 'New Customers', 'At Risk', 'Hibernating']

    test_df = pd.DataFrame({
        'segment': np.random.choice(segments, 500),
        'monetary': np.random.lognormal(5, 1, 500),
        'frequency': np.random.poisson(5, 500),
        'recency': np.random.exponential(30, 500)
    })

    # 生成建议
    advisor = MarketingAdvisor(total_budget=50000)
    segment_analysis = advisor.analyze_segments(test_df, segment_col='segment')
    recommendations = advisor.generate_recommendations()

    # 创建计划
    plan = advisor.create_campaign_plan(budget=50000)

    print(f"营销活动计划：{plan.campaign_name}")
    print(f"总预算：¥{plan.total_budget:,.2f}")
    print(f"预期收益：¥{plan.expected_revenue:,.2f}")
    print(f"预期 ROI: {plan.expected_roi:.2f}x")
    print(f"\n分群策略:")
    for rec in plan.recommendations[:5]:
        print(f"  {rec.segment}: {rec.strategy} (ROI: {rec.expected_roi:.1f}x)")

    # ROI 估算
    roi_est = advisor.calculate_roi_estimation('Champions', 10000)
    print(f"\nChampions 分群 ROI 估算:")
    print(f"  预期回报：¥{roi_est['expected_return']:,.2f}")
    print(f"  净利润：¥{roi_est['net_profit']:,.2f}")
