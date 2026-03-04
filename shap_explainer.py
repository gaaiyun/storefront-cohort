"""
SHAP 解释模块 - SHAP Explainer Module
提供模型可解释性和特征贡献分析
"""

from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings

warnings.filterwarnings('ignore')

try:
    import shap as shap_lib
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    shap_lib = None


@dataclass
class SHAPExplanation:
    """SHAP 解释结果"""
    shap_values: np.ndarray
    base_values: float
    feature_names: List[str]
    data: np.ndarray
    summary_stats: Dict[str, Any]
    feature_importance: pd.DataFrame


class SHAPExplainer:
    """
    SHAP 解释器
    支持多种模型的特征贡献分析
    """

    def __init__(self, model: Any, random_state: int = 42):
        """
        初始化 SHAP 解释器

        Args:
            model: 训练好的模型 (sklearn, xgboost, lightgbm 等)
            random_state: 随机种子
        """
        self.model = model
        self.random_state = random_state
        self._explainer = None
        self._shap_values = None
        self._feature_names = None
        self._background_data = None

    def explain(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        feature_names: Optional[List[str]] = None,
        max_samples: int = 1000,
        sample_method: str = 'random'
    ) -> SHAPExplanation:
        """
        计算 SHAP 值

        Args:
            X: 输入特征数据
            feature_names: 特征名称列表
            max_samples: 最大样本数 (用于加速计算)
            sample_method: 采样方法 ('random', 'kmeans', 'permutation')

        Returns:
            SHAPExplanation: SHAP 解释结果
        """
        if isinstance(X, pd.DataFrame):
            if feature_names is None:
                feature_names = X.columns.tolist()
            X_values = X.values
        else:
            X_values = X
            if feature_names is None:
                feature_names = [f'Feature_{i}' for i in range(X.shape[1])]

        # 采样
        if len(X_values) > max_samples:
            if sample_method == 'random':
                idx = np.random.choice(
                    len(X_values), max_samples, replace=False
                )
            elif sample_method == 'first':
                idx = np.arange(max_samples)
            else:
                idx = np.random.choice(
                    len(X_values), max_samples, replace=False
                )
            X_sample = X_values[idx]
        else:
            X_sample = X_values

        self._feature_names = feature_names
        self._background_data = X_sample

        # 计算 SHAP 值
        if SHAP_AVAILABLE:
            self._shap_values = self._calculate_shap_with_lib(X_sample)
        else:
            self._shap_values = self._calculate_shap_approx(X_sample)

        # 创建解释结果
        return self._create_explanation(X_sample, self._shap_values)

    def _calculate_shap_with_lib(
        self,
        X_sample: np.ndarray
    ) -> np.ndarray:
        """使用 SHAP 库计算 SHAP 值"""
        if self._explainer is None:
            # 根据模型类型选择解释器
            model_type = type(self.model).__name__.lower()

            if 'tree' in model_type or 'forest' in model_type or 'gradient' in model_type:
                try:
                    self._explainer = shap_lib.TreeExplainer(self.model)
                except Exception:
                    self._explainer = shap_lib.Explainer(
                        self.model,
                        self._background_data,
                        feature_names=self._feature_names
                    )
            elif 'linear' in model_type:
                self._explainer = shap_lib.LinearExplainer(
                    self.model,
                    self._background_data
                )
            else:
                self._explainer = shap_lib.Explainer(
                    self.model,
                    self._background_data,
                    feature_names=self._feature_names
                )

        # 计算 SHAP 值
        shap_values = self._explainer.shap_values(self._background_data)

        # 处理多分类情况
        if isinstance(shap_values, list):
            # 多分类：取绝对值平均
            shap_values = np.mean([np.abs(sv) for sv in shap_values], axis=0)

        return shap_values

    def _calculate_shap_approx(
        self,
        X_sample: np.ndarray
    ) -> np.ndarray:
        """
        近似 SHAP 值计算 (当 SHAP 库不可用时)
        使用基于排列的重要性近似
        """
        n_samples, n_features = X_sample.shape
        shap_values = np.zeros((n_samples, n_features))

        # 获取基准预测
        if hasattr(self.model, 'predict_proba'):
            base_pred = self.model.predict_proba(X_sample)
            if base_pred.ndim > 1:
                base_pred = base_pred[:, 1]
        else:
            base_pred = self.model.predict(X_sample)

        # 对每个特征计算贡献
        for feat_idx in range(n_features):
            # 创建排列版本
            X_permuted = X_sample.copy()
            np.random.shuffle(X_permuted[:, feat_idx])

            # 获取排列后的预测
            if hasattr(self.model, 'predict_proba'):
                perm_pred = self.model.predict_proba(X_permuted)
                if perm_pred.ndim > 1:
                    perm_pred = perm_pred[:, 1]
            else:
                perm_pred = self.model.predict(X_permuted)

            # SHAP 值近似
            shap_values[:, feat_idx] = base_pred - perm_pred

        return shap_values

    def _create_explanation(
        self,
        X_data: np.ndarray,
        shap_values: np.ndarray
    ) -> SHAPExplanation:
        """创建 SHAP 解释结果"""
        # 特征重要性 (基于平均绝对 SHAP 值)
        mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
        feature_importance = pd.DataFrame({
            'feature': self._feature_names,
            'importance': mean_abs_shap,
            'mean_shap': np.mean(shap_values, axis=0),
            'std_shap': np.std(shap_values, axis=0)
        }).sort_values('importance', ascending=False)

        # 摘要统计
        summary_stats = {
            'n_samples': len(X_data),
            'n_features': len(self._feature_names),
            'base_value': float(np.mean(shap_values.flatten())),
            'total_shap_variance': float(np.var(shap_values)),
            'top_features': feature_importance['feature'].head(5).tolist()
        }

        return SHAPExplanation(
            shap_values=shap_values,
            base_values=summary_stats['base_value'],
            feature_names=self._feature_names,
            data=X_data,
            summary_stats=summary_stats,
            feature_importance=feature_importance
        )

    def explain_instance(
        self,
        instance: Union[pd.Series, np.ndarray],
        instance_idx: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        解释单个预测实例

        Args:
            instance: 单个样本数据
            instance_idx: 如果是解释已计算的样本，提供索引

        Returns:
            Dict: 实例解释结果
        """
        if isinstance(instance, pd.Series):
            instance_values = instance.values
            instance_features = instance.index.tolist()
        else:
            instance_values = instance
            instance_features = self._feature_names or list(
                range(len(instance))
            )

        # 获取预测
        if hasattr(self.model, 'predict_proba'):
            pred_proba = self.model.predict_proba([instance_values])[0]
            prediction = np.argmax(pred_proba)
            confidence = pred_proba[prediction]
        else:
            prediction = self.model.predict([instance_values])[0]
            confidence = None

        # 获取 SHAP 值
        if instance_idx is not None and self._shap_values is not None:
            shap_vals = self._shap_values[instance_idx]
        else:
            shap_vals = self._calculate_shap_approx(
                instance_values.reshape(1, -1)
            )[0]

        # 创建解释
        explanation = {
            'prediction': int(prediction) if hasattr(prediction, 'item') else prediction,
            'confidence': float(confidence) if confidence is not None else None,
            'feature_contributions': {}
        }

        for i, (feat, shap_val) in enumerate(
            zip(instance_features, shap_vals)
        ):
            explanation['feature_contributions'][feat] = {
                'value': float(instance_values[i]),
                'shap_value': float(shap_val),
                'direction': 'positive' if shap_val > 0 else 'negative'
            }

        return explanation

    def plot_summary(
        self,
        explanation: SHAPExplanation,
        max_features: int = 20,
        title: str = 'SHAP Summary Plot'
    ) -> go.Figure:
        """
        绘制 SHAP 摘要图

        Args:
            explanation: SHAP 解释结果
            max_features: 最多展示的特征数
            title: 图表标题

        Returns:
            go.Figure: Plotly 图表
        """
        # 选择 top 特征
        top_features = explanation.feature_importance.head(max_features)[
            'feature'
        ].tolist()
        top_indices = [
            explanation.feature_names.index(f) for f in top_features
        ]

        # 准备数据
        shap_data = []
        for feat_idx in reversed(top_indices):
            feat_name = explanation.feature_names[feat_idx]
            shap_vals = explanation.shap_values[:, feat_idx]
            feature_vals = explanation.data[:, feat_idx]

            # 按 SHAP 值排序
            sort_idx = np.argsort(shap_vals)

            for val, shap in zip(
                feature_vals[sort_idx], shap_vals[sort_idx]
            ):
                shap_data.append({
                    'feature': feat_name,
                    'shap_value': shap,
                    'feature_value': val
                })

        shap_df = pd.DataFrame(shap_data)

        # 创建图表
        fig = go.Figure()

        # 添加散点
        for feat in top_features:
            feat_data = shap_df[shap_df['feature'] == feat]
            fig.add_trace(go.Scatter(
                x=feat_data['feature_value'],
                y=feat_data['feature'],
                mode='markers',
                name=feat,
                marker=dict(
                    size=8,
                    color=feat_data['shap_value'],
                    colorscale='RdBu',
                    showscale=False,
                    opacity=0.7
                )
            ))

        fig.update_layout(
            title=title,
            xaxis_title='SHAP Value (impact on model output)',
            yaxis_title='Feature',
            showlegend=False,
            width=900,
            height=150 * min(max_features, 15)
        )

        # 添加垂直零线
        fig.add_shape(
            type='line',
            x0=0, y0=0, x1=0, y1=1,
            yref='paper',
            line=dict(color='gray', dash='dash')
        )

        return fig

    def plot_beeswarm(
        self,
        explanation: SHAPExplanation,
        max_features: int = 15,
        title: str = 'SHAP Beeswarm Plot'
    ) -> go.Figure:
        """
        绘制 SHAP 蜂群图

        Args:
            explanation: SHAP 解释结果
            max_features: 最多展示的特征数
            title: 图表标题

        Returns:
            go.Figure: Plotly 图表
        """
        top_features = explanation.feature_importance.head(max_features)[
            'feature'
        ].tolist()
        top_indices = [
            explanation.feature_names.index(f) for f in top_features
        ]

        all_points = []

        for rank, feat_idx in enumerate(reversed(top_indices)):
            feat_name = explanation.feature_names[feat_idx]
            shap_vals = explanation.shap_values[:, feat_idx]
            feature_vals = explanation.data[:, feat_idx]

            # 归一化特征值用于着色
            if feature_vals.max() > feature_vals.min():
                norm_vals = (feature_vals - feature_vals.min()) / (
                    feature_vals.max() - feature_vals.min()
                )
            else:
                norm_vals = np.zeros_like(feature_vals)

            # 添加抖动
            y_jitter = np.random.uniform(-0.3, 0.3, len(shap_vals))

            for i, (shap, val, norm) in enumerate(
                zip(shap_vals, feature_vals, norm_vals)
            ):
                all_points.append({
                    'feature': feat_name,
                    'y_pos': rank + y_jitter[i],
                    'shap_value': shap,
                    'feature_value': val,
                    'norm_value': norm
                })

        points_df = pd.DataFrame(all_points)

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=points_df['shap_value'],
            y=points_df['y_pos'],
            mode='markers',
            marker=dict(
                size=6,
                color=points_df['norm_value'],
                colorscale='RdBu',
                colorbar=dict(
                    title='Feature\nValue',
                    x=1.02,
                    thickness=15
                ),
                opacity=0.6
            ),
            customdata=points_df[['feature_value']].values,
            hovertemplate='Feature: %{customdata[0]}<br>SHAP: %{x:.3f}<extra></extra>'
        ))

        fig.update_layout(
            title=title,
            xaxis_title='SHAP Value',
            yaxis_title='Features',
            yaxis=dict(
                ticktext=top_features[::-1],
                tickvals=list(range(len(top_features))),
                range=[-0.5, len(top_features) - 0.5]
            ),
            showlegend=False,
            width=900,
            height=100 * min(max_features, 15)
        )

        fig.add_shape(
            type='line',
            x0=0, y0=-0.5, x1=0, y1=len(top_features) - 0.5,
            line=dict(color='gray', dash='dash')
        )

        return fig

    def plot_feature_importance(
        self,
        explanation: SHAPExplanation,
        max_features: int = 20,
        title: str = 'Feature Importance (SHAP)'
    ) -> go.Figure:
        """
        绘制 SHAP 特征重要性图

        Args:
            explanation: SHAP 解释结果
            max_features: 最多展示的特征数
            title: 图表标题

        Returns:
            go.Figure: Plotly 图表
        """
        importance_df = explanation.feature_importance.head(max_features).copy()
        importance_df = importance_df.sort_values('importance', ascending=True)

        fig = px.bar(
            importance_df,
            x='importance',
            y='feature',
            orientation='h',
            title=title,
            color='importance',
            color_continuous_scale='Viridis'
        )

        fig.update_layout(
            xaxis_title='Mean |SHAP Value|',
            yaxis_title='Feature',
            showlegend=False,
            width=700,
            height=80 * min(max_features, 20)
        )

        return fig

    def plot_dependence(
        self,
        explanation: SHAPExplanation,
        feature: str,
        interaction_feature: Optional[str] = None,
        title: Optional[str] = None
    ) -> go.Figure:
        """
        绘制 SHAP 依赖图

        Args:
            explanation: SHAP 解释结果
            feature: 目标特征
            interaction_feature: 交互特征 (可选)
            title: 图表标题

        Returns:
            go.Figure: Plotly 图表
        """
        feat_idx = explanation.feature_names.index(feature)
        shap_vals = explanation.shap_values[:, feat_idx]
        feature_vals = explanation.data[:, feat_idx]

        # 确定着色变量
        if interaction_feature:
            int_idx = explanation.feature_names.index(interaction_feature)
            color_vals = explanation.data[:, int_idx]
            color_name = interaction_feature
        else:
            # 自动选择交互特征
            interactions = np.abs(
                np.corrcoef(shap_vals, explanation.data.T)[0, 1:]
            )
            if len(interactions) > 1:
                int_idx = np.argmax(interactions)
                color_vals = explanation.data[:, int_idx]
                color_name = explanation.feature_names[int_idx]
            else:
                color_vals = shap_vals
                color_name = 'SHAP'

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=feature_vals,
            y=shap_vals,
            mode='markers',
            marker=dict(
                size=8,
                color=color_vals,
                colorscale='Viridis',
                colorbar=dict(title=color_name),
                opacity=0.6
            ),
            hovertemplate=f'{feature}: %{{x:.3f}}<br>SHAP: %{{y:.3f}}<extra></extra>'
        ))

        fig.update_layout(
            title=title or f'SHAP Dependence: {feature}',
            xaxis_title=feature,
            yaxis_title='SHAP Value',
            showlegend=False,
            width=700,
            height=500
        )

        return fig

    def plot_waterfall(
        self,
        instance_explanation: Dict[str, Any],
        title: str = 'SHAP Waterfall Plot'
    ) -> go.Figure:
        """
        绘制 SHAP 瀑布图

        Args:
            instance_explanation: 实例解释结果
            title: 图表标题

        Returns:
            go.Figure: Plotly 图表
        """
        contributions = instance_explanation['feature_contributions']

        # 排序贡献
        sorted_contribs = sorted(
            contributions.items(),
            key=lambda x: abs(x[1]['shap_value']),
            reverse=True
        )

        # 计算累积值
        base_value = instance_explanation.get('confidence', 0.5)
        cumulative = base_value
        y_values = [cumulative]
        x_labels = ['Base Value']

        for feat_name, contrib in sorted_contribs[:15]:
            shap_val = contrib['shap_value']
            cumulative += shap_val
            y_values.append(cumulative)
            x_labels.append(feat_name[:15])

        y_values.append(cumulative)
        x_labels.append('Final Prediction')

        # 创建瀑布图
        fig = go.Figure(go.Waterfall(
            name="SHAP Contributions",
            orientation="v",
            measure=['relative'] * (len(x_labels) - 2) + ['total'] + ['total'],
            x=x_labels,
            textposition="outside",
            text=[f'{v:.3f}' for v in y_values],
            y=y_values,
            connector={"line": {"color": "rgb(63, 63, 63)"}},
            decreasing={"marker": {"color": "#EF553B"}},
            increasing={"marker": {"color": "#00CC96"}},
            totals={"marker": {"color": "#636EFA"}}
        ))

        fig.update_layout(
            title=title,
            showlegend=False,
            width=800,
            height=500
        )

        return fig

    def create_dashboard_data(
        self,
        explanation: SHAPExplanation
    ) -> Dict[str, Any]:
        """
        创建仪表板数据

        Args:
            explanation: SHAP 解释结果

        Returns:
            Dict: 仪表板数据
        """
        return {
            'summary_stats': explanation.summary_stats,
            'top_features': explanation.feature_importance.head(10).to_dict(
                'records'
            ),
            'shap_values': explanation.shap_values.tolist(),
            'feature_names': explanation.feature_names,
            'data_sample': explanation.data[:100].tolist()
        }

    def export_explanation(
        self,
        explanation: SHAPExplanation,
        file_path: str,
        format: str = 'csv'
    ) -> str:
        """
        导出 SHAP 解释结果

        Args:
            explanation: SHAP 解释结果
            file_path: 输出路径
            format: 格式 ('csv', 'excel')

        Returns:
            str: 输出路径
        """
        # 创建完整结果 DataFrame
        result_df = pd.DataFrame(
            explanation.shap_values,
            columns=[f'{f}_shap' for f in explanation.feature_names]
        )

        feature_df = pd.DataFrame(
            explanation.data,
            columns=explanation.feature_names
        )

        result_df = pd.concat([feature_df, result_df], axis=1)

        if format == 'csv':
            result_df.to_csv(file_path, index=False)
        elif format == 'excel':
            with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
                result_df.to_excel(writer, sheet_name='SHAP Values', index=False)
                explanation.feature_importance.to_excel(
                    writer, sheet_name='Feature Importance', index=False
                )

        return file_path


def explain_model(
    model: Any,
    X: Union[pd.DataFrame, np.ndarray],
    feature_names: Optional[List[str]] = None,
    max_samples: int = 1000
) -> Tuple[SHAPExplainer, SHAPExplanation]:
    """
    便捷函数：解释模型

    Args:
        model: 训练好的模型
        X: 特征数据
        feature_names: 特征名称
        max_samples: 最大样本数

    Returns:
        Tuple: (解释器，SHAP 解释结果)
    """
    explainer = SHAPExplainer(model)
    explanation = explainer.explain(X, feature_names, max_samples)
    return explainer, explanation


if __name__ == "__main__":
    # 测试
    print("SHAP 解释模块测试")
    print("=" * 50)

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import make_classification

    # 生成测试数据
    X, y = make_classification(
        n_samples=500, n_features=10, n_informative=7,
        random_state=42
    )

    feature_names = [f'feature_{i}' for i in range(10)]

    # 训练模型
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X, y)

    # 解释
    explainer = SHAPExplainer(model)
    explanation = explainer.explain(X[:100], feature_names, max_samples=100)

    print(f"特征数量：{explanation.summary_stats['n_features']}")
    print(f"样本数量：{explanation.summary_stats['n_samples']}")
    print(f"\nTop 5 特征:")
    print(explanation.feature_importance.head(5))

    # 解释单个实例
    instance_exp = explainer.explain_instance(X[0], instance_idx=0)
    print(f"\n实例预测：{instance_exp['prediction']}")

    # 可视化
    fig = explainer.plot_feature_importance(explanation)
    fig.show()
