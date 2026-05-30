"""
聚类引擎 - Clustering Engine
提供多种聚类算法 (K-Means, DBSCAN, Hierarchical)
自动选择最佳算法和最优聚类数
"""

from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
)
from sklearn.decomposition import PCA
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings

warnings.filterwarnings('ignore')


@dataclass
class ClusteringResult:
    """聚类结果"""
    df: pd.DataFrame
    algorithm: str
    n_clusters: int
    labels: np.ndarray
    centroids: Optional[np.ndarray]
    metrics: Dict[str, float]
    cluster_summary: pd.DataFrame
    scaler: StandardScaler
    features_used: List[str]


class ClusteringEngine:
    """
    聚类引擎
    支持多种算法和自动参数选择
    """

    ALGORITHMS = ['kmeans', 'dbscan', 'hierarchical', 'gmm']

    def __init__(
        self,
        n_clusters_range: Tuple[int, int] = (2, 10),
        scaling_method: str = 'standard',
        random_state: int = 42
    ):
        """
        初始化聚类引擎

        Args:
            n_clusters_range: 聚类数搜索范围 (min, max)
            scaling_method: 标准化方法 ('standard', 'minmax', 'robust')
            random_state: 随机种子
        """
        self.n_clusters_range = n_clusters_range
        self.scaling_method = scaling_method
        self.random_state = random_state
        self._scaler = None
        self._best_algorithm = None
        self._best_n_clusters = None
        self._best_score = -1

    def fit(
        self,
        df: pd.DataFrame,
        features: Optional[List[str]] = None,
        algorithm: str = 'auto',
        n_clusters: Optional[int] = None,
        dbscan_eps: float = 0.5,
        dbscan_min_samples: int = 5
    ) -> ClusteringResult:
        """
        执行聚类分析

        Args:
            df: 输入数据
            features: 用于聚类的特征列
            algorithm: 算法选择 ('kmeans', 'dbscan', 'hierarchical', 'gmm', 'auto')
            n_clusters: 聚类数 (可选，如不指定则自动选择)
            dbscan_eps: DBSCAN eps 参数
            dbscan_min_samples: DBSCAN min_samples 参数

        Returns:
            ClusteringResult: 聚类结果
        """
        df = df.copy()

        # 选择特征
        if features is None:
            features = df.select_dtypes(include=[np.number]).columns.tolist()

        # 处理缺失值
        X = df[features].copy()
        X = X.fillna(X.median())

        # 标准化
        self._scaler = self._get_scaler()
        X_scaled = self._scaler.fit_transform(X)

        # 自动选择算法和参数
        if algorithm == 'auto':
            return self._auto_cluster(df, X_scaled, features)

        # 确定聚类数
        if n_clusters is None:
            n_clusters = self._find_optimal_clusters(X_scaled, algorithm)

        # 执行聚类
        labels, centroids = self._run_clustering(
            X_scaled, algorithm, n_clusters, dbscan_eps, dbscan_min_samples
        )

        # 计算评估指标
        metrics = self._calculate_metrics(X_scaled, labels)

        # 创建结果
        result_df = df.copy()
        result_df['cluster'] = labels

        cluster_summary = self._calculate_cluster_summary(result_df, features, 'cluster')

        return ClusteringResult(
            df=result_df,
            algorithm=algorithm,
            n_clusters=n_clusters if n_clusters > 0 else len(set(labels)) - (1 if -1 in labels else 0),
            labels=labels,
            centroids=centroids,
            metrics=metrics,
            cluster_summary=cluster_summary,
            scaler=self._scaler,
            features_used=features
        )

    def _get_scaler(self) -> StandardScaler:
        """获取标准化器"""
        if self.scaling_method == 'standard':
            return StandardScaler()
        elif self.scaling_method == 'minmax':
            return MinMaxScaler()
        else:
            return StandardScaler()

    def _auto_cluster(
        self,
        df: pd.DataFrame,
        X_scaled: np.ndarray,
        features: List[str]
    ) -> ClusteringResult:
        """自动选择最佳聚类算法和参数"""
        best_result = None
        best_score = -1

        for algo in ['kmeans', 'hierarchical', 'gmm']:
            # 寻找最优聚类数
            n_clusters = self._find_optimal_clusters(X_scaled, algo)

            # 执行聚类
            labels, centroids = self._run_clustering(X_scaled, algo, n_clusters)

            # 计算分数
            if len(set(labels)) > 1:
                score = silhouette_score(X_scaled, labels)

                if score > best_score:
                    best_score = score
                    best_result = (algo, n_clusters, labels, centroids)

        # 尝试 DBSCAN
        try:
            dbscan_labels = self._run_dbscan_optimized(X_scaled)
            if len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0) > 1:
                dbscan_score = silhouette_score(
                    X_scaled[dbscan_labels != -1],
                    dbscan_labels[dbscan_labels != -1]
                )
                if dbscan_score > best_score:
                    best_result = ('dbscan', len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0), dbscan_labels, None)
                    best_score = dbscan_score
        except Exception:
            pass

        if best_result is None:
            # 默认使用 K-Means
            n_clusters = self._find_optimal_clusters(X_scaled, 'kmeans')
            labels, centroids = self._run_clustering(X_scaled, 'kmeans', n_clusters)
            best_result = ('kmeans', n_clusters, labels, centroids)

        algo, n_clusters, labels, centroids = best_result
        metrics = self._calculate_metrics(X_scaled, labels)

        result_df = df.copy()
        result_df['cluster'] = labels
        cluster_summary = self._calculate_cluster_summary(result_df, features, 'cluster')

        self._best_algorithm = algo
        self._best_n_clusters = n_clusters
        self._best_score = best_score

        return ClusteringResult(
            df=result_df,
            algorithm=algo,
            n_clusters=n_clusters,
            labels=labels,
            centroids=centroids,
            metrics=metrics,
            cluster_summary=cluster_summary,
            scaler=self._scaler,
            features_used=features
        )

    def _find_optimal_clusters(
        self,
        X_scaled: np.ndarray,
        algorithm: str
    ) -> int:
        """寻找最优聚类数"""
        min_k, max_k = self.n_clusters_range
        max_k = min(max_k, len(X_scaled) - 2)

        if max_k < 2:
            return 2

        scores = []
        k_values = range(min_k, max_k + 1)

        for k in k_values:
            try:
                if algorithm == 'kmeans':
                    model = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
                elif algorithm == 'hierarchical':
                    model = AgglomerativeClustering(n_clusters=k)
                elif algorithm == 'gmm':
                    model = GaussianMixture(n_components=k, random_state=self.random_state)
                else:
                    continue

                if algorithm in ['kmeans', 'gmm']:
                    labels = model.fit_predict(X_scaled)
                else:
                    labels = model.fit_predict(X_scaled)

                if len(set(labels)) > 1:
                    score = silhouette_score(X_scaled, labels)
                    scores.append((k, score))
            except Exception:
                continue

        if not scores:
            return 4  # 默认值

        # 选择最高 silhouette 分数的 k
        best_k, best_score = max(scores, key=lambda x: x[1])
        return best_k

    def _run_clustering(
        self,
        X_scaled: np.ndarray,
        algorithm: str,
        n_clusters: int,
        eps: float = 0.5,
        min_samples: int = 5
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """执行聚类"""
        if algorithm == 'kmeans':
            model = KMeans(
                n_clusters=n_clusters,
                random_state=self.random_state,
                n_init=10,
                max_iter=300
            )
            labels = model.fit_predict(X_scaled)
            centroids = model.cluster_centers_

        elif algorithm == 'hierarchical':
            model = AgglomerativeClustering(
                n_clusters=n_clusters,
                linkage='ward'
            )
            labels = model.fit_predict(X_scaled)
            centroids = None  # 层次聚类没有质心

        elif algorithm == 'gmm':
            model = GaussianMixture(
                n_components=n_clusters,
                random_state=self.random_state,
                covariance_type='full'
            )
            labels = model.fit_predict(X_scaled)
            centroids = model.means_

        elif algorithm == 'dbscan':
            model = DBSCAN(eps=eps, min_samples=min_samples)
            labels = model.fit_predict(X_scaled)
            centroids = None

        else:
            raise ValueError(f"不支持的算法：{algorithm}")

        return labels, centroids

    def _run_dbscan_optimized(self, X_scaled: np.ndarray) -> np.ndarray:
        """优化的 DBSCAN 参数搜索"""
        # 使用 k-distance graph 估算 eps
        from sklearn.neighbors import NearestNeighbors

        k = min(5, len(X_scaled) - 1)
        neighbors = NearestNeighbors(n_neighbors=k)
        neighbors_fit = neighbors.fit(X_scaled)
        distances, _ = neighbors_fit.kneighbors(X_scaled)

        # 使用平均 k 距离
        k_distances = distances[:, -1]
        eps = np.percentile(k_distances, 90)

        model = DBSCAN(eps=eps, min_samples=k)
        return model.fit_predict(X_scaled)

    def _calculate_metrics(
        self,
        X_scaled: np.ndarray,
        labels: np.ndarray
    ) -> Dict[str, float]:
        """计算聚类评估指标"""
        unique_labels = set(labels)
        n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)

        if n_clusters < 2:
            return {
                'silhouette_score': 0,
                'calinski_harabasz_score': 0,
                'davies_bouldin_score': float('inf'),
                'n_clusters': n_clusters,
                'noise_ratio': np.sum(labels == -1) / len(labels) if -1 in unique_labels else 0
            }

        # 排除噪声点计算
        if -1 in unique_labels:
            mask = labels != -1
            X_calc = X_scaled[mask]
            labels_calc = labels[mask]
        else:
            X_calc = X_scaled
            labels_calc = labels

        return {
            'silhouette_score': silhouette_score(X_calc, labels_calc),
            'calinski_harabasz_score': calinski_harabasz_score(X_calc, labels_calc),
            'davies_bouldin_score': davies_bouldin_score(X_calc, labels_calc),
            'n_clusters': n_clusters,
            'noise_ratio': np.sum(labels == -1) / len(labels) if -1 in unique_labels else 0
        }

    def _calculate_cluster_summary(
        self,
        df: pd.DataFrame,
        features: List[str],
        cluster_col: str
    ) -> pd.DataFrame:
        """计算聚类摘要统计"""
        numeric_cols = features + [cluster_col]
        available_cols = [c for c in numeric_cols if c in df.columns]

        summary = df.groupby(cluster_col)[available_cols].agg([
            ('mean', lambda x: x.mean()),
            ('std', lambda x: x.std()),
            ('min', lambda x: x.min()),
            ('max', lambda x: x.max()),
            ('count', lambda x: x.count())
        ])

        # 添加簇大小
        cluster_sizes = df[cluster_col].value_counts().sort_index()
        summary[('size', '')] = cluster_sizes
        summary[('percentage', '')] = (cluster_sizes / len(df) * 100).round(2)

        return summary

    def plot_elbow(
        self,
        X_scaled: np.ndarray,
        max_k: int = 10
    ) -> go.Figure:
        """
        绘制肘部法则图

        Args:
            X_scaled: 标准化后的数据
            max_k: 最大聚类数

        Returns:
            go.Figure: Plotly 图表
        """
        max_k = min(max_k, len(X_scaled) - 2)
        inertias = []
        k_range = range(1, max_k + 1)

        for k in k_range:
            model = KMeans(
                n_clusters=k,
                random_state=self.random_state,
                n_init=10
            )
            model.fit(X_scaled)
            inertias.append(model.inertia_)

        # 计算二阶导数找肘部
        if len(inertias) > 2:
            first_diff = np.diff(inertias)
            second_diff = np.diff(first_diff)
            elbow_idx = np.argmax(second_diff) + 1
        else:
            elbow_idx = 0

        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Elbow Method', 'Silhouette Analysis')
        )

        # 肘部图
        fig.add_trace(
            go.Scatter(
                x=list(k_range),
                y=inertias,
                mode='lines+markers',
                name='Inertia',
                marker=dict(size=10)
            ),
            row=1, col=1
        )

        # 标记肘部
        if elbow_idx > 0:
            fig.add_trace(
                go.Scatter(
                    x=[elbow_idx + 1],
                    y=[inertias[elbow_idx]],
                    mode='markers',
                    marker=dict(size=15, color='red', symbol='star'),
                    name=f'Elbow (k={elbow_idx + 1})',
                    showlegend=False
                ),
                row=1, col=1
            )

        # Silhouette 分数
        silhouette_scores = []
        for k in range(2, max_k + 1):
            model = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            labels = model.fit_predict(X_scaled)
            score = silhouette_score(X_scaled, labels)
            silhouette_scores.append(score)

        fig.add_trace(
            go.Scatter(
                x=list(range(2, max_k + 1)),
                y=silhouette_scores,
                mode='lines+markers',
                name='Silhouette',
                marker=dict(size=10)
            ),
            row=1, col=2
        )

        fig.update_layout(
            title='Optimal Cluster Selection',
            xaxis_title='Number of Clusters (k)',
            yaxis_title='Inertia',
            width=900,
            height=400
        )

        fig.update_xaxes(title_text='Number of Clusters', row=1, col=2)
        fig.update_yaxes(title_text='Silhouette Score', row=1, col=2)

        return fig

    def plot_clusters_2d(
        self,
        result: ClusteringResult,
        title: str = 'Cluster Visualization'
    ) -> go.Figure:
        """
        2D 聚类可视化 (使用 PCA)

        Args:
            result: 聚类结果
            title: 图表标题

        Returns:
            go.Figure: Plotly 图表
        """
        X_scaled = result.scaler.transform(result.df[result.features_used])

        # PCA 降维
        pca = PCA(n_components=2, random_state=self.random_state)
        X_pca = pca.fit_transform(X_scaled)

        result.df['pca_1'] = X_pca[:, 0]
        result.df['pca_2'] = X_pca[:, 1]

        # 创建散点图
        fig = px.scatter(
            result.df,
            x='pca_1',
            y='pca_2',
            color='cluster',
            title=f'{title}<br>(PCA explained variance: {pca.explained_variance_ratio_.sum():.2%})',
            color_discrete_sequence=px.colors.qualitative.Set3,
            opacity=0.7
        )

        # 添加质心
        if result.centroids is not None:
            centroid_pca = pca.transform(result.centroids)
            fig.add_trace(
                go.Scatter(
                    x=centroid_pca[:, 0],
                    y=centroid_pca[:, 1],
                    mode='markers+text',
                    marker=dict(
                        size=20,
                        color='red',
                        symbol='x',
                        line=dict(width=3, color='darkred')
                    ),
                    text=list(range(len(centroid_pca))),
                    textposition='top center',
                    name='Centroids'
                )
            )

        fig.update_layout(
            xaxis_title='PCA Component 1',
            yaxis_title='PCA Component 2',
            width=700,
            height=600,
            showlegend=True
        )

        return fig

    def plot_clusters_3d(
        self,
        result: ClusteringResult,
        title: str = '3D Cluster Visualization'
    ) -> go.Figure:
        """
        3D 聚类可视化

        Args:
            result: 聚类结果
            title: 图表标题

        Returns:
            go.Figure: Plotly 图表
        """
        X_scaled = result.scaler.transform(result.df[result.features_used])

        # PCA 降维到 3D
        pca = PCA(n_components=3, random_state=self.random_state)
        X_pca = pca.fit_transform(X_scaled)

        result_df = result.df.copy()
        result_df['pca_1'] = X_pca[:, 0]
        result_df['pca_2'] = X_pca[:, 1]
        result_df['pca_3'] = X_pca[:, 2]

        fig = px.scatter_3d(
            result_df,
            x='pca_1',
            y='pca_2',
            z='pca_3',
            color='cluster',
            title=f'{title}<br>(PCA explained variance: {pca.explained_variance_ratio_.sum():.2%})',
            color_discrete_sequence=px.colors.qualitative.Set3,
            opacity=0.7,
            size_max=10
        )

        fig.update_layout(
            scene=dict(
                xaxis_title='PCA Component 1',
                yaxis_title='PCA Component 2',
                zaxis_title='PCA Component 3'
            ),
            width=800,
            height=700
        )

        return fig

    def plot_dendrogram(
        self,
        df: pd.DataFrame,
        features: Optional[List[str]] = None,
        max_clusters: int = 10
    ) -> go.Figure:
        """
        绘制树状图

        Args:
            df: 输入数据
            features: 特征列
            max_clusters: 最大聚类数

        Returns:
            go.Figure: Plotly 图表
        """
        from scipy.cluster.hierarchy import dendrogram, linkage
        from scipy.spatial.distance import squareform

        if features is None:
            features = df.select_dtypes(include=[np.number]).columns.tolist()

        X = df[features].fillna(df[features].median())
        X_scaled = StandardScaler().fit_transform(X)

        # 采样以避免过多数据
        if len(X_scaled) > 100:
            sample_idx = np.random.choice(len(X_scaled), 100, replace=False)
            X_sample = X_scaled[sample_idx]
        else:
            X_sample = X_scaled

        # 层次聚类
        linkage_matrix = linkage(X_sample, method='ward')

        # 创建树状图
        fig = go.Figure()

        # 使用 scipy  dendrogram
        from scipy.cluster.hierarchy import dendrogram

        # 创建 FigureWidget
        import matplotlib.pyplot as plt
        from scipy.cluster.hierarchy import dendrogram as scipy_dendrogram

        fig, ax = plt.subplots(figsize=(10, 6))
        scipy_dendrogram(linkage_matrix, ax=ax, truncate_mode='level', p=max_clusters)

        # 转换为 Plotly
        traces = []
        for i, line in enumerate(ax.lines):
            traces.append(go.Scatter(
                x=line.get_xdata(),
                y=line.get_ydata(),
                mode='lines',
                line=dict(color='steelblue', width=1),
                showlegend=False
            ))

        plt.close()

        fig = go.Figure(data=traces)
        fig.update_layout(
            title='Hierarchical Clustering Dendrogram',
            xaxis_title='Sample Index',
            yaxis_title='Distance',
            width=800,
            height=500
        )

        return fig

    def get_cluster_profile(
        self,
        result: ClusteringResult,
        cluster_id: int
    ) -> Dict[str, Any]:
        """
        获取特定聚类的详细画像

        Args:
            result: 聚类结果
            cluster_id: 聚类 ID

        Returns:
            Dict: 聚类画像信息
        """
        cluster_data = result.df[result.df['cluster'] == cluster_id]

        if len(cluster_data) == 0:
            return {'error': f'Cluster {cluster_id} not found'}

        profile = {
            'cluster_id': cluster_id,
            'size': len(cluster_data),
            'percentage': round(len(cluster_data) / len(result.df) * 100, 2),
            'feature_profiles': {}
        }

        for feature in result.features_used:
            if feature in cluster_data.columns:
                values = cluster_data[feature]
                profile['feature_profiles'][feature] = {
                    'mean': round(values.mean(), 4),
                    'std': round(values.std(), 4),
                    'median': round(values.median(), 4),
                    'min': round(values.min(), 4),
                    'max': round(values.max(), 4),
                    'cluster_vs_global': round(
                        values.mean() / result.df[feature].mean() - 1, 4
                    ) if result.df[feature].mean() != 0 else 0
                }

        return profile

    def export_results(
        self,
        result: ClusteringResult,
        file_path: str,
        format: str = 'csv'
    ) -> str:
        """
        导出聚类结果

        Args:
            result: 聚类结果
            file_path: 输出路径
            format: 格式 ('csv', 'excel')

        Returns:
            str: 输出路径
        """
        export_df = result.df.copy()

        if format == 'csv':
            export_df.to_csv(file_path, index=False)
        elif format == 'excel':
            with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
                export_df.to_excel(writer, sheet_name='Clusters', index=False)
                result.cluster_summary.to_excel(writer, sheet_name='Summary')

        return file_path


def cluster_customers(
    df: pd.DataFrame,
    features: Optional[List[str]] = None,
    algorithm: str = 'auto'
) -> ClusteringResult:
    """
    便捷函数：客户聚类

    Args:
        df: 客户数据
        features: 特征列
        algorithm: 算法

    Returns:
        ClusteringResult: 聚类结果
    """
    engine = ClusteringEngine()
    return engine.fit(df, features=features, algorithm=algorithm)


if __name__ == "__main__":
    # 测试
    print("聚类引擎测试")
    print("=" * 50)

    # 生成测试数据
    from sklearn.datasets import make_blobs

    X, y = make_blobs(n_samples=500, centers=4, n_features=5, random_state=42)
    test_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(5)])
    test_df['true_label'] = y

    # 自动聚类
    engine = ClusteringEngine()
    result = engine.fit(test_df, algorithm='auto')

    print(f"最佳算法：{result.algorithm}")
    print(f"聚类数：{result.n_clusters}")
    print(f"Silhouette 分数：{result.metrics['silhouette_score']:.4f}")
    print(f"\n聚类分布:")
    print(result.df['cluster'].value_counts())

    # 可视化
    fig = engine.plot_clusters_2d(result)
    fig.show()
