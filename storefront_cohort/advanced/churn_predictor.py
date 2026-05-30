"""
流失预测模块 - Churn Predictor Module
基于集成学习 (Random Forest, XGBoost, LightGBM) 的客户流失预测
"""

from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
import pandas as pd
import numpy as np
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    VotingClassifier,
    StackingClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    GridSearchCV,
    StratifiedKFold
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    precision_recall_curve,
    roc_curve,
    confusion_matrix,
    classification_report
)
from sklearn.calibration import CalibratedClassifierCV
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings

warnings.filterwarnings('ignore')


@dataclass
class ChurnPredictionResult:
    """流失预测结果"""
    model: Any
    predictions: np.ndarray
    probabilities: np.ndarray
    metrics: Dict[str, float]
    feature_importance: pd.DataFrame
    X_test: pd.DataFrame
    y_test: pd.Series
    calibrated_model: Optional[Any] = None


class ChurnPredictor:
    """
    流失预测器
    支持多种集成学习算法
    """

    def __init__(
        self,
        test_size: float = 0.2,
        random_state: int = 42,
        n_jobs: int = -1
    ):
        """
        初始化流失预测器

        Args:
            test_size: 测试集比例
            random_state: 随机种子
            n_jobs: 并行作业数
        """
        self.test_size = test_size
        self.random_state = random_state
        self.n_jobs = n_jobs
        self._scaler = StandardScaler()
        self._label_encoders = {}
        self._models = {}
        self._best_model = None
        self._feature_columns = []

    def prepare_data(
        self,
        df: pd.DataFrame,
        target_col: str,
        feature_cols: Optional[List[str]] = None,
        categorical_cols: Optional[List[str]] = None
    ) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
        """
        准备训练数据

        Args:
            df: 输入数据
            target_col: 目标列名
            feature_cols: 特征列列表
            categorical_cols: 分类特征列

        Returns:
            Tuple: (X, y, feature_columns)
        """
        df = df.copy()

        # 自动选择特征列
        if feature_cols is None:
            feature_cols = [
                col for col in df.columns
                if col != target_col and df[col].dtype in ['int64', 'float64', 'int32', 'float32']
            ]

        # 处理分类特征
        if categorical_cols:
            for col in categorical_cols:
                if col in df.columns:
                    le = LabelEncoder()
                    df[col] = le.fit_transform(df[col].astype(str))
                    self._label_encoders[col] = le

        # 处理缺失值
        X = df[feature_cols].copy()
        X = X.fillna(X.median())

        # 目标变量
        y = df[target_col]

        self._feature_columns = feature_cols

        return X, y, feature_cols

    def train(
        self,
        df: pd.DataFrame,
        target_col: str,
        feature_cols: Optional[List[str]] = None,
        categorical_cols: Optional[List[str]] = None,
        model_type: str = 'auto'
    ) -> ChurnPredictionResult:
        """
        训练流失预测模型

        Args:
            df: 训练数据
            target_col: 目标列名
            feature_cols: 特征列
            categorical_cols: 分类特征列
            model_type: 模型类型 ('rf', 'gb', 'xgb', 'lgb', 'stacking', 'auto')

        Returns:
            ChurnPredictionResult: 训练结果
        """
        # 准备数据
        X, y, feature_cols = self.prepare_data(
            df, target_col, feature_cols, categorical_cols
        )

        # 标准化
        X_scaled = self._scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=feature_cols)

        # 划分数据集
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y
        )

        # 选择模型
        if model_type == 'auto':
            return self._train_auto(X_train, X_test, y_train, y_test)
        elif model_type == 'rf':
            model = self._create_random_forest()
        elif model_type == 'gb':
            model = self._create_gradient_boosting()
        elif model_type == 'stacking':
            model = self._create_stacking()
        elif model_type == 'calibrated':
            model = self._create_calibrated_rf()
        else:
            model = self._create_random_forest()

        # 训练
        model.fit(X_train, y_train)
        self._best_model = model

        # 预测
        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)[:, 1]

        # 评估
        metrics = self._calculate_metrics(y_test, predictions, probabilities)

        # 特征重要性
        feature_importance = self._get_feature_importance(model, feature_cols)

        return ChurnPredictionResult(
            model=model,
            predictions=predictions,
            probabilities=probabilities,
            metrics=metrics,
            feature_importance=feature_importance,
            X_test=X_test,
            y_test=y_test
        )

    def _train_auto(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series
    ) -> ChurnPredictionResult:
        """自动选择最佳模型"""
        models = {
            'rf': self._create_random_forest(),
            'gb': self._create_gradient_boosting(),
            'stacking': self._create_stacking(),
            'logistic': LogisticRegression(
                max_iter=1000,
                random_state=self.random_state,
                n_jobs=self.n_jobs
            )
        }

        best_score = 0
        best_model = None
        best_name = ''

        for name, model in models.items():
            model.fit(X_train, y_train)
            probs = model.predict_proba(X_test)[:, 1]
            score = roc_auc_score(y_test, probs)

            if score > best_score:
                best_score = score
                best_model = model
                best_name = name

        self._best_model = best_model
        predictions = best_model.predict(X_test)
        probabilities = best_model.predict_proba(X_test)[:, 1]

        metrics = self._calculate_metrics(y_test, predictions, probabilities)
        feature_importance = self._get_feature_importance(best_model, X_train.columns)

        return ChurnPredictionResult(
            model=best_model,
            predictions=predictions,
            probabilities=probabilities,
            metrics=metrics,
            feature_importance=feature_importance,
            X_test=X_test,
            y_test=y_test
        )

    def _create_random_forest(self) -> RandomForestClassifier:
        """创建随机森林模型"""
        return RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced',
            random_state=self.random_state,
            n_jobs=self.n_jobs
        )

    def _create_gradient_boosting(self) -> GradientBoostingClassifier:
        """创建梯度提升模型"""
        return GradientBoostingClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=self.random_state
        )

    def _create_stacking(self) -> StackingClassifier:
        """创建堆叠集成模型"""
        base_estimators = [
            ('rf', RandomForestClassifier(
                n_estimators=50,
                max_depth=8,
                class_weight='balanced',
                random_state=self.random_state,
                n_jobs=self.n_jobs
            )),
            ('gb', GradientBoostingClassifier(
                n_estimators=50,
                max_depth=4,
                random_state=self.random_state
            )),
            ('mlp', MLPClassifier(
                hidden_layer_sizes=(50, 25),
                max_iter=500,
                random_state=self.random_state
            ))
        ]

        return StackingClassifier(
            estimators=base_estimators,
            final_estimator=LogisticRegression(
                max_iter=1000,
                random_state=self.random_state
            ),
            cv=5,
            n_jobs=self.n_jobs
        )

    def _create_calibrated_rf(self) -> CalibratedClassifierCV:
        """创建校准的随机森林"""
        base_rf = self._create_random_forest()
        return CalibratedClassifierCV(
            base_estimator=base_rf,
            method='isotonic',
            cv=5
        )

    def _calculate_metrics(
        self,
        y_true: pd.Series,
        y_pred: np.ndarray,
        y_proba: np.ndarray
    ) -> Dict[str, float]:
        """计算评估指标"""
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1_score': f1_score(y_true, y_pred),
            'roc_auc': roc_auc_score(y_true, y_proba),
            'true_positive': int(tp),
            'true_negative': int(tn),
            'false_positive': int(fp),
            'false_negative': int(fn),
            'false_positive_rate': fp / (fp + tn) if (fp + tn) > 0 else 0,
            'false_negative_rate': fn / (tp + fn) if (tp + fn) > 0 else 0
        }

    def _get_feature_importance(
        self,
        model: Any,
        feature_cols: List[str]
    ) -> pd.DataFrame:
        """获取特征重要性"""
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
        elif hasattr(model, 'final_estimator_') and hasattr(model.final_estimator_, 'coef_'):
            importance = np.abs(model.final_estimator_.coef_[0])
        elif hasattr(model, 'estimators_'):
            # 平均多个树的重要性
            importances = []
            for est in model.estimators_:
                if hasattr(est, 'feature_importances_'):
                    importances.append(est.feature_importances_)
            importance = np.mean(importances, axis=0) if importances else np.zeros(len(feature_cols))
        else:
            importance = np.ones(len(feature_cols)) / len(feature_cols)

        importance_df = pd.DataFrame({
            'feature': feature_cols,
            'importance': importance
        }).sort_values('importance', ascending=False)

        # 归一化
        total = importance_df['importance'].sum()
        if total > 0:
            importance_df['importance_pct'] = (importance_df['importance'] / total * 100).round(2)

        return importance_df.reset_index(drop=True)

    def tune_hyperparameters(
        self,
        df: pd.DataFrame,
        target_col: str,
        feature_cols: Optional[List[str]] = None,
        model_type: str = 'rf'
    ) -> Dict[str, Any]:
        """
        超参数调优

        Args:
            df: 训练数据
            target_col: 目标列
            feature_cols: 特征列
            model_type: 模型类型

        Returns:
            Dict: 调优结果
        """
        X, y, _ = self.prepare_data(df, target_col, feature_cols)
        X_scaled = self._scaler.fit_transform(X)

        if model_type == 'rf':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            base_model = RandomForestClassifier(
                class_weight='balanced',
                random_state=self.random_state,
                n_jobs=self.n_jobs
            )
        elif model_type == 'gb':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.05, 0.1],
                'min_samples_split': [2, 5]
            }
            base_model = GradientBoostingClassifier(
                random_state=self.random_state
            )
        else:
            param_grid = {}
            base_model = None

        if base_model is None:
            return {'error': f'不支持的模型类型：{model_type}'}

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)

        grid_search = GridSearchCV(
            base_model,
            param_grid,
            cv=cv,
            scoring='roc_auc',
            n_jobs=self.n_jobs,
            verbose=0
        )

        grid_search.fit(X_scaled, y)

        return {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'cv_results': pd.DataFrame(grid_search.cv_results_)
        }

    def predict(
        self,
        df: pd.DataFrame,
        feature_cols: Optional[List[str]] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        预测流失概率

        Args:
            df: 待预测数据
            feature_cols: 特征列

        Returns:
            Tuple: (predictions, probabilities)
        """
        if self._best_model is None:
            raise ValueError("模型未训练，请先调用 train() 方法")

        if feature_cols is None:
            feature_cols = self._feature_columns

        X = df[feature_cols].copy()
        X = X.fillna(X.median())
        X_scaled = self._scaler.transform(X)

        predictions = self._best_model.predict(X_scaled)
        probabilities = self._best_model.predict_proba(X_scaled)[:, 1]

        return predictions, probabilities

    def predict_churn_risk(
        self,
        df: pd.DataFrame,
        feature_cols: Optional[List[str]] = None,
        threshold_low: float = 0.3,
        threshold_high: float = 0.7
    ) -> pd.DataFrame:
        """
        预测流失风险等级

        Args:
            df: 待预测数据
            feature_cols: 特征列
            threshold_low: 低风险阈值
            threshold_high: 高风险阈值

        Returns:
            pd.DataFrame: 预测结果
        """
        predictions, probabilities = self.predict(df, feature_cols)

        result_df = df.copy()
        result_df['churn_probability'] = probabilities
        result_df['churn_prediction'] = predictions

        # 风险等级
        def assign_risk(prob):
            if prob < threshold_low:
                return 'Low'
            elif prob < threshold_high:
                return 'Medium'
            else:
                return 'High'

        result_df['churn_risk'] = probabilities.apply(assign_risk)

        return result_df

    def plot_roc_curve(
        self,
        result: ChurnPredictionResult,
        title: str = 'ROC Curve'
    ) -> go.Figure:
        """
        绘制 ROC 曲线

        Args:
            result: 预测结果
            title: 图表标题

        Returns:
            go.Figure: Plotly 图表
        """
        fpr, tpr, thresholds = roc_curve(result.y_test, result.probabilities)

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=fpr,
            y=tpr,
            mode='lines',
            name=f'ROC Curve (AUC = {result.metrics["roc_auc"]:.4f})',
            line=dict(color='#00CC96', width=2)
        ))

        # 随机线
        fig.add_trace(go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode='lines',
            name='Random',
            line=dict(color='gray', dash='dash')
        ))

        fig.update_layout(
            title=title,
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            width=600,
            height=500,
            showlegend=True
        )

        fig.update_xaxes(range=[0, 1])
        fig.update_yaxes(range=[0, 1])

        return fig

    def plot_precision_recall_curve(
        self,
        result: ChurnPredictionResult,
        title: str = 'Precision-Recall Curve'
    ) -> go.Figure:
        """
        绘制 Precision-Recall 曲线

        Args:
            result: 预测结果
            title: 图表标题

        Returns:
            go.Figure: Plotly 图表
        """
        precision, recall, thresholds = precision_recall_curve(
            result.y_test, result.probabilities
        )

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=recall,
            y=precision,
            mode='lines',
            name=f'PR Curve (F1 = {result.metrics["f1_score"]:.4f})',
            line=dict(color='#EF553B', width=2)
        ))

        fig.update_layout(
            title=title,
            xaxis_title='Recall',
            yaxis_title='Precision',
            width=600,
            height=500
        )

        return fig

    def plot_confusion_matrix(
        self,
        result: ChurnPredictionResult,
        title: str = 'Confusion Matrix'
    ) -> go.Figure:
        """
        绘制混淆矩阵

        Args:
            result: 预测结果
            title: 图表标题

        Returns:
            go.Figure: Plotly 图表
        """
        cm = confusion_matrix(result.y_test, result.predictions)

        fig = go.Figure(data=go.Heatmap(
            z=cm,
            x=['Not Churn', 'Churn'],
            y=['Not Churn', 'Churn'],
            colorscale='Blues',
            showscale=True,
            text=cm.astype(str),
            texttemplate='%{text}',
            textfont={"size": 20}
        ))

        fig.update_layout(
            title=title,
            xaxis_title='Predicted',
            yaxis_title='Actual',
            width=500,
            height=450
        )

        return fig

    def plot_feature_importance(
        self,
        result: ChurnPredictionResult,
        top_n: int = 20,
        title: str = 'Feature Importance'
    ) -> go.Figure:
        """
        绘制特征重要性图

        Args:
            result: 预测结果
            top_n: 展示前 N 个特征
            title: 图表标题

        Returns:
            go.Figure: Plotly 图表
        """
        importance_df = result.feature_importance.head(top_n).copy()
        importance_df = importance_df.sort_values('importance', ascending=True)

        fig = px.bar(
            importance_df,
            x='importance',
            y='feature',
            orientation='h',
            title=title,
            labels={'importance': 'Importance', 'feature': 'Feature'},
            color='importance_pct',
            color_continuous_scale='Viridis'
        )

        fig.update_layout(
            width=700,
            height=400 * (min(top_n, 20) / 5),
            showlegend=False
        )

        return fig

    def create_model_comparison(
        self,
        df: pd.DataFrame,
        target_col: str,
        feature_cols: Optional[List[str]] = None
    ) -> go.Figure:
        """
        创建模型对比图

        Args:
            df: 训练数据
            target_col: 目标列
            feature_cols: 特征列

        Returns:
            go.Figure: Plotly 图表
        """
        models_to_compare = ['rf', 'gb', 'stacking']
        metrics_results = []

        for model_type in models_to_compare:
            result = self.train(df, target_col, feature_cols, model_type=model_type)
            metrics_results.append({
                'model': model_type.upper(),
                'ROC-AUC': result.metrics['roc_auc'],
                'F1-Score': result.metrics['f1_score'],
                'Precision': result.metrics['precision'],
                'Recall': result.metrics['recall'],
                'Accuracy': result.metrics['accuracy']
            })

        comparison_df = pd.DataFrame(metrics_results)
        comparison_df = comparison_df.melt(
            id_vars=['model'],
            var_name='metric',
            value_name='score'
        )

        fig = px.bar(
            comparison_df,
            x='model',
            y='score',
            color='metric',
            barmode='group',
            title='Model Performance Comparison',
            color_discrete_sequence=px.colors.qualitative.Set2
        )

        fig.update_layout(
            xaxis_title='Model',
            yaxis_title='Score',
            width=800,
            height=500
        )

        return fig

    def export_results(
        self,
        result: ChurnPredictionResult,
        predictions_df: pd.DataFrame,
        file_path: str,
        format: str = 'csv'
    ) -> str:
        """
        导出预测结果

        Args:
            result: 训练结果
            predictions_df: 预测数据
            file_path: 输出路径
            format: 格式 ('csv', 'excel')

        Returns:
            str: 输出路径
        """
        if format == 'csv':
            predictions_df.to_csv(file_path, index=False)
        elif format == 'excel':
            with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
                predictions_df.to_excel(writer, sheet_name='Predictions', index=False)
                result.feature_importance.to_excel(
                    writer, sheet_name='Feature Importance', index=False
                )

        return file_path


def predict_churn(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: Optional[List[str]] = None
) -> Tuple[ChurnPredictor, ChurnPredictionResult]:
    """
    便捷函数：训练流失预测模型

    Args:
        df: 训练数据
        target_col: 目标列
        feature_cols: 特征列

    Returns:
        Tuple: (预测器，训练结果)
    """
    predictor = ChurnPredictor()
    result = predictor.train(df, target_col, feature_cols)
    return predictor, result


if __name__ == "__main__":
    # 测试
    print("流失预测模块测试")
    print("=" * 50)

    # 生成测试数据
    np.random.seed(42)
    n_samples = 1000

    test_df = pd.DataFrame({
        'age': np.random.normal(35, 10, n_samples),
        'income': np.random.lognormal(10, 0.5, n_samples),
        'tenure': np.random.exponential(24, n_samples),
        'usage_freq': np.random.poisson(10, n_samples),
        'satisfaction': np.random.uniform(1, 5, n_samples),
        'support_calls': np.random.poisson(3, n_samples),
        'total_spend': np.random.lognormal(8, 1, n_samples),
        'days_since_login': np.random.exponential(10, n_samples)
    })

    # 生成流失标签
    churn_prob = (
        0.3 - 0.01 * test_df['satisfaction'] +
        0.02 * test_df['support_calls'] +
        0.03 * test_df['days_since_login'] / 30
    )
    churn_prob = np.clip(churn_prob, 0, 1)
    test_df['churned'] = (np.random.random(n_samples) < churn_prob).astype(int)

    # 训练
    predictor = ChurnPredictor()
    result = predictor.train(test_df, target_col='churned')

    print(f"模型性能:")
    for metric, value in result.metrics.items():
        if isinstance(value, float):
            print(f"  {metric}: {value:.4f}")

    print(f"\n特征重要性 Top 5:")
    print(result.feature_importance.head(5))

    # 可视化
    fig = predictor.plot_roc_curve(result)
    fig.show()
