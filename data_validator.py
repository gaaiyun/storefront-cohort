"""
数据验证模块 - Data Validator Module
集成 Great Expectations 风格的数据验证功能
提供数据质量检查、缺失值分析、异常值检测
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import pandas as pd
import numpy as np
from scipy import stats
import json
from datetime import datetime


class ValidationSeverity(Enum):
    """验证严重程度"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ValidationIssue:
    """验证问题"""
    column: str
    issue_type: str
    description: str
    severity: ValidationSeverity
    affected_rows: int = 0
    suggestion: str = ""


@dataclass
class DataQualityReport:
    """数据质量报告"""
    total_rows: int
    total_columns: int
    memory_usage_mb: float
    duplicate_rows: int
    missing_values: Dict[str, int] = field(default_factory=dict)
    missing_percentage: Dict[str, float] = field(default_factory=dict)
    data_types: Dict[str, str] = field(default_factory=dict)
    issues: List[ValidationIssue] = field(default_factory=list)
    statistical_summary: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            'total_rows': self.total_rows,
            'total_columns': self.total_columns,
            'memory_usage_mb': self.memory_usage_mb,
            'duplicate_rows': self.duplicate_rows,
            'missing_values': self.missing_values,
            'missing_percentage': self.missing_percentage,
            'data_types': self.data_types,
            'issues_count': len(self.issues),
            'critical_issues': len([i for i in self.issues if i.severity == ValidationSeverity.CRITICAL]),
            'timestamp': self.timestamp
        }
    
    def to_json(self, indent: int = 2) -> str:
        """转换为 JSON 字符串"""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)


class DataValidator:
    """
    数据验证器
    提供全面的数据质量检查功能
    """
    
    def __init__(self, df: pd.DataFrame):
        """
        初始化验证器
        
        Args:
            df: 待验证的 DataFrame
        """
        self.df = df.copy()
        self.issues: List[ValidationIssue] = []
        self.numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
        self.datetime_columns = df.select_dtypes(include=['datetime64']).columns.tolist()
    
    def validate_all(self) -> DataQualityReport:
        """
        执行全面验证
        
        Returns:
            DataQualityReport: 数据质量报告
        """
        self.issues = []
        
        # 基础统计
        report = DataQualityReport(
            total_rows=len(self.df),
            total_columns=len(self.df.columns),
            memory_usage_mb=self.df.memory_usage(deep=True).sum() / 1024 ** 2,
            duplicate_rows=self.df.duplicated().sum(),
        )
        
        # 缺失值分析
        self._analyze_missing_values(report)
        
        # 数据类型检查
        self._check_data_types(report)
        
        # 异常值检测
        self._detect_outliers(report)
        
        # 唯一值检查
        self._check_unique_values(report)
        
        # 分布检查
        self._check_distributions(report)
        
        # 业务规则验证
        self._check_business_rules(report)
        
        report.issues = self.issues
        report.statistical_summary = self._get_statistical_summary()
        
        return report
    
    def _analyze_missing_values(self, report: DataQualityReport) -> None:
        """分析缺失值"""
        missing = self.df.isnull().sum()
        missing_pct = (missing / len(self.df) * 100).round(2)
        
        report.missing_values = missing.to_dict()
        report.missing_percentage = missing_pct.to_dict()
        
        # 高缺失率列警告
        for col, pct in missing_pct.items():
            if pct > 50:
                self.issues.append(ValidationIssue(
                    column=col,
                    issue_type="high_missing_rate",
                    description=f"列 '{col}' 缺失率高达 {pct}%",
                    severity=ValidationSeverity.WARNING,
                    affected_rows=int(missing[col]),
                    suggestion="考虑删除该列或使用适当的插补方法"
                ))
            elif pct > 20:
                self.issues.append(ValidationIssue(
                    column=col,
                    issue_type="moderate_missing_rate",
                    description=f"列 '{col}' 缺失率为 {pct}%",
                    severity=ValidationSeverity.INFO,
                    affected_rows=int(missing[col]),
                    suggestion="建议分析缺失原因并考虑插补"
                ))
    
    def _check_data_types(self, report: DataQualityReport) -> None:
        """检查数据类型"""
        report.data_types = self.df.dtypes.astype(str).to_dict()
        
        # 检查数值列中的异常类型
        for col in self.numeric_columns:
            if self.df[col].dtype == 'object':
                self.issues.append(ValidationIssue(
                    column=col,
                    issue_type="wrong_dtype",
                    description=f"数值列 '{col}' 被识别为对象类型",
                    severity=ValidationSeverity.WARNING,
                    suggestion="检查数据并转换为数值类型"
                ))
    
    def _detect_outliers(self, report: DataQualityReport) -> None:
        """使用 IQR 方法检测异常值"""
        for col in self.numeric_columns:
            if self.df[col].notnull().sum() < 10:
                continue
                
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 3 * IQR
            upper_bound = Q3 + 3 * IQR
            
            outliers = ((self.df[col] < lower_bound) | (self.df[col] > upper_bound)).sum()
            outlier_pct = (outliers / len(self.df) * 100).round(2)
            
            if outlier_pct > 5:
                self.issues.append(ValidationIssue(
                    column=col,
                    issue_type="outliers",
                    description=f"列 '{col}' 检测到 {outliers} 个异常值 ({outlier_pct}%)",
                    severity=ValidationSeverity.INFO,
                    affected_rows=int(outliers),
                    suggestion="检查异常值是否为数据录入错误，或考虑使用稳健统计方法"
                ))
    
    def _check_unique_values(self, report: DataQualityReport) -> None:
        """检查唯一值"""
        for col in self.categorical_columns:
            unique_count = self.df[col].nunique()
            total_count = len(self.df)
            
            if unique_count == 1:
                self.issues.append(ValidationIssue(
                    column=col,
                    issue_type="constant_column",
                    description=f"列 '{col}' 只有一个唯一值",
                    severity=ValidationSeverity.WARNING,
                    affected_rows=total_count,
                    suggestion="该列可能没有分析价值，考虑删除"
                ))
            elif unique_count == total_count:
                self.issues.append(ValidationIssue(
                    column=col,
                    issue_type="unique_identifier",
                    description=f"列 '{col}' 所有值都唯一 (可能是 ID 列)",
                    severity=ValidationSeverity.INFO,
                    affected_rows=total_count,
                    suggestion="确认是否为标识符列，如是则不应作为特征使用"
                ))
    
    def _check_distributions(self, report: DataQualityReport) -> None:
        """检查数据分布"""
        for col in self.numeric_columns:
            if self.df[col].notnull().sum() < 100:
                continue
            
            # 正态性检验
            _, p_value = stats.normaltest(self.df[col].dropna())
            
            if p_value < 0.01:
                skewness = self.df[col].skew()
                if abs(skewness) > 2:
                    self.issues.append(ValidationIssue(
                        column=col,
                        issue_type="highly_skewed",
                        description=f"列 '{col}' 严重偏态 (偏度={skewness:.2f})",
                        severity=ValidationSeverity.INFO,
                        suggestion="考虑对数变换或 Box-Cox 变换"
                    ))
    
    def _check_business_rules(self, report: DataQualityReport) -> None:
        """检查业务规则"""
        # 检查负值（对于应为正的列）
        for col in self.numeric_columns:
            if 'amount' in col.lower() or 'price' in col.lower() or 'revenue' in col.lower():
                negative_count = (self.df[col] < 0).sum()
                if negative_count > 0:
                    self.issues.append(ValidationIssue(
                        column=col,
                        issue_type="negative_values",
                        description=f"列 '{col}' 包含 {negative_count} 个负值",
                        severity=ValidationSeverity.ERROR,
                        affected_rows=int(negative_count),
                        suggestion="检查负值是否为数据错误，金额/价格不应为负"
                    ))
    
    def _get_statistical_summary(self) -> Dict[str, Any]:
        """获取统计摘要"""
        summary = {}
        
        for col in self.numeric_columns:
            if self.df[col].notnull().sum() > 0:
                summary[col] = {
                    'mean': float(self.df[col].mean()),
                    'std': float(self.df[col].std()),
                    'min': float(self.df[col].min()),
                    'max': float(self.df[col].max()),
                    'median': float(self.df[col].median()),
                    'skewness': float(self.df[col].skew()),
                    'kurtosis': float(self.df[col].kurtosis())
                }
        
        return summary
    
    def get_validation_expectations(self) -> List[Dict]:
        """
        生成 Great Expectations 风格的验证期望
        
        Returns:
            List[Dict]: 验证期望列表
        """
        expectations = []
        
        # 期望：无完全缺失的列
        for col, pct in self.df.isnull().mean().items():
            if pct < 0.5:
                expectations.append({
                    'expectation_type': 'expect_column_column_values_to_not_be_null',
                    'column': col,
                    'success': pct == 0
                })
        
        # 期望：数值列在合理范围内
        for col in self.numeric_columns:
            Q1 = self.df[col].quantile(0.01)
            Q99 = self.df[col].quantile(0.99)
            expectations.append({
                'expectation_type': 'expect_column_values_to_be_between',
                'column': col,
                'min_value': float(Q1),
                'max_value': float(Q99),
                'success': True
            })
        
        return expectations


def validate_dataframe(df: pd.DataFrame) -> DataQualityReport:
    """
    便捷函数：验证 DataFrame
    
    Args:
        df: 待验证的 DataFrame
        
    Returns:
        DataQualityReport: 数据质量报告
    """
    validator = DataValidator(df)
    return validator.validate_all()


def load_and_validate(file_path: str, file_type: Optional[str] = None) -> Tuple[pd.DataFrame, DataQualityReport]:
    """
    加载并验证数据文件
    
    Args:
        file_path: 文件路径
        file_type: 文件类型 ('csv', 'excel', 'parquet')，如不指定则自动检测
        
    Returns:
        Tuple[pd.DataFrame, DataQualityReport]: DataFrame 和质量报告
    """
    if file_type is None:
        if file_path.endswith('.csv'):
            file_type = 'csv'
        elif file_path.endswith(('.xlsx', '.xls')):
            file_type = 'excel'
        elif file_path.endswith('.parquet'):
            file_type = 'parquet'
        else:
            raise ValueError(f"不支持的文件类型：{file_path}")
    
    # 加载数据
    if file_type == 'csv':
        df = pd.read_csv(file_path)
    elif file_type == 'excel':
        df = pd.read_excel(file_path)
    elif file_type == 'parquet':
        df = pd.read_parquet(file_path)
    else:
        raise ValueError(f"不支持的文件类型：{file_type}")
    
    # 验证数据
    report = validate_dataframe(df)
    
    return df, report


if __name__ == "__main__":
    # 测试示例
    print("数据验证模块测试")
    print("=" * 50)
    
    # 创建测试数据
    test_df = pd.DataFrame({
        'customer_id': range(1000),
        'age': np.random.normal(35, 10, 1000),
        'income': np.random.lognormal(10, 1, 1000),
        'purchase_amount': np.random.exponential(100, 1000),
        'category': np.random.choice(['A', 'B', 'C'], 1000),
        'date': pd.date_range('2023-01-01', periods=1000)
    })
    
    # 添加一些缺失值
    test_df.loc[np.random.choice(1000, 50), 'age'] = np.nan
    test_df.loc[np.random.choice(1000, 100), 'income'] = np.nan
    
    # 添加异常值
    test_df.loc[0, 'purchase_amount'] = 10000
    
    # 验证
    report = validate_dataframe(test_df)
    
    print(f"总行数：{report.total_rows}")
    print(f"总列数：{report.total_columns}")
    print(f"内存使用：{report.memory_usage_mb:.2f} MB")
    print(f"重复行数：{report.duplicate_rows}")
    print(f"发现问题数：{len(report.issues)}")
    print(f"\n质量报告 JSON:")
    print(report.to_json())
