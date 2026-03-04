"""
报告生成模块 - Report Generator Module
支持 PDF、Excel、PPT 格式的报告导出
"""

from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
import pandas as pd
import numpy as np
from datetime import datetime
import io
import base64
import warnings

warnings.filterwarnings('ignore')

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter, A4, landscape
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch, cm
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image
    from reportlab.lib.enums import TA_CENTER, TA_LEFT
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False

try:
    from fpdf import FPDF
    FPDF_AVAILABLE = True
except ImportError:
    FPDF_AVAILABLE = False


@dataclass
class ReportConfig:
    """报告配置"""
    title: str
    subtitle: str
    author: str
    date: str
    logo_path: Optional[str]
    page_size: str
    orientation: str
    color_scheme: Dict[str, str]


class ReportGenerator:
    """
    报告生成器
    支持多种格式的报告导出
    """

    COLOR_SCHEMES = {
        'blue': {
            'primary': '#1f77b4',
            'secondary': '#aec7e8',
            'accent': '#ff7f0e',
            'text': '#2c3e50'
        },
        'green': {
            'primary': '#2ca02c',
            'secondary': '#98df8a',
            'accent': '#d62728',
            'text': '#2c3e50'
        },
        'professional': {
            'primary': '#2c3e50',
            'secondary': '#34495e',
            'accent': '#e74c3c',
            'text': '#1a1a1a'
        }
    }

    def __init__(
        self,
        title: str,
        subtitle: str = "",
        author: str = "Customer Analytics Platform",
        color_scheme: str = 'professional'
    ):
        """
        初始化报告生成器

        Args:
            title: 报告标题
            subtitle: 副标题
            author: 作者
            color_scheme: 配色方案
        """
        self.config = ReportConfig(
            title=title,
            subtitle=subtitle,
            author=author,
            date=datetime.now().strftime('%Y-%m-%d'),
            logo_path=None,
            page_size='A4',
            orientation='portrait',
            color_scheme=self.COLOR_SCHEMES.get(color_scheme, self.COLOR_SCHEMES['professional'])
        )
        self._sections = []
        self._charts = {}

    def add_section(
        self,
        title: str,
        content: str,
        data: Optional[Any] = None,
        chart: Optional[Any] = None
    ) -> 'ReportGenerator':
        """
        添加报告章节

        Args:
            title: 章节标题
            content: 章节内容
            data: 相关数据
            chart: 相关图表

        Returns:
            self
        """
        self._sections.append({
            'title': title,
            'content': content,
            'data': data,
            'chart': chart
        })
        return self

    def add_chart(
        self,
        chart_id: str,
        chart: Any,
        caption: str = ""
    ) -> 'ReportGenerator':
        """
        添加图表

        Args:
            chart_id: 图表 ID
            chart: Plotly 图表对象
            caption: 图表说明

        Returns:
            self
        """
        self._charts[chart_id] = {'chart': chart, 'caption': caption}
        return self

    def generate_pdf(
        self,
        output_path: str,
        include_charts: bool = True
    ) -> str:
        """
        生成 PDF 报告

        Args:
            output_path: 输出路径
            include_charts: 是否包含图表

        Returns:
            str: 输出路径
        """
        if not REPORTLAB_AVAILABLE and not FPDF_AVAILABLE:
            return self._generate_pdf_fallback(output_path)

        if REPORTLAB_AVAILABLE:
            return self._generate_pdf_reportlab(output_path, include_charts)
        else:
            return self._generate_pdf_fpdf(output_path, include_charts)

    def _generate_pdf_reportlab(
        self,
        output_path: str,
        include_charts: bool
    ) -> str:
        """使用 ReportLab 生成 PDF"""
        # 页面大小
        if self.config.orientation == 'landscape':
            page_size = landscape(A4)
        else:
            page_size = A4

        doc = SimpleDocTemplate(
            output_path,
            pagesize=page_size,
            rightMargin=2*cm,
            leftMargin=2*cm,
            topMargin=2*cm,
            bottomMargin=2*cm
        )

        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            textColor=self.config.color_scheme['primary'],
            spaceAfter=12,
            alignment=TA_CENTER
        )
        subtitle_style = ParagraphStyle(
            'CustomSubtitle',
            parent=styles['Heading2'],
            fontSize=14,
            textColor=self.config.color_scheme['secondary'],
            spaceAfter=20,
            alignment=TA_CENTER
        )
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=16,
            textColor=self.config.color_scheme['primary'],
            spaceAfter=10,
            spaceBefore=15
        )
        normal_style = ParagraphStyle(
            'CustomNormal',
            parent=styles['Normal'],
            fontSize=11,
            textColor=self.config.color_scheme['text'],
            spaceAfter=8
        )

        story = []

        # 标题页
        story.append(Paragraph(self.config.title, title_style))
        if self.config.subtitle:
            story.append(Paragraph(self.config.subtitle, subtitle_style))
        story.append(Paragraph(f"Generated: {self.config.date}", normal_style))
        story.append(Paragraph(f"By: {self.config.author}", normal_style))
        story.append(Spacer(1, 0.5*inch))

        # 各章节
        for section in self._sections:
            story.append(Paragraph(section['title'], heading_style))
            story.append(Paragraph(section['content'], normal_style))
            story.append(Spacer(1, 0.2*inch))

            # 添加数据表格
            if section['data'] is not None:
                if isinstance(section['data'], pd.DataFrame):
                    table_data = self._create_table_data(section['data'].head(10))
                    table = Table(table_data)
                    table.setStyle(TableStyle([
                        ('BACKGROUND', (0, 0), (-1, 0), self.config.color_scheme['primary']),
                        ('TEXTCOLOR', (0, 0), (-1, 0), 'white'),
                        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                        ('FONTSIZE', (0, 0), (-1, 0), 10),
                        ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
                        ('BACKGROUND', (0, 1), (-1, -1), '#f5f5f5'),
                        ('FONTSIZE', (0, 1), (-1, -1), 8),
                        ('GRID', (0, 0), (-1, -1), 0.5, '#cccccc')
                    ]))
                    story.append(table)
                    story.append(Spacer(1, 0.2*inch))

        # 构建 PDF
        doc.build(story)

        return output_path

    def _generate_pdf_fpdf(
        self,
        output_path: str,
        include_charts: bool
    ) -> str:
        """使用 FPDF 生成 PDF (备用方案)"""
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()

        # 标题
        pdf.set_font('Arial', 'B', 20)
        pdf.cell(0, 15, self.config.title, ln=True, align='C')

        if self.config.subtitle:
            pdf.set_font('Arial', 'I', 14)
            pdf.cell(0, 10, self.config.subtitle, ln=True, align='C')

        pdf.set_font('Arial', '', 10)
        pdf.cell(0, 8, f"Generated: {self.config.date}", ln=True, align='C')
        pdf.cell(0, 8, f"By: {self.config.author}", ln=True, align='C')

        pdf.ln(10)

        # 章节
        for section in self._sections:
            pdf.set_font('Arial', 'B', 14)
            pdf.cell(0, 10, section['title'], ln=True)

            pdf.set_font('Arial', '', 11)
            # 处理中文需要特殊处理，这里简化
            for line in section['content'].split('\n'):
                if line.strip():
                    pdf.cell(0, 6, line[:80], ln=True)

            pdf.ln(5)

        pdf.output(output_path)
        return output_path

    def _generate_pdf_fallback(
        self,
        output_path: str
    ) -> str:
        """无 PDF 库时的降级方案"""
        # 生成 HTML 格式
        html_content = self._generate_html_report()

        output_html = output_path.replace('.pdf', '.html')
        with open(output_html, 'w', encoding='utf-8') as f:
            f.write(html_content)

        return output_html

    def _create_table_data(
        self,
        df: pd.DataFrame,
        max_rows: int = 20
    ) -> List[List[str]]:
        """创建表格数据"""
        # 表头
        table_data = [df.columns.tolist()]

        # 数据行
        for _, row in df.head(max_rows).iterrows():
            table_data.append([str(v) for v in row.values])

        return table_data

    def generate_excel(
        self,
        output_path: str,
        include_formulas: bool = True,
        sheet_names: Optional[List[str]] = None
    ) -> str:
        """
        生成 Excel 报告

        Args:
            output_path: 输出路径
            include_formulas: 是否包含公式
            sheet_names: 工作表名称列表

        Returns:
            str: 输出路径
        """
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            for i, section in enumerate(self._sections):
                sheet_name = sheet_names[i] if sheet_names else f"Section_{i+1}"[:31]

                if section['data'] is not None and isinstance(section['data'], pd.DataFrame):
                    section['data'].to_excel(
                        writer,
                        sheet_name=sheet_name,
                        index=False
                    )

                    # 添加格式
                    worksheet = writer.sheets[sheet_name]

                    # 设置列宽
                    for idx, col in enumerate(section['data'].columns):
                        max_len = max(
                            section['data'][col].astype(str).map(len).max(),
                            len(col)
                        ) + 2
                        worksheet.column_dimensions[chr(65 + idx)].width = min(max_len, 50)

                    # 添加公式
                    if include_formulas:
                        numeric_cols = section['data'].select_dtypes(
                            include=[np.number]
                        ).columns
                        for col in numeric_cols:
                            col_idx = section['data'].columns.get_loc(col)
                            row_num = len(section['data']) + 2
                            col_letter = chr(65 + col_idx)
                            worksheet[f'{col_letter}{row_num}'] = f'=SUM({col_letter}2:{col_letter}{row_num-1})'

            # 添加摘要工作表
            summary_data = self._generate_summary_data()
            if summary_data is not None:
                summary_data.to_excel(writer, sheet_name='Summary', index=False)

        return output_path

    def generate_html(
        self,
        output_path: str,
        include_interactive: bool = True
    ) -> str:
        """
        生成 HTML 报告

        Args:
            output_path: 输出路径
            include_interactive: 是否包含交互式图表

        Returns:
            str: 输出路径
        """
        html_content = self._generate_html_report(include_interactive)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        return output_path

    def _generate_html_report(
        self,
        include_interactive: bool = True
    ) -> str:
        """生成 HTML 报告内容"""
        # 收集所有 Plotly 图表的 JSON
        charts_json = {}
        if include_interactive:
            for chart_id, chart_info in self._charts.items():
                import json
                charts_json[chart_id] = chart_info['chart'].to_json()

        html = f"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{self.config.title}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            line-height: 1.6;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            color: {self.config.color_scheme['text']};
        }}
        .header {{
            text-align: center;
            padding: 40px 0;
            border-bottom: 3px solid {self.config.color_scheme['primary']};
            margin-bottom: 30px;
        }}
        .title {{
            font-size: 2.5em;
            color: {self.config.color_scheme['primary']};
            margin: 0;
        }}
        .subtitle {{
            font-size: 1.3em;
            color: {self.config.color_scheme['secondary']};
            margin: 10px 0;
        }}
        .meta {{
            color: #666;
            font-size: 0.9em;
        }}
        .section {{
            margin: 30px 0;
            padding: 20px;
            background: #f8f9fa;
            border-radius: 8px;
        }}
        .section-title {{
            font-size: 1.5em;
            color: {self.config.color_scheme['primary']};
            border-bottom: 2px solid {self.config.color_scheme['accent']};
            padding-bottom: 10px;
        }}
        .data-table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        .data-table th, .data-table td {{
            border: 1px solid #ddd;
            padding: 10px;
            text-align: left;
        }}
        .data-table th {{
            background: {self.config.color_scheme['primary']};
            color: white;
        }}
        .data-table tr:nth-child(even) {{
            background: #f5f5f5;
        }}
        .chart-container {{
            margin: 20px 0;
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .kpi-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .kpi-card {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .kpi-value {{
            font-size: 2em;
            font-weight: bold;
            color: {self.config.color_scheme['primary']};
        }}
        .kpi-label {{
            color: #666;
            margin-top: 5px;
        }}
    </style>
"""
        if include_interactive and charts_json:
            html += """
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
"""

        html += """
</head>
<body>
    <div class="header">
"""
        html += f"        <h1 class='title'>{self.config.title}</h1>\n"
        if self.config.subtitle:
            html += f"        <h2 class='subtitle'>{self.config.subtitle}</h2>\n"
        html += f"""        <p class='meta'>Generated: {self.config.date} | By: {self.config.author}</p>
    </div>
"""

        # 各章节
        for section in self._sections:
            html += f"""
    <div class='section'>
        <h2 class='section-title'>{section['title']}</h2>
        <p>{section['content']}</p>
"""

            # 添加数据表格
            if section['data'] is not None and isinstance(section['data'], pd.DataFrame):
                html += "        <table class='data-table'>\n"
                html += "            <thead>\n                <tr>\n"
                for col in section['data'].columns:
                    html += f"                    <th>{col}</th>\n"
                html += "                </tr>\n            </thead>\n"
                html += "            <tbody>\n"
                for _, row in section['data'].head(10).iterrows():
                    html += "                <tr>\n"
                    for val in row.values:
                        html += f"                    <td>{val}</td>\n"
                    html += "                </tr>\n"
                html += "            </tbody>\n        </table>\n"

            html += "    </div>\n"

        # 添加图表
        if include_interactive and self._charts:
            html += """
    <div class='section'>
        <h2 class='section-title'>Charts</h2>
"""
            for chart_id, chart_info in self._charts.items():
                html += f"""
        <div class='chart-container' id='{chart_id}'></div>
        <p><em>{chart_info['caption']}</em></p>
"""

        html += """
    </div>
"""

        html += """
</body>
</html>
"""
        return html

    def _generate_summary_data(self) -> Optional[pd.DataFrame]:
        """生成摘要数据"""
        summary = []
        for section in self._sections:
            if section['data'] is not None and isinstance(section['data'], pd.DataFrame):
                summary.append({
                    'Section': section['title'],
                    'Rows': len(section['data']),
                    'Columns': len(section['data'].columns)
                })

        if summary:
            return pd.DataFrame(summary)
        return None

    def generate_all(
        self,
        output_dir: str,
        base_name: str,
        formats: List[str] = ['excel', 'html']
    ) -> Dict[str, str]:
        """
        生成多种格式的报告

        Args:
            output_dir: 输出目录
            base_name: 基础文件名
            formats: 格式列表

        Returns:
            Dict: 生成的文件路径
        """
        generated_files = {}

        for fmt in formats:
            if fmt == 'excel':
                path = f"{output_dir}/{base_name}.xlsx"
                self.generate_excel(path)
                generated_files['excel'] = path
            elif fmt == 'html':
                path = f"{output_dir}/{base_name}.html"
                self.generate_html(path)
                generated_files['html'] = path
            elif fmt == 'pdf':
                path = f"{output_dir}/{base_name}.pdf"
                self.generate_pdf(path)
                generated_files['pdf'] = path

        return generated_files

    def clear(self) -> 'ReportGenerator':
        """清空报告内容"""
        self._sections = []
        self._charts = {}
        return self


def create_customer_report(
    rfm_df: pd.DataFrame,
    clustering_result: Optional[Any] = None,
    ltv_predictions: Optional[pd.DataFrame] = None,
    churn_predictions: Optional[pd.DataFrame] = None,
    output_dir: str = 'reports'
) -> Dict[str, str]:
    """
    便捷函数：创建客户分析报告

    Args:
        rfm_df: RFM 分析结果
        clustering_result: 聚类结果
        ltv_predictions: LTV 预测
        churn_predictions: 流失预测
        output_dir: 输出目录

    Returns:
        Dict: 生成的文件路径
    """
    generator = ReportGenerator(
        title='Customer Analytics Report',
        subtitle='Comprehensive Customer Analysis',
        color_scheme='professional'
    )

    # RFM 分析章节
    generator.add_section(
        title='RFM Analysis',
        content='Customer segmentation based on Recency, Frequency, and Monetary value.',
        data=rfm_df.groupby('segment').agg({
            'customer_id': 'count',
            'monetary': 'mean',
            'frequency': 'mean'
        }).reset_index()
    )

    # 聚类分析章节
    if clustering_result is not None:
        generator.add_section(
            title='Customer Clustering',
            content='Unsupervised learning based customer grouping.',
            data=clustering_result.cluster_summary
        )

    # LTV 预测章节
    if ltv_predictions is not None:
        ltv_summary = ltv_predictions[[
            col for col in ltv_predictions.columns if 'ltv' in col.lower()
        ]].describe()
        generator.add_section(
            title='Lifetime Value Prediction',
            content='Predicted customer lifetime value for the next 12 months.',
            data=ltv_summary
        )

    # 流失预测章节
    if churn_predictions is not None:
        churn_summary = churn_predictions.groupby('churn_risk').agg({
            'customer_id': 'count',
            'churn_probability': 'mean'
        }).reset_index()
        generator.add_section(
            title='Churn Prediction',
            content='Customer churn risk assessment and prediction.',
            data=churn_summary
        )

    # 生成报告
    import os
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    files = generator.generate_all(
        output_dir=output_dir,
        base_name=f'customer_report_{timestamp}',
        formats=['excel', 'html']
    )

    return files


if __name__ == "__main__":
    # 测试
    print("报告生成模块测试")
    print("=" * 50)

    # 生成测试数据
    test_df = pd.DataFrame({
        'segment': ['Champions', 'Loyal', 'New', 'At Risk'],
        'customers': [150, 350, 100, 100],
        'revenue': [75000, 70000, 10000, 15000],
        'avg_ltv': [5000, 2000, 1000, 1500]
    })

    # 创建报告
    generator = ReportGenerator(
        title='Customer Analytics Report',
        subtitle='Q1 2026 Analysis',
        color_scheme='blue'
    )

    generator.add_section(
        title='Executive Summary',
        content='This report provides comprehensive analysis of customer segments, behavior patterns, and predictions.',
        data=test_df
    )

    generator.add_section(
        title='Segment Analysis',
        content='Detailed breakdown of customer segments and their characteristics.',
        data=test_df
    )

    # 生成 Excel
    output_path = generator.generate_excel('reports/test_report.xlsx')
    print(f"Excel 报告：{output_path}")

    # 生成 HTML
    output_path = generator.generate_html('reports/test_report.html')
    print(f"HTML 报告：{output_path}")
