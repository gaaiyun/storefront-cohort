"""
样本数据生成器 - Sample Data Generator
生成 realistic customer transaction data for testing
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random


def generate_customer_data(n_customers: int = 10000, seed: int = 42) -> pd.DataFrame:
    """
    生成客户交易数据

    Args:
        n_customers: 客户数量
        seed: 随机种子

    Returns:
        pd.DataFrame: 客户交易数据
    """
    np.random.seed(seed)
    random.seed(seed)

    # 基础日期范围
    end_date = datetime(2026, 3, 1)
    start_date = end_date - timedelta(days=730)  # 2 年数据

    customers = []
    transactions = []

    # 客户分群参数
    segments = {
        'high_value': {'weight': 0.15, 'freq_mean': 20, 'freq_std': 5, 'amount_mean': 500, 'amount_std': 150},
        'medium_value': {'weight': 0.35, 'freq_mean': 10, 'freq_std': 3, 'amount_mean': 200, 'amount_std': 80},
        'low_value': {'weight': 0.30, 'freq_mean': 4, 'freq_std': 2, 'amount_mean': 80, 'amount_std': 30},
        'at_risk': {'weight': 0.10, 'freq_mean': 8, 'freq_std': 3, 'amount_mean': 150, 'amount_std': 50},
        'new': {'weight': 0.10, 'freq_mean': 3, 'freq_std': 1, 'amount_mean': 120, 'amount_std': 40}
    }

    customer_id = 0

    for segment_name, params in segments.items():
        n_segment = int(n_customers * params['weight'])

        for _ in range(n_segment):
            customer_id += 1

            # 客户属性
            if segment_name == 'new':
                registration_date = end_date - timedelta(days=random.randint(30, 180))
            elif segment_name == 'at_risk':
                registration_date = end_date - timedelta(days=random.randint(400, 600))
            else:
                registration_date = end_date - timedelta(days=random.randint(200, 700))

            age = int(np.random.normal(35, 12))
            age = max(18, min(75, age))

            gender = random.choice(['M', 'F', 'Other'])

            # 收入分布
            if segment_name == 'high_value':
                income = np.random.lognormal(11, 0.5)
            elif segment_name == 'medium_value':
                income = np.random.lognormal(10, 0.4)
            else:
                income = np.random.lognormal(9.5, 0.4)

            income = int(income / 1000) * 1000  # 取整到千
            income = max(20000, min(500000, income))

            region = random.choice(['North', 'South', 'East', 'West', 'Central'])
            city_tier = random.choices([1, 2, 3], weights=[0.2, 0.35, 0.45])[0]

            # 渠道偏好
            channel_pref = random.choices(
                ['Online', 'Offline', 'Omnichannel'],
                weights=[0.5, 0.2, 0.3]
            )[0]

            # 生成交易
            if segment_name == 'at_risk':
                # 最近购买较少
                last_purchase_days_ago = random.randint(90, 180)
                n_transactions = max(1, int(np.random.normal(params['freq_mean'] * 0.3, params['freq_std'])))
            elif segment_name == 'new':
                last_purchase_days_ago = random.randint(1, 30)
                n_transactions = max(1, int(np.random.normal(params['freq_mean'], params['freq_std'])))
            else:
                last_purchase_days_ago = random.randint(1, 60)
                n_transactions = max(1, int(np.random.normal(params['freq_mean'], params['freq_std'])))

            avg_amount = params['amount_mean']
            amount_std = params['amount_std']

            customer_transactions = []
            total_revenue = 0

            for txn_idx in range(n_transactions):
                # 交易日期
                if txn_idx == n_transactions - 1:
                    # 最后一次购买
                    txn_date = end_date - timedelta(days=last_purchase_days_ago)
                else:
                    days_before = last_purchase_days_ago + random.randint(5, 60) * (n_transactions - txn_idx)
                    days_before = min(days_before, 700)
                    txn_date = end_date - timedelta(days=days_before)

                if txn_date < registration_date:
                    txn_date = registration_date + timedelta(days=random.randint(1, 30))

                # 交易金额
                amount = max(10, np.random.normal(avg_amount, amount_std))
                if segment_name == 'high_value':
                    amount *= random.choice([1.0, 1.0, 1.0, 1.5, 2.0])  # 偶尔有大单

                amount = round(amount, 2)
                total_revenue += amount

                # 交易详情
                category = random.choices(
                    ['Electronics', 'Clothing', 'Home', 'Beauty', 'Sports', 'Books', 'Food'],
                    weights=[0.2, 0.15, 0.15, 0.12, 0.1, 0.08, 0.2]
                )[0]

                payment_method = random.choices(
                    ['Credit Card', 'Debit Card', 'Digital Wallet', 'Bank Transfer'],
                    weights=[0.35, 0.25, 0.3, 0.1]
                )[0]

                discount = round(random.uniform(0, 0.2) * amount, 2) if random.random() < 0.3 else 0

                customer_transactions.append({
                    'transaction_id': f"TXN{customer_id:05d}{txn_idx:03d}",
                    'customer_id': f"CUST{customer_id:05d}",
                    'transaction_date': txn_date.strftime('%Y-%m-%d'),
                    'amount': amount,
                    'discount': discount,
                    'net_amount': round(amount - discount, 2),
                    'category': category,
                    'payment_method': payment_method,
                    'quantity': random.randint(1, 5)
                })

            # 客户记录
            customers.append({
                'customer_id': f"CUST{customer_id:05d}",
                'registration_date': registration_date.strftime('%Y-%m-%d'),
                'age': age,
                'gender': gender,
                'income': income,
                'region': region,
                'city_tier': city_tier,
                'channel_pref': channel_pref,
                'total_transactions': len(customer_transactions),
                'total_revenue': round(total_revenue, 2),
                'avg_transaction_value': round(total_revenue / len(customer_transactions), 2) if customer_transactions else 0,
                'days_since_first_purchase': (end_date - registration_date).days,
                'segment': segment_name,
                'email_subscribed': random.choice([True, False]),
                'loyalty_member': random.choices([True, False], weights=[0.6, 0.4])[0],
                'loyalty_points': int(total_revenue * 0.01 * random.uniform(0.8, 1.2)),
                'last_login_date': (end_date - timedelta(days=random.randint(0, 30))).strftime('%Y-%m-%d')
            })

            transactions.extend(customer_transactions)

    customers_df = pd.DataFrame(customers)
    transactions_df = pd.DataFrame(transactions)

    return customers_df, transactions_df


def generate_churn_labels(customers_df: pd.DataFrame, transactions_df: pd.DataFrame) -> pd.DataFrame:
    """
    生成流失标签

    Args:
        customers_df: 客户数据
        transactions_df: 交易数据

    Returns:
        pd.DataFrame: 带流失标签的客户数据
    """
    end_date = datetime(2026, 3, 1)

    # 计算每个客户的最后购买日期
    transactions_df['transaction_date'] = pd.to_datetime(transactions_df['transaction_date'])
    last_purchase = transactions_df.groupby('customer_id')['transaction_date'].max().reset_index()
    last_purchase.columns = ['customer_id', 'last_purchase_date']

    # 合并
    customers_df = customers_df.merge(last_purchase, on='customer_id', how='left')

    # 计算流失概率特征
    customers_df['days_since_last_purchase'] = (end_date - customers_df['last_purchase_date']).dt.days
    customers_df['avg_days_between_purchases'] = customers_df['days_since_first_purchase'] / customers_df['total_transactions'].replace(0, 1)

    # 流失标签
    def assign_churn(row):
        if pd.isna(row.get('last_purchase_date')):
            return 1

        days_inactive = row['days_since_last_purchase']
        avg_gap = row['avg_days_between_purchases']

        # 流失条件：超过 2 倍平均购买间隔未购买，或超过 90 天未购买
        if days_inactive > max(90, avg_gap * 2):
            return 1
        elif days_inactive > 60 and row['total_transactions'] < 5:
            return 1
        else:
            return 0

    customers_df['churned'] = customers_df.apply(assign_churn, axis=1)

    return customers_df


if __name__ == "__main__":
    print("生成样本客户数据...")

    customers, transactions = generate_customer_data(n_customers=10000)
    customers = generate_churn_labels(customers, transactions)

    # 保存
    customers.to_csv('data/customers.csv', index=False)
    transactions.to_csv('data/transactions.csv', index=False)

    print(f"生成 {len(customers)} 个客户记录")
    print(f"生成 {len(transactions)} 条交易记录")
    print(f"流失客户比例：{customers['churned'].mean():.2%}")
    print("\n文件已保存到 data/ 目录")
