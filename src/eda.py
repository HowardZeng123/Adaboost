import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def show_overview(df):
    """
    Hiển thị thông tin tổng quan của dữ liệu (các cột, kiểu dữ liệu, số lượng dòng không null).
    """
    print("=" * 55)
    print("THÔNG TIN TỔNG QUAN BỘ DỮ LIỆU")
    print("=" * 55)
    df.info()

def show_statistics(df):
    """
    Hiển thị thống kê mô tả của các đặc trưng số.
    """
    print("Thống kê mô tả các đặc trưng số:")
    return df.describe().round(3)

def check_missing(df):
    """
    Kiểm tra xem dữ liệu có giá trị thiếu hay không và in ra các cột bị thiếu.
    """
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if len(missing) == 0:
        print("✅ Không có giá trị thiếu trong bộ dữ liệu!")
    else:
        print("⚠️ Các cột có giá trị thiếu:")
        print(missing)

def plot_diagnosis_distribution(df):
    """
    Vẽ biểu đồ Bar Chart và Pie Chart để trực quan hóa phân bố nhãn diagnosis.
    Hỗ trợ cả định dạng gốc ('M'/'B') và định dạng mã hóa (+1/-1).
    """
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    fig.suptitle("Phân bố nhãn trong bộ dữ liệu Breast Cancer", fontsize=14, fontweight='bold')

    counts = df['diagnosis'].value_counts()
    
    # Kiểm tra nhãn là dạng chuỗi hay số đã mã hóa
    if 'M' in counts.index or 'B' in counts.index:
        labels_map = {'M': 'Ác tính (M)', 'B': 'Lành tính (B)'}
        colors = ['#E53935', '#1E88E5']
        index_labels = [labels_map.get(k, k) for k in counts.index]
        pie_labels = index_labels
    else:
        labels_map = {1: 'Ác tính (+1)', -1: 'Lành tính (-1)'}
        colors = ['#E53935', '#1E88E5']
        index_labels = [labels_map.get(k, k) for k in counts.index]
        pie_labels = index_labels

    # Biểu đồ cột (Bar chart)
    bars = axes[0].bar(index_labels, counts.values,
                       color=colors, edgecolor='black', alpha=0.85, width=0.5)
    axes[0].set_title('Bar Chart', fontweight='bold')
    axes[0].set_ylabel('Số lượng mẫu')
    for bar, val in zip(bars, counts.values):
        axes[0].text(bar.get_x() + bar.get_width()/2, val + 3, str(val),
                     ha='center', fontweight='bold', fontsize=12)
    axes[0].set_ylim(0, counts.max() * 1.15)
    axes[0].grid(axis='y', alpha=0.3)

    # Biểu đồ tròn (Pie chart)
    axes[1].pie(counts.values,
                labels=pie_labels,
                colors=colors, autopct='%1.1f%%',
                startangle=90, explode=[0.05, 0.05],
                wedgeprops={'edgecolor': 'black'},
                textprops={'fontsize': 12})
    axes[1].set_title('Pie Chart', fontweight='bold')

    plt.tight_layout()
    plt.show()

    total = len(df)
    if 'M' in counts.index or 'B' in counts.index:
        print(f"\n📊 Tổng số mẫu   : {total}")
        print(f"   Ác tính (M)   : {counts.get('M', 0)} mẫu ({counts.get('M', 0)/total*100:.1f}%)")
        print(f"   Lành tính (B) : {counts.get('B', 0)} mẫu ({counts.get('B', 0)/total*100:.1f}%)")
    else:
        print(f"\n📊 Tổng số mẫu   : {total}")
        print(f"   Ác tính (+1)  : {counts.get(1, 0)} mẫu ({counts.get(1, 0)/total*100:.1f}%)")
        print(f"   Lành tính (-1): {counts.get(-1, 0)} mẫu ({counts.get(-1, 0)/total*100:.1f}%)")

def plot_feature_histograms(df, feature_cols=None):
    """
    Vẽ histogram cho 10 đặc trưng đầu tiên theo hai nhãn ác tính/lành tính để quan sát sự khác biệt phân bố.
    """
    if feature_cols is None:
        feature_cols = [c for c in df.columns if c not in ['id', 'diagnosis', 'Unnamed: 32']]
        
    top10 = feature_cols[:10]
    fig, axes = plt.subplots(2, 5, figsize=(18, 7))
    fig.suptitle('Phân bố 10 đặc trưng đầu tiên theo nhãn', fontsize=14, fontweight='bold')

    unique_labels = df['diagnosis'].unique()
    if 'M' in unique_labels or 'B' in unique_labels:
        label_configs = [('M', '#E53935', 'Ác tính'), ('B', '#1E88E5', 'Lành tính')]
    else:
        label_configs = [(1, '#E53935', 'Ác tính (+1)'), (-1, '#1E88E5', 'Lành tính (-1)')]

    for ax, col in zip(axes.flatten(), top10):
        for label, color, name in label_configs:
            ax.hist(df[df['diagnosis'] == label][col],
                    bins=25, alpha=0.6, color=color, label=name, edgecolor='none')
        ax.set_title(col, fontsize=10, fontweight='bold')
        ax.set_xlabel('')
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.show()

def plot_correlation_heatmap(df, top_n=15):
    """
    Vẽ ma trận tương quan (heatmap) cho top N đặc trưng tương quan mạnh nhất đến diagnosis.
    """
    df_corr = df.copy()
    if 'id' in df_corr.columns:
        df_corr.drop(columns=['id'], errors='ignore', inplace=True)
    if 'Unnamed: 32' in df_corr.columns:
        df_corr.drop(columns=['Unnamed: 32'], errors='ignore', inplace=True)

    if df_corr['diagnosis'].dtype == object or df_corr['diagnosis'].dtype == str:
        df_corr['diagnosis'] = (df_corr['diagnosis'] == 'M').astype(int)
    else:
        df_corr['diagnosis'] = (df_corr['diagnosis'] == 1).astype(int)

    top_features = df_corr.corr()['diagnosis'].abs().nlargest(top_n).index.tolist()
    corr_matrix = df_corr[top_features].corr()

    plt.figure(figsize=(13, 10))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f',
                cmap='RdYlBu_r', center=0, linewidths=0.4,
                annot_kws={'size': 8})
    plt.title(f'Ma trận tương quan – Top {top_n} đặc trưng liên quan nhất đến diagnosis',
              fontsize=13, fontweight='bold', pad=12)
    plt.tight_layout()
    plt.show()
