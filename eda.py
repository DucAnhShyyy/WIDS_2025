import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_correlation_heatmap(df, threshold=0.8, figsize=(12, 10), cmap='coolwarm'):
    """
    Vẽ heatmap thể hiện correlation của dữ liệu và in ra các feature có correlation cao hơn threshold
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame cần phân tích
    threshold : float, default=0.8
        Ngưỡng để xác định correlation cao
    figsize : tuple, default=(12, 10)
        Kích thước của biểu đồ
    cmap : str, default='coolwarm'
        Bảng màu cho heatmap
    """
    # Tính correlation matrix
    corr_matrix = df.corr()
    
    # Vẽ heatmap
    plt.figure(figsize=figsize)
    sns.heatmap(corr_matrix, annot=True, cmap=cmap, vmin=-1, vmax=1, center=0, fmt='.2f')
    plt.title('Correlation Heatmap')
    plt.tight_layout()
    plt.show()
    
    # Tìm và in ra các cặp feature có correlation cao hơn threshold
    corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            if abs(corr_matrix.iloc[i, j]) > threshold:
                corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_matrix.iloc[i, j]))
    
    if corr_pairs:
        print(f"Các cặp feature có correlation cao hơn {threshold}:")
        for feature1, feature2, corr in corr_pairs:
            print(f"{feature1} - {feature2}: {corr:.4f}")
    else:
        print(f"Không có cặp feature nào có correlation cao hơn {threshold}")

def plot_boxplots(df1, df2=None, figsize=(15, 10), common_columns=None, n_cols=3):
    """
    Vẽ boxplot cho từng cột của một hoặc hai DataFrame
    
    Parameters:
    -----------
    df1 : pandas.DataFrame
        DataFrame thứ nhất
    df2 : pandas.DataFrame, optional
        DataFrame thứ hai, nếu có
    figsize : tuple, default=(15, 10)
        Kích thước của biểu đồ
    common_columns : list, optional
        Danh sách các cột cần vẽ. Nếu None, sẽ vẽ tất cả các cột số
    n_cols : int, default=3
        Số cột trong grid layout
    """
    # Nếu common_columns không được cung cấp, lấy tất cả các cột số
    if common_columns is None:
        numeric_cols1 = df1.select_dtypes(include=[np.number]).columns.tolist()
        if df2 is not None:
            numeric_cols2 = df2.select_dtypes(include=[np.number]).columns.tolist()
            common_columns = [col for col in numeric_cols1 if col in numeric_cols2]
        else:
            common_columns = numeric_cols1
    
    # Tính số hàng cần thiết
    n_rows = int(np.ceil(len(common_columns) / n_cols))
    
    # Tạo figure và axes
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if n_rows * n_cols > 1 else [axes]
    
    # Vẽ boxplot cho từng cột
    for i, col in enumerate(common_columns):
        if i < len(axes):
            if df2 is not None:
                # Chuẩn bị dữ liệu cho boxplot
                df1_data = df1[col].dropna()
                df2_data = df2[col].dropna()
                
                # Vẽ boxplot cho cả hai DataFrame
                box_data = [df1_data, df2_data]
                axes[i].boxplot(box_data, labels=['DF1', 'DF2'], patch_artist=True)
                axes[i].set_title(f'Boxplot: {col}')
                axes[i].set_ylabel(col)
            else:
                # Vẽ boxplot cho một DataFrame
                sns.boxplot(y=df1[col], ax=axes[i])
                axes[i].set_title(f'Boxplot: {col}')
    
    # Ẩn các axes dư thừa
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    
    plt.tight_layout()
    plt.show()

def plot_histograms(df1, df2=None, figsize=(15, 10), common_columns=None, n_cols=3, bins=30, alpha=0.6):
    """
    Vẽ histogram cho từng cột của một hoặc hai DataFrame
    
    Parameters:
    -----------
    df1 : pandas.DataFrame
        DataFrame thứ nhất
    df2 : pandas.DataFrame, optional
        DataFrame thứ hai, nếu có
    figsize : tuple, default=(15, 10)
        Kích thước của biểu đồ
    common_columns : list, optional
        Danh sách các cột cần vẽ. Nếu None, sẽ vẽ tất cả các cột số
    n_cols : int, default=3
        Số cột trong grid layout
    bins : int, default=30
        Số bins cho histogram
    alpha : float, default=0.6
        Độ trong suốt của histogram
    """
    # Nếu common_columns không được cung cấp, lấy tất cả các cột số
    if common_columns is None:
        numeric_cols1 = df1.select_dtypes(include=[np.number]).columns.tolist()
        if df2 is not None:
            numeric_cols2 = df2.select_dtypes(include=[np.number]).columns.tolist()
            common_columns = [col for col in numeric_cols1 if col in numeric_cols2]
        else:
            common_columns = numeric_cols1
    
    # Tính số hàng cần thiết
    n_rows = int(np.ceil(len(common_columns) / n_cols))
    
    # Tạo figure và axes
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if n_rows * n_cols > 1 else [axes]
    
    # Vẽ histogram cho từng cột
    for i, col in enumerate(common_columns):
        if i < len(axes):
            if df2 is not None:
                # Vẽ histogram cho cả hai DataFrame
                axes[i].hist(df1[col].dropna(), bins=bins, alpha=alpha, label='DF1', color='blue')
                axes[i].hist(df2[col].dropna(), bins=bins, alpha=alpha, label='DF2', color='orange')
                axes[i].legend()
            else:
                # Vẽ histogram cho một DataFrame
                axes[i].hist(df1[col].dropna(), bins=bins, alpha=alpha, color='blue')
            
            axes[i].set_title(f'Histogram: {col}')
            axes[i].set_xlabel(col)
            axes[i].set_ylabel('Frequency')
    
    # Ẩn các axes dư thừa
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    
    plt.tight_layout()
    plt.show()
