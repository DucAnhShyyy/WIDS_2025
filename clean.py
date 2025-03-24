import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.ensemble import IsolationForest
from scipy import stats

def detect_duplicates(df):
    """
    Phát hiện các dòng trùng lặp trong DataFrame.
    - Nếu trùng lặp cả dòng: xóa bỏ
    - Nếu không: chỉ ra các cột trong dòng đó trùng giá trị
    
    Args:
        df (pandas.DataFrame): DataFrame cần kiểm tra
        
    Returns:
        tuple: (DataFrame đã loại bỏ các dòng trùng lặp, dictionary chứa thông tin về duplicates)

    Explain:
        Hàm này thực hiện kiểm tra dữ liệu trùng lặp theo hai cách:
        - Trùng lặp hoàn toàn: Nếu một hàng trùng hoàn toàn với một hàng khác, nó sẽ bị loại bỏ.
        - Trùng lặp từng phần: Nếu một số cột có giá trị trùng lặp nhưng cả dòng không trùng hoàn toàn, chúng sẽ được ghi nhận vào một dictionary.
    """
    # Tạo bản sao của DataFrame: tránh làm thay đổi dữ liệu gốc khi xử lí
    df_clean = df.copy()
    
    # Tìm các dòng trùng lặp hoàn toàn
    duplicate_rows = df_clean.duplicated() # Tạo 1 Series có giá trị Boolean True cho các dòng trùng lặp hoản toàn
    full_duplicates = df_clean[duplicate_rows] # Lọc ra các dòng trùng lặp hoản toàn
    
    # Loại bỏ các dòng trùng lặp hoàn toàn
    df_clean = df_clean.drop_duplicates(keep='first').reset_index(drop=True)
    
    # Kiểm tra trùng lặp từng phần (theo cột)
    partial_duplicates = {} #Tạo dictionary đẻ lưu
    for col in df_clean.columns:
        # Tìm các giá trị trùng lặp trong cột
        value_counts = df_clean[col].value_counts()
        duplicated_values = value_counts[value_counts > 1].index.tolist() #Lọc các giá trị xuất hiện hơn 1 lần 
        
        if duplicated_values:
            # Lưu trữ các hàng có giá trị trùng lặp cho mỗi cột
            for value in duplicated_values:
                duplicate_indices = df_clean[df_clean[col] == value].index.tolist()
                # Nếu có giá trị bị trùng, lấy chỉ mục (index) của các hàng có giá trị đó.
                if len(duplicate_indices) > 1:
                    if col not in partial_duplicates:
                        partial_duplicates[col] = {}
                    partial_duplicates[col][value] = duplicate_indices
    
    duplicates_info = {
        'full_duplicates': full_duplicates,
        'partial_duplicates': partial_duplicates
    }
    
    return df_clean, duplicates_info

def impute_missing_values(df, method='median', columns=None, k=5):
    """
    Điền các giá trị thiếu trong DataFrame.
    
    Args:
        df (pandas.DataFrame): DataFrame cần điền missing values
        method (str): Phương pháp điền missing values ('mean', 'median', 'mode', 'knn', 'regression', 'ffill', 'bfill')
        columns (list): Danh sách các cột cần điền missing values. Nếu None, tất cả các cột có missing values sẽ được điền.
        k (int): Số lượng neighbors trong KNN imputation
        
    Returns:
        pandas.DataFrame: DataFrame đã được điền missing values
    """
    # Nếu không chỉ định cột, áp dụng cho tất cả các cột có missing values
    if columns is None:
        columns = df.columns[df.isnull().any()].tolist()
    
    # Sao chép DataFrame để tránh thay đổi dữ liệu gốc
    df_imputed = df.copy()
    
    for col in columns:
        # Kiểm tra xem cột có missing values không
        if df_imputed[col].isnull().sum() > 0:
            # Kiểm tra kiểu dữ liệu của cột
            is_numeric = pd.api.types.is_numeric_dtype(df_imputed[col])
            
            if method == 'mean' and is_numeric:
                df_imputed[col] = df_imputed[col].fillna(df_imputed[col].mean())
            
            elif method == 'median' and is_numeric:
                df_imputed[col] = df_imputed[col].fillna(df_imputed[col].median())
            
            elif method == 'mode':
                # Mode có thể áp dụng cho cả numeric và categorical
                mode_value = df_imputed[col].mode()[0]
                df_imputed[col] = df_imputed[col].fillna(mode_value)
            
            elif method == 'knn' and is_numeric:
                # Chỉ áp dụng KNN cho các cột numeric
                numeric_cols = df_imputed.select_dtypes(include=[np.number]).columns.tolist()
                
                if col in numeric_cols:
                    # Tạo một bản sao của các cột numeric
                    numeric_data = df_imputed[numeric_cols].copy()
                    
                    # Khởi tạo KNN imputer
                    imputer = KNNImputer(n_neighbors=k)
                    
                    # Fit và transform dữ liệu
                    imputed_data = imputer.fit_transform(numeric_data)
                    
                    # Cập nhật giá trị trong DataFrame
                    df_imputed[numeric_cols] = pd.DataFrame(imputed_data, columns=numeric_cols)
            
            elif method == 'regression' and is_numeric:
                # Chỉ áp dụng Regression cho các cột numeric
                df_temp = df_imputed.select_dtypes(include=[np.number]).copy()
                
                # Tạo masks cho dữ liệu huấn luyện và dữ liệu cần dự đoán
                train_mask = df_temp[col].notna()
                predict_mask = df_temp[col].isna()
                
                # Nếu có đủ dữ liệu để huấn luyện và dự đoán
                if train_mask.sum() > 0 and predict_mask.sum() > 0:
                    # Loại bỏ cột đích khỏi features
                    features = df_temp.drop(columns=[col])
                    
                    # Tạo X_train và y_train
                    X_train = features.loc[train_mask].dropna(axis=1)
                    y_train = df_temp.loc[train_mask, col]
                    
                    # Nếu có đủ features để huấn luyện
                    if X_train.shape[1] > 0:
                        # Tạo và huấn luyện mô hình regression
                        model = LinearRegression()
                        model.fit(X_train, y_train)
                        
                        # Dự đoán giá trị missing
                        X_predict = features.loc[predict_mask, X_train.columns]
                        predicted_values = model.predict(X_predict)
                        
                        # Điền giá trị dự đoán vào DataFrame
                        df_imputed.loc[predict_mask, col] = predicted_values
            
            elif method == 'ffill':
                # Forward fill
                df_imputed[col] = df_imputed[col].fillna(method='ffill')
                
            elif method == 'bfill':
                # Backward fill
                df_imputed[col] = df_imputed[col].fillna(method='bfill')
    
    return df_imputed

def detect_outliers(df, method='zscore', threshold=3):
    """
    Phát hiện outliers trong dữ liệu.
    
    Args:
        df (pandas.DataFrame): DataFrame cần kiểm tra
        method (str): Phương pháp phát hiện outliers ('zscore', 'iqr', 'isolation_forest')
        threshold (float): Ngưỡng để xác định outliers
        
    Returns:
        dict: Dictionary chứa outliers cho mỗi cột
    """
    outliers = {}
    
    # Chỉ áp dụng phát hiện outliers cho các cột numeric
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    for col in numeric_cols:
        # Lấy dữ liệu không phải NaN
        data = df[col].dropna()
        
        if method == 'zscore':
            # Sử dụng Z-score để phát hiện outliers
            z_scores = np.abs(stats.zscore(data))
            outlier_indices = np.where(z_scores > threshold)[0]
            outlier_values = data.iloc[outlier_indices]
            
            if len(outlier_indices) > 0:
                outliers[col] = {
                    'indices': data.index[outlier_indices].tolist(),
                    'values': outlier_values.tolist()
                }
        
        elif method == 'iqr':
            # Sử dụng IQR để phát hiện outliers
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            
            outlier_mask = (data < lower_bound) | (data > upper_bound)
            outlier_indices = np.where(outlier_mask)[0]
            outlier_values = data.iloc[outlier_indices]
            
            if len(outlier_indices) > 0:
                outliers[col] = {
                    'indices': data.index[outlier_indices].tolist(),
                    'values': outlier_values.tolist(),
                    'bounds': {
                        'lower': lower_bound,
                        'upper': upper_bound
                    }
                }
        
        elif method == 'isolation_forest':
            # Sử dụng Isolation Forest để phát hiện outliers
            data_reshape = data.values.reshape(-1, 1)
            
            # Khởi tạo và huấn luyện mô hình
            iso_forest = IsolationForest(contamination=threshold/100, random_state=42)
            predictions = iso_forest.fit_predict(data_reshape)
            
            # -1 đại diện cho outliers
            outlier_mask = predictions == -1
            outlier_indices = np.where(outlier_mask)[0]
            outlier_values = data.iloc[outlier_indices]
            
            if len(outlier_indices) > 0:
                outliers[col] = {
                    'indices': data.index[outlier_indices].tolist(),
                    'values': outlier_values.tolist()
                }
    
    return outliers

def classify_columns(df):
    """
    Phân loại các cột thành numerical và categorical.
    
    Args:
        df (pandas.DataFrame): DataFrame cần phân loại
        
    Returns:
        dict: Dictionary chứa thông tin về loại dữ liệu của mỗi cột
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    
    # Phân loại các cột numeric có ít giá trị riêng biệt thành categorical
    for col in numeric_cols.copy():
        unique_values = df[col].nunique()
        if unique_values <= 10:  # Ngưỡng cho số lượng giá trị riêng biệt
            categorical_cols.append(col)
            numeric_cols.remove(col)
    
    column_types = {
        'numeric': numeric_cols,
        'categorical': categorical_cols
    }
    
    return column_types

def scale_features(df, method='standard', columns=None):
    """
    Đưa giá trị của các cột về cùng một scale.
    
    Args:
        df (pandas.DataFrame): DataFrame cần scaling
        method (str): Phương pháp scaling ('standard', 'minmax')
        columns (list): Danh sách các cột cần scaling. Nếu None, tất cả các cột numeric sẽ được scaling.
        
    Returns:
        pandas.DataFrame: DataFrame đã được scaling
    """
    # Nếu không chỉ định cột, áp dụng cho tất cả các cột numeric
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Sao chép DataFrame để tránh thay đổi dữ liệu gốc
    df_scaled = df.copy()
    
    if method == 'standard':
        # Standard scaling (z-score normalization)
        scaler = StandardScaler()
        df_scaled[columns] = scaler.fit_transform(df_scaled[columns])
    
    elif method == 'minmax':
        # Min-Max scaling
        scaler = MinMaxScaler()
        df_scaled[columns] = scaler.fit_transform(df_scaled[columns])
    
    return df_scaled

def encode_categorical_data(df, method='label', columns=None):
    """
    Mã hóa dữ liệu categorical.
    
    Args:
        df (pandas.DataFrame): DataFrame cần encoding
        method (str): Phương pháp encoding ('label', 'onehot')
        columns (list): Danh sách các cột cần encoding. Nếu None, tất cả các cột categorical sẽ được encoding.
        
    Returns:
        pandas.DataFrame: DataFrame đã được encoding
    """
    # Nếu không chỉ định cột, áp dụng cho tất cả các cột categorical
    if columns is None:
        column_types = classify_columns(df)
        columns = column_types['categorical']
    
    # Sao chép DataFrame để tránh thay đổi dữ liệu gốc
    df_encoded = df.copy()
    
    if method == 'label':
        # Label Encoding
        for col in columns:
            if col in df_encoded.columns:
                le = LabelEncoder()
                # Chỉ áp dụng cho các giá trị không phải NaN
                non_nan_mask = df_encoded[col].notna()
                df_encoded.loc[non_nan_mask, col] = le.fit_transform(df_encoded.loc[non_nan_mask, col])
    
    elif method == 'onehot':
        # One-Hot Encoding
        for col in columns:
            if col in df_encoded.columns:
                # Tạo dummy variables
                dummies = pd.get_dummies(df_encoded[col], prefix=col, drop_first=False)
                
                # Thêm dummy variables vào DataFrame
                df_encoded = pd.concat([df_encoded, dummies], axis=1)
                
                # Xóa cột gốc
                df_encoded.drop(columns=[col], inplace=True)
    
    return df_encoded

def clean_data_pipeline(df, config=None):
    """
    Pipeline làm sạch dữ liệu tự động.
    
    Args:
        df (pandas.DataFrame): DataFrame cần làm sạch
        config (dict): Cấu hình cho pipeline (optional)
        
    Returns:
        tuple: (DataFrame đã làm sạch, dictionary chứa thông tin về quá trình làm sạch)
    """
    # Tạo bản sao của DataFrame
    df_clean = df.copy()
    
    # Cấu hình mặc định
    if config is None:
        config = {
            'remove_duplicates': True,
            'impute_method': 'median',
            'detect_outliers': True,
            'outlier_method': 'zscore',
            'outlier_threshold': 3,
            'scale_features': True,
            'scale_method': 'standard',
            'encode_categorical': True,
            'encode_method': 'label'
        }
    
    # Tạo dictionary để lưu trữ thông tin về quá trình làm sạch
    cleaning_info = {
        'original_shape': df.shape
    }
    
    # Phát hiện và xóa duplicates
    if config.get('remove_duplicates', True):
        df_clean, duplicates_info = detect_duplicates(df_clean)
        cleaning_info['duplicates'] = duplicates_info
    
    # Tìm missing values
    missing_info = find_missing_values(df_clean)
    cleaning_info['missing_values'] = missing_info
    
    # Impute missing values
    if config.get('impute_method'):
        df_clean = impute_missing_values(
            df_clean, 
            method=config['impute_method'],
            k=config.get('knn_neighbors', 5)
        )
    
    # Phát hiện outliers
    if config.get('detect_outliers', True):
        outliers_info = detect_outliers(
            df_clean,
            method=config.get('outlier_method', 'zscore'),
            threshold=config.get('outlier_threshold', 3)
        )
        cleaning_info['outliers'] = outliers_info
    
    # Phân loại các cột
    column_types = classify_columns(df_clean)
    cleaning_info['column_types'] = column_types
    
    # Scale features
    if config.get('scale_features', True):
        df_clean = scale_features(
            df_clean,
            method=config.get('scale_method', 'standard')
        )
    
    # Encode categorical data
    if config.get('encode_categorical', True):
        df_clean = encode_categorical_data(
            df_clean,
            method=config.get('encode_method', 'label')
        )
    
    # Thêm thông tin về shape cuối cùng của DataFrame
    cleaning_info['final_shape'] = df_clean.shape
    
    return df_clean, cleaning_info
