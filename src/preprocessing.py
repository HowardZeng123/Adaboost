import numpy as np
import pandas as pd
from pandas.api.types import is_string_dtype
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

def preprocess(df, test_size=0.2, random_state=42):
    """
    Tiền xử lý dữ liệu:
    1. Loại bỏ các cột không cần thiết ('id', 'Unnamed: 32').
    2. Loại bỏ các dòng có giá trị thiếu.
    3. Mã hóa nhãn: M (Ác tính) -> +1, B (Lành tính) -> -1.
    4. Tách đặc trưng (X) và nhãn (y).
    5. Chia tập Train/Test (phân lớp có đối xứng/stratified).
    6. Chuẩn hóa đặc trưng về khoảng [0, 1] bằng MinMaxScaler.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame dữ liệu thô.
    test_size : float
        Tỷ lệ tập kiểm thử (mặc định: 0.2).
    random_state : int
        Seed ngẫu nhiên để tái lập kết quả (mặc định: 42).
        
    Returns:
    --------
    X_train : np.ndarray
        Đặc trưng tập huấn luyện đã chuẩn hóa.
    X_test : np.ndarray
        Đặc trưng tập kiểm thử đã chuẩn hóa.
    y_train : np.ndarray
        Nhãn tập huấn luyện (+1/-1).
    y_test : np.ndarray
        Nhãn tập kiểm thử (+1/-1).
    feature_names : list of str
        Tên các đặc trưng sau khi xử lý.
    scaler : MinMaxScaler
        Đối tượng scaler đã fit trên tập huấn luyện.
    """
    df_clean = df.copy()
    
    # 1. Loại bỏ các cột không cần thiết
    drop_cols = [c for c in df_clean.columns if 'id' in c.lower() or 'unnamed' in c.lower()]
    df_clean.drop(columns=drop_cols, inplace=True, errors='ignore')
    print(f"✅ Đã xóa các cột không cần thiết: {drop_cols}")
    
    # 2. Xóa các dòng có giá trị thiếu
    before = len(df_clean)
    df_clean.dropna(inplace=True)
    print(f"✅ Xóa {before - len(df_clean)} dòng có giá trị thiếu (còn {len(df_clean)} dòng)")
    
    # 3. Mã hóa nhãn: M -> +1 (ác tính), B -> -1 (lành tính)
    if 'diagnosis' not in df_clean.columns:
        raise ValueError("Thiếu cột 'diagnosis' trong dữ liệu.")

    if is_string_dtype(df_clean['diagnosis']):
        df_clean['diagnosis'] = df_clean['diagnosis'].map({'M': 1, 'B': -1})

    invalid_labels = set(df_clean['diagnosis'].dropna().unique()) - {1, -1}
    if df_clean['diagnosis'].isna().any() or invalid_labels:
        raise ValueError("Cột 'diagnosis' chỉ hỗ trợ nhãn M/B hoặc 1/-1.")

    df_clean['diagnosis'] = df_clean['diagnosis'].astype(int)
    print(f"✅ Mã hóa nhãn thành công: M (Ác tính) → +1 | B (Lành tính) → -1")
    print("   Phân bố nhãn sau khi mã hóa:")
    counts = df_clean['diagnosis'].value_counts()
    print(f"   - Ác tính (+1)  : {counts.get(1, 0)} mẫu")
    print(f"   - Lành tính (-1): {counts.get(-1, 0)} mẫu")
            
    # 4. Tách X, y
    X = df_clean.drop('diagnosis', axis=1).values
    y = df_clean['diagnosis'].values
    feature_names = df_clean.drop('diagnosis', axis=1).columns.tolist()
    
    print(f"✅ Tách X (đặc trưng) và y (nhãn):")
    print(f"   - Kích thước X: {X.shape}")
    print(f"   - Kích thước y: {y.shape}")
    print(f"   - Số lượng đặc trưng: {len(feature_names)}")
    
    # 5. Chia tập Train/Test theo tỷ lệ (phân bố nhãn cân đối - stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"✅ Phân chia tập huấn luyện / kiểm thử (tỷ lệ {int((1-test_size)*100)}/{int(test_size*100)}):")
    print(f"   - Tập Train: {X_train.shape[0]} mẫu")
    print(f"   - Tập Test : {X_test.shape[0]} mẫu")
    
    # 6. Chuẩn hóa Min-Max Scaling (chỉ fit trên tập huấn luyện để tránh data leakage)
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    print(f"✅ Chuẩn hóa đặc trưng bằng MinMaxScaler về đoạn [0, 1]")
    print(f"   - Giá trị nhỏ nhất của Train: {X_train.min():.2f}")
    print(f"   - Giá trị lớn nhất của Train: {X_train.max():.2f}")
    
    return X_train, X_test, y_train, y_test, feature_names, scaler
