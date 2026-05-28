import os
import pandas as pd

def load_data(data_path="data/data.csv", gdrive_file_id="1z107uUGL7Z1yO2sE4Q_CWPGsrXTAm66G"):
    """
    Tải và đọc dữ liệu từ file CSV.
    Nếu file không tồn tại tại data_path, hàm sẽ tự động tải từ Google Drive thông qua gdrive_file_id.
    
    Parameters:
    -----------
    data_path : str
        Đường dẫn tới file dữ liệu CSV.
    gdrive_file_id : str
        ID file trên Google Drive để tải trong trường hợp không có file local.
        
    Returns:
    --------
    pd.DataFrame
        DataFrame chứa dữ liệu thô.
    """
    if not os.path.exists(data_path):
        # Tạo thư mục cha nếu chưa có
        dir_name = os.path.dirname(data_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
            
        print(f"⚠️ Không tìm thấy file {data_path}. Đang tiến hành tải từ Google Drive...")
        try:
            import gdown
        except ImportError:
            import subprocess
            import sys
            print("📦 Đang cài đặt thư viện 'gdown'...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown"])
            import gdown
            
        url = f"https://drive.google.com/uc?id={gdrive_file_id}"
        gdown.download(url, data_path, quiet=False)
        print("✅ Đã tải dữ liệu thành công!")
        
    df_raw = pd.read_csv(data_path)
    print(f"📦 Kích thước bộ dữ liệu: {df_raw.shape[0]} mẫu × {df_raw.shape[1]} cột")
    print(f"📋 Danh sách cột: {list(df_raw.columns)[:5]} ... (tổng cộng {len(df_raw.columns)} cột)")
    return df_raw
