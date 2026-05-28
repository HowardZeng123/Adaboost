# 🫀 Dự án Phân lớp Ung Thư Vú bằng thuật toán AdaBoost từ Scratch

Dự án này thực hiện phân lớp dữ liệu ung thư vú sử dụng bộ dữ liệu **Breast Cancer Wisconsin** từ Kaggle và thuật toán **AdaBoost** được lập trình hoàn toàn từ scratch (không sử dụng thư viện ensemble có sẵn).

Dự án đã được tái cấu trúc thành một cấu trúc thư mục rõ ràng, chuyên nghiệp, giúp dễ đọc hiểu, mở rộng và bảo trì.

---

## 📂 Cấu trúc thư mục dự án

```text
Adaboost/
│
├── data/                           # 📂 Thư mục chứa dữ liệu
│   └── data.csv                    #    Bộ dữ liệu Breast Cancer Wisconsin
│
├── notebooks/                      # 📂 Thư mục chứa Jupyter Notebooks
│   └── main.ipynb                  #    Notebook chính, gọn gàng, điều phối luồng chạy
│
├── src/                            # 📂 Mã nguồn chính (Modules)
│   ├── __init__.py                 #    Định nghĩa Python package
│   ├── data_loader.py              #    Đọc dữ liệu & tự động tải từ Drive nếu thiếu
│   ├── eda.py                      #    Phân tích khám phá dữ liệu (EDA) và trực quan hóa phân bố
│   ├── preprocessing.py            #    Tiền xử lý, chuẩn hóa MinMaxScaler & chia tập Train/Test
│   ├── adaboost.py                 #    Lập trình DecisionStump & AdaBoost từ scratch
│   ├── evaluation.py               #    Hàm đánh giá chất lượng mô hình (Accuracy, Precision, Recall, F1)
│   └── visualization.py            #    Trực quan hóa nâng cao (Confusion Matrix, PCA 2D, ROC Curve, Thực nghiệm)
│
├── adaboost_breast_cancer.ipynb    #    (GIỮ LẠI) Notebook gốc để đối chiếu khi cần
├── requirements.txt                #    Danh sách các thư viện phụ thuộc
└── README.md                       #    Tài liệu hướng dẫn sử dụng này
```

---

## 🛠️ Cài đặt thư viện

Bạn cần cài đặt các thư viện cần thiết trước khi chạy dự án. Mở Terminal/Command Prompt trong thư mục gốc của dự án và chạy lệnh sau:

```bash
pip install -r requirements.txt
```

Các thư viện chính bao gồm:
* `pandas` & `numpy` (Xử lý dữ liệu)
* `matplotlib` & `seaborn` (Vẽ biểu đồ)
* `scikit-learn` (Dùng MinMaxScaler, train_test_split, PCA và tính metrics đánh giá)
* `gdown` (Tự động tải dữ liệu nếu chưa có)

---

## 🚀 Hướng dẫn chạy dự án

1. Hãy mở thư mục dự án bằng **Jupyter Notebook** hoặc **VS Code**.
2. Mở file [notebooks/main.ipynb](file:///c:/Users/Dam%20Hieu/Desktop/KPDL%20code/Adaboost/notebooks/main.ipynb).
3. Chạy lần lượt từng ô (cell) từ trên xuống dưới. 
   * Dữ liệu trong thư mục `data/data.csv` sẽ tự động được sử dụng. Nếu chưa có file dữ liệu, hàm `load_data()` trong `src/data_loader.py` sẽ tự động tải file từ Google Drive về cho bạn.

---

## 💡 Các tính năng chính

* **Tải dữ liệu thông minh:** Tự động kiểm tra file local và tải từ Drive của nhóm nếu thiếu.
* **EDA Trực quan:** Vẽ biểu đồ tròn/cột phân bố nhãn ác tính/lành tính, phân phối histogram đặc trưng và heatmap ma trận tương quan của top 15 đặc trưng quan trọng nhất.
* **AdaBoost từ Scratch:** Triển khai lớp học yếu `DecisionStump` và giải thuật `AdaBoost` chính xác theo lý thuyết cập nhật trọng số mẫu và tính toán giá trị $\alpha$.
* **Đánh giá & Trực quan hóa đa chiều:**
  * Ma trận nhầm lẫn (Confusion Matrix) trên cả tập Train và Test.
  * Biểu đồ phân phối trọng số $\alpha$ (Amount of say) để hiểu đóng góp của từng bộ phân lớp yếu.
  * Trực quan hóa ranh giới và kết quả dự đoán trên không gian giảm chiều **PCA 2D**.
  * Vẽ đường cong **ROC** và tính toán giá trị **AUC**.
  * **Thực nghiệm chuyên sâu:** Đồ thị khảo sát sự thay đổi của Accuracy theo số lượng Weak Classifiers (`n_clf` chạy từ 1 đến 100) và biểu đồ khoảng cách Overfitting (Overfitting Gap).
