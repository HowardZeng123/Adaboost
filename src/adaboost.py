import numpy as np

class DecisionStump:
    """
    Cây quyết định cụt (Cây quyết định 1 nút) - Weak Classifier (Mô hình học yếu) trong AdaBoost.
    Phân lớp dựa trên 1 đặc trưng duy nhất và 1 ngưỡng cụ thể.
    
    Attributes:
    -----------
    polarity : int
        Chiều so sánh (+1 hoặc -1).
        - Nếu 1: mẫu có giá trị đặc trưng lớn hơn ngưỡng sẽ được dự đoán là +1, ngược lại là -1.
        - Nếu -1: mẫu có giá trị đặc trưng lớn hơn ngưỡng sẽ được dự đoán là -1, ngược lại là +1.
    feature_idx : int
        Chỉ số đặc trưng được chọn để phân loại.
    threshold : float
        Ngưỡng giá trị để phân tách đặc trưng.
    alpha : float
        Trọng số biểu thị "tiếng nói" (amount of say) của classifier này trong quyết định chung.
    """
    def __init__(self):
        self.polarity    = 1
        self.feature_idx = None
        self.threshold   = None
        self.alpha       = None

    def predict(self, X):
        """
        Dự đoán nhãn (-1 hoặc +1) cho các mẫu dựa trên đặc trưng và ngưỡng của DecisionStump.
        """
        n_samples = X.shape[0]
        X_col     = X[:, self.feature_idx]
        preds     = np.ones(n_samples)

        if self.polarity == 1:
            preds[X_col < self.threshold] = -1
        else:
            preds[X_col > self.threshold] = -1

        return preds


class AdaBoost:
    """
    Thuật toán AdaBoost (Adaptive Boosting) tự cài đặt từ Scratch.
    
    Parameters:
    -----------
    n_clf : int
        Số lượng weak classifier (số vòng lặp huấn luyện, mặc định là 50).
        
    Attributes:
    -----------
    clfs : list of DecisionStump
        Danh sách các mô hình học yếu sau khi được huấn luyện xong.
    """
    def __init__(self, n_clf=50):
        self.n_clf = n_clf
        self.clfs  = []   # Danh sách DecisionStump đã huấn luyện

    def fit(self, X, y):
        """
        Huấn luyện thuật toán AdaBoost trên tập dữ liệu X và nhãn y.
        """
        n_samples, n_features = X.shape
        self.clfs = []

        # Bước 1: Khởi tạo trọng số mẫu đều nhau w_i = 1/N
        w = np.full(n_samples, 1 / n_samples)

        for _ in range(self.n_clf):
            clf       = DecisionStump()
            min_error = float('inf')

            # Bước 2a: Tìm Stump tốt nhất bằng cách duyệt qua mọi đặc trưng và ngưỡng giá trị duy nhất
            for feat_i in range(n_features):
                X_col      = X[:, feat_i]
                thresholds = np.unique(X_col)

                for thr in thresholds:
                    for polarity in [1, -1]:
                        preds = np.ones(n_samples)
                        if polarity == 1:
                            preds[X_col < thr] = -1
                        else:
                            preds[X_col > thr] = -1

                        # Bước 2b: Tính tỷ lệ lỗi có trọng số của Stump hiện tại
                        error = np.sum(w[y != preds])

                        # Nếu lỗi > 0.5 -> đảo chiều phân lớp (vì nhị phân vẫn mang thông tin có giá trị)
                        if error > 0.5:
                            error    = 1 - error
                            polarity = -polarity

                        # Lưu lại stump có lỗi nhỏ nhất
                        if error < min_error:
                            min_error        = error
                            clf.polarity     = polarity
                            clf.feature_idx  = feat_i
                            clf.threshold    = thr

            # Bước 2c: Tính alpha (trọng số của stump này)
            EPS       = 1e-10
            clf.alpha = 0.5 * np.log((1 - min_error) / (min_error + EPS))

            # Bước 2d: Cập nhật trọng số mẫu w_i
            preds = clf.predict(X)
            w    *= np.exp(-clf.alpha * y * preds)
            
            # Bước 2e: Chuẩn hóa trọng số mẫu w_i để tổng bằng 1
            w    /= np.sum(w)

            # Lưu weak classifier vào danh sách
            self.clfs.append(clf)

    def predict(self, X):
        """
        Dự đoán nhãn cuối cùng bằng tổ hợp bỏ phiếu có trọng số (weighted vote) của các weak classifier.
        F(x) = sign( Σ α_t * h_t(x) )
        """
        agg = np.sum([clf.alpha * clf.predict(X) for clf in self.clfs], axis=0)
        return np.where(agg >= 0, 1, -1)

    def predict_score(self, X):
        """
        Trả về tổng điểm có trọng số (chưa qua hàm sign) để vẽ đường cong ROC.
        """
        return np.sum([clf.alpha * clf.predict(X) for clf in self.clfs], axis=0)

# Alias để hỗ trợ cả 2 tên gọi phổ biến
AdaBoostScratch = AdaBoost
