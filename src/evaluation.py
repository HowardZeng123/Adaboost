import numpy as np
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score

def acc(y_true, y_pred):
    """
    Tính độ chính xác (Accuracy).
    """
    return np.sum(y_true == y_pred) / len(y_true)

def evaluate_model(model, X_train, y_train, X_test, y_test):
    """
    Đánh giá chi tiết mô hình AdaBoost trên cả hai tập huấn luyện (train) và kiểm thử (test).
    In báo cáo phân lớp (Classification Report) và trả về dictionary các metrics.
    
    Parameters:
    -----------
    model : AdaBoost
        Mô hình AdaBoost đã được huấn luyện.
    X_train : np.ndarray
        Đặc trưng tập train.
    y_train : np.ndarray
        Nhãn tập train.
    X_test : np.ndarray
        Đặc trưng tập test.
    y_test : np.ndarray
        Nhãn tập test.
        
    Returns:
    --------
    dict
        Chứa kết quả đánh giá (accuracy train/test, precision, recall, f1_score).
    """
    y_pred_train = model.predict(X_train)
    y_pred_test  = model.predict(X_test)

    acc_train = acc(y_train, y_pred_train) * 100
    acc_test  = acc(y_test,  y_pred_test)  * 100
    prec      = precision_score(y_test, y_pred_test)
    rec       = recall_score(y_test, y_pred_test)
    f1        = f1_score(y_test, y_pred_test)

    print("=" * 50)
    print("  KẾT QUẢ ĐÁNH GIÁ MÔ HÌNH ADABOOST")
    print("=" * 50)
    print(f"  Số weak classifiers  : {len(model.clfs)}")
    print(f"  Accuracy (Train)     : {acc_train:.2f}%")
    print(f"  Accuracy (Test)      : {acc_test:.2f}%")
    print(f"  Precision            : {prec:.4f}")
    print(f"  Recall               : {rec:.4f}")
    print(f"  F1-Score             : {f1:.4f}")
    print("=" * 50)
    print()
    print("Classification Report (Tập Test):")
    print(classification_report(y_test, y_pred_test,
                                target_names=['Lành tính (B)', 'Ác tính (M)']))
    
    return {
        'y_pred_train': y_pred_train,
        'y_pred_test': y_pred_test,
        'acc_train': acc_train,
        'acc_test': acc_test,
        'precision': prec,
        'recall': rec,
        'f1_score': f1
    }
