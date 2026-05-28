import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.decomposition import PCA
from .adaboost import AdaBoost

def plot_confusion_matrix(y_train, y_pred_train, y_test, y_pred_test, acc_train, acc_test):
    """
    Vẽ ma trận nhầm lẫn (Confusion Matrix) cho cả tập huấn luyện (Train) và tập kiểm thử (Test).
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f'Confusion Matrix – AdaBoost', fontsize=14, fontweight='bold')

    for ax, y_true, y_pred, title in [
        (axes[0], y_train, y_pred_train, f'Tập Train  (Acc = {acc_train:.2f}%)'),
        (axes[1], y_test,  y_pred_test,  f'Tập Test   (Acc = {acc_test:.2f}%)'),
    ]:
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['Lành tính (B)', 'Ác tính (M)'],
                    yticklabels=['Lành tính (B)', 'Ác tính (M)'],
                    linewidths=0.5, linecolor='gray',
                    annot_kws={'size': 14, 'weight': 'bold'})
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('Dự đoán', fontsize=11)
        ax.set_ylabel('Thực tế', fontsize=11)

    plt.tight_layout()
    plt.show()

def plot_alphas_distribution(model):
    """
    Vẽ biểu đồ phân bố và giá trị của trọng số alpha theo từng vòng lặp của weak classifiers.
    """
    alphas = [clf.alpha for clf in model.clfs]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    fig.suptitle('Trọng số α (Amount of Say) của các Weak Classifier',
                 fontsize=13, fontweight='bold')

    # Biểu đồ cột (Bar chart) biểu diễn giá trị alpha theo vòng lặp
    axes[0].bar(range(1, len(alphas) + 1), alphas,
                color='steelblue', alpha=0.8, edgecolor='navy', linewidth=0.4)
    axes[0].set_xlabel('Weak Classifier thứ t')
    axes[0].set_ylabel('Giá trị α')
    axes[0].set_title('α theo vòng lặp')
    axes[0].grid(axis='y', alpha=0.3)
    axes[0].axhline(np.mean(alphas), color='red', linestyle='--',
                    label=f'Mean α = {np.mean(alphas):.3f}')
    axes[0].legend()

    # Biểu đồ tần suất (Histogram) phân bố giá trị alpha
    axes[1].hist(alphas, bins=20, color='steelblue', edgecolor='navy',
                 alpha=0.8, linewidth=0.4)
    axes[1].set_xlabel('Giá trị α')
    axes[1].set_ylabel('Tần suất')
    axes[1].set_title('Phân bố α')
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.show()

    print(f"α min  = {min(alphas):.4f}")
    print(f"α max  = {max(alphas):.4f}")
    print(f"α mean = {np.mean(alphas):.4f}")

def plot_pca_results(X_test, y_test, y_pred_test):
    """
    Giảm chiều tập kiểm thử bằng PCA xuống không gian 2D và vẽ ranh giới/nhãn thực tế cùng kết quả dự đoán.
    """
    pca      = PCA(n_components=2)
    X_test2D = pca.fit_transform(X_test)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'Trực quan hóa kết quả phân lớp AdaBoost (PCA 2D)',
                 fontsize=14, fontweight='bold')

    # Vẽ nhãn thực tế (Ground Truth)
    for label, color, marker, name in [
        (-1, '#1E88E5', 'o', 'Lành tính (B)'),
        ( 1, '#E53935', 's', 'Ác tính (M)'),
    ]:
        idx = y_test == label
        axes[0].scatter(X_test2D[idx, 0], X_test2D[idx, 1],
                        c=color, marker=marker, label=name,
                        alpha=0.75, edgecolors='white', linewidths=0.3, s=50)
    axes[0].set_title('Nhãn thực tế', fontsize=12, fontweight='bold')
    axes[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    axes[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    axes[0].legend(fontsize=10)
    axes[0].grid(alpha=0.25)

    # Vẽ kết quả dự đoán (Dự đoán đúng vs Dự đoán sai)
    correct   = y_test == y_pred_test
    incorrect = ~correct
    axes[1].scatter(X_test2D[correct, 0],   X_test2D[correct, 1],
                    c='#43A047', marker='o', label='Dự đoán đúng',
                    alpha=0.75, edgecolors='white', linewidths=0.3, s=50)
    axes[1].scatter(X_test2D[incorrect, 0], X_test2D[incorrect, 1],
                    c='#FF5722', marker='X', label='Dự đoán sai',
                    alpha=0.95, edgecolors='black', linewidths=0.6, s=120)
    axes[1].set_title('Kết quả dự đoán', fontsize=12, fontweight='bold')
    axes[1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    axes[1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    axes[1].legend(fontsize=10)
    axes[1].grid(alpha=0.25)

    plt.tight_layout()
    plt.show()

    print(f"PCA giải thích {sum(pca.explained_variance_ratio_)*100:.1f}% phương sai")
    print(f"Số mẫu dự đoán sai: {incorrect.sum()} / {len(y_test)}")

def plot_roc_curve(model, X_test, y_test):
    """
    Vẽ đường cong ROC (Receiver Operating Characteristic) và tính diện tích dưới đường cong (AUC).
    """
    scores = model.predict_score(X_test)
    fpr, tpr, _ = roc_curve(y_test, scores)
    roc_auc      = auc(fpr, tpr)

    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, color='#1E88E5', lw=2.5,
             label=f'AdaBoost (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=1.5, label='Random')
    plt.fill_between(fpr, tpr, alpha=0.08, color='#1E88E5')
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(f'ROC Curve – AdaBoost (n_clf = {len(model.clfs)})', fontsize=13, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    print(f"AUC = {roc_auc:.4f}")

def plot_accuracy_vs_estimators(X_train, y_train, X_test, y_test, clf_range=None):
    """
    Vẽ biểu đồ liên hệ giữa độ chính xác (Accuracy) trên tập huấn luyện và tập kiểm thử
    khi số lượng weak classifiers thay đổi, đồng thời vẽ biểu đồ khoảng cách Overfitting.
    """
    if clf_range is None:
        clf_range = list(range(1, 101, 5))
        
    train_accs  = []
    test_accs   = []

    print("⏳ Đang chạy thực nghiệm thay đổi số weak classifier... (vui lòng chờ)")
    for n in clf_range:
        m = AdaBoost(n_clf=n)
        m.fit(X_train, y_train)
        
        y_pred_tr = m.predict(X_train)
        y_pred_te = m.predict(X_test)
        
        train_acc = np.sum(y_train == y_pred_tr) / len(y_train) * 100
        test_acc  = np.sum(y_test == y_pred_te) / len(y_test) * 100
        
        train_accs.append(train_acc)
        test_accs.append(test_acc)

    best_idx = np.argmax(test_accs)
    print(f"✅ Xong! Kết quả tốt nhất trên Test: {test_accs[best_idx]:.2f}% (n_clf = {clf_range[best_idx]})")

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle('Thực nghiệm: Accuracy theo số Weak Classifier', fontsize=14, fontweight='bold')

    # Line chart đại diện cho Accuracy tập train và test
    axes[0].plot(clf_range, train_accs, 'o-', color='#1E88E5',
                 label='Train Accuracy', markersize=5, linewidth=2)
    axes[0].plot(clf_range, test_accs, 's-', color='#E53935',
                 label='Test Accuracy',  markersize=5, linewidth=2)
    axes[0].axhline(max(test_accs), color='#E53935', linestyle='--', alpha=0.4,
                    label=f'Best Test = {max(test_accs):.2f}%')
    axes[0].set_xlabel('Số weak classifiers (n_clf)', fontsize=12)
    axes[0].set_ylabel('Accuracy (%)', fontsize=12)
    axes[0].set_title('Accuracy vs n_clf', fontsize=12, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(alpha=0.3)
    axes[0].set_ylim([min(min(train_accs), min(test_accs)) - 2, 101])

    # Gap chart biểu hiện mức độ Overfitting (Khoảng cách giữa Train và Test)
    gap = [tr - te for tr, te in zip(train_accs, test_accs)]
    axes[1].fill_between(clf_range, gap, alpha=0.35, color='#FF7043')
    axes[1].plot(clf_range, gap, 'o-', color='#FF7043', markersize=5, linewidth=2)
    axes[1].axhline(0, color='gray', linestyle='--', linewidth=1)
    axes[1].set_xlabel('Số weak classifiers (n_clf)', fontsize=12)
    axes[1].set_ylabel('Train Acc − Test Acc (%)', fontsize=12)
    axes[1].set_title('Khoảng cách Train–Test (Overfitting gap)', fontsize=12, fontweight='bold')
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.show()
    
    return clf_range, train_accs, test_accs
