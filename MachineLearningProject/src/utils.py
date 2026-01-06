import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, classification_report
import json


def plot_results(results, save_path='results.png'):
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Colorectal Texture Classifier Results', fontsize=16)

    aucs = {k: v['auc'] for k, v in results.items()}
    best_name = max(aucs, key=aucs.get)
    axes[0, 0].bar(aucs.keys(), aucs.values(), color='skyblue', alpha=0.8)
    axes[0, 0].set_title('Test ROC-AUC Scores')
    axes[0, 0].set_ylabel('AUC')
    for i, (name, auc) in enumerate(aucs.items()):
        axes[0, 0].text(i, auc + 0.01, f'{auc:.3f}', ha='center')

    # # 2. Best Model Confusion Matrix (dummy for LC25000 - use real in eval)
    # best_model = results[best_name]['model']
    # # Use test data from train_evaluate (pass as param or dummy)
    # cm = np.array([[950, 50], [30, 970]])  # Example 97% acc
    # disp = ConfusionMatrixDisplay(cm, display_labels=['Healthy', 'Tumor'])
    # disp.plot(ax=axes[0, 1], cmap='Blues', values_format='.0f')
    # axes[0, 1].set_title(f'{best_name} Confusion Matrix')

    # 3. ROC Curve (dummy for demo)
    # fpr = np.linspace(0, 1, 100)
    # tpr = fpr ** 0.5  # Example ROC
    # axes[1, 0].plot(fpr, tpr, label=f'{best_name} (AUC={aucs[best_name]:.3f})')
    # axes[1, 0].plot([0, 1], [0, 1], 'k--', label='Random')
    # axes[1, 0].set_xlabel('False Positive Rate')
    # axes[1, 0].set_ylabel('True Positive Rate')
    # axes[1, 0].set_title('ROC Curve')
    # axes[1, 0].legend()

    # 4. Feature Importance (RF/XGB only)
    # if 'RF' in results:
    #     importances = results['RF']['model'].feature_importances_
    #     top10_idx = np.argsort(importances)[-10:]
    #     sns.barplot(x=importances[top10_idx], y=[f'F{i}' for i in top10_idx], ax=axes[1, 1])
    #     axes[1, 1].set_title('Top 10 RF Feature Importance')
    # else:
    #     axes[1, 1].text(0.5, 0.5, 'No Tree Model', ha='center', va='center', transform=axes[1, 1].transAxes)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def save_results(results, path='results.json'):
    serializable = {}
    for name, res in results.items():
        serializable[name] = {
            'auc': float(res['auc']),
            'f1_macro': res['report']['macro avg']['f1-score'],
            'accuracy': res['report']['accuracy']
        }
    with open(path, 'w') as f:
        json.dump(serializable, f, indent=2)
