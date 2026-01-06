import numpy as np
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, accuracy_score, f1_score
from sklearn.model_selection import cross_val_score

from src.models import get_models, preprocess


def train_and_evaluate(X_train, y_train, X_val, y_val, X_test, y_test):

    X_train_p, y_train_p, X_test_p = preprocess(
        X_train, y_train, X_test
    )

    models = get_models()
    results = {}

    for name, model in models.items():
        model.fit(X_train_p, y_train_p)

        y_pred = model.predict(X_test_p)
        y_proba = model.predict_proba(X_test_p)[:, 1]

        results[name] = {
            'auc': roc_auc_score(y_test, y_proba),
            'f1': f1_score(y_test, y_pred),
            'accuracy': accuracy_score(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'report': classification_report(
                y_test, y_pred,
                target_names=['Healthy', 'Tumor'],
                output_dict=True
            ),
            'model': model,
            'y_test': y_test,
            'y_pred': y_pred,
            'y_proba': y_proba
        }

        cm = results[name]['confusion_matrix']
        tn, fp, fn, tp = cm.ravel()

        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)

        print(f"\n{name}:")
        print(f"  AUC:        {results[name]['auc']:.4f}")
        print(f"  Accuracy:   {results[name]['accuracy']:.4f}")
        print(f"  F1 Score:   {results[name]['f1']:.4f}")
        print(f"  Sensitivity: {sensitivity:.4f} ({tp} z {tp + fn})")
        print(f"  Specificity: {specificity:.4f} ({tn} z {tn + fp})")
        print(f"  Confusion Matrix (1000×1000):")
        print(f"    TN={tn}  FP={fp}")
        print(f"    FN={fn}  TP={tp}")

    return results

