from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE


def get_models():
    return {
        'SVM': SVC(C=10, kernel='rbf', gamma='scale', probability=True, class_weight='balanced'),
        # C = 30, gamma = 70 -> "auc": 0.789115, "f1_macro": 0.6885223762911075, "accuracy": 0.714
        'RF': RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42),
        'XGB': XGBClassifier(n_estimators=200, max_depth=6, random_state=42, eval_metric='logloss'),
        'GB': GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=7, min_samples_split=5,
                                         min_samples_leaf=2, subsample=0.8)
    }


def preprocess(X_train, y_train, X_test):
    scaler = StandardScaler()
    smote = SMOTE(random_state=42)

    X_train_scaled = scaler.fit_transform(X_train)
    X_train_bal, y_train_bal = smote.fit_resample(X_train_scaled, y_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_bal, y_train_bal, X_test_scaled
