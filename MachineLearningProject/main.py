import numpy as np
from sklearn.metrics import classification_report
from src.data_loader import load_config, load_lc25000_features, load_crchgd_features, get_splits, load_crchgd_balanced
from src.texture_features import extract_all_features
from src.train_evaluate import train_and_evaluate
from src.utils import plot_results, save_results
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="skimage.feature.texture")


def main():
    config = load_config()

    def feat_extractor(img):
        return extract_all_features(img, config)

    print("1️⃣ Extracting LC25000 features")
    X_lc, y_lc = load_lc25000_features(feat_extractor, config['data']['lc25000_path'], config)
    print("X_lc shape:", X_lc.shape)

    (X_train, y_train), (X_val, y_val), (X_test, y_test) = get_splits(X_lc, y_lc, config)

    print("\n2️⃣ Training models")
    results_lc = train_and_evaluate(X_train, y_train, X_val, y_val, X_test, y_test)

    aucs = {k: f"{v['auc']:.3f}" for k, v in results_lc.items()}
    print("\n✅ LC25000 Results (Test AUC):")
    for name, auc in aucs.items():
        print(f"  {name:4}: {auc}")

    best_model = max(results_lc, key=lambda k: results_lc[k]['auc'])
    print(f"\n🏆 Best: {best_model} (AUC={aucs[best_model]})")

    print("\n3️⃣ CRC-HGD-v1 Transfer Test")
    try:
        X_crc, y_crc = load_crchgd_balanced(feat_extractor, config['data']['crchgd_path'], config)
        if X_crc.shape[0] > 0:
            y_pred_crc = results_lc[best_model]['model'].predict(X_crc)
            print(f"Healthy predicted: {np.sum(y_pred_crc == 0)}")  # Should be ~280
            print(f"Tumor predicted: {np.sum(y_pred_crc == 1)}")  # Should be ~280
            print("\n📊 CRC-HGD-v1 Results:")
            print(classification_report(y_crc, y_pred_crc, target_names=['Healthy', 'Tumor'], zero_division=0))
        else:
            print("No CRC-HGD-v1 data found (LC25000 OK)")
    except Exception as e:
        print(f"CRC-HGD-v1 skipped: {e}")

    print("\nSaving results")
    save_results(results_lc, 'results_lc.json')
    plot_results(results_lc, 'results_lc.png')

    print("Files: results_lc.json, results_lc.png")


if __name__ == "__main__":
    main()
