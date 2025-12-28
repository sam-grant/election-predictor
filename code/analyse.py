import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report
from matplotlib import pyplot as plt
from pathlib import Path
style_path = Path(__file__).parent / "plot.mplstyle"
plt.style.use(style_path)

class Analyse:
    """
    Analyse model results
    """
    def __init__(self, results):
        self.results = results
        
    def df_confusion_matrix(self):
        cm = confusion_matrix(self.results["y_test"], self.results["y_pred"])
        return pd.DataFrame(
            cm,
            index=["True 0", "True 1"],
            columns=["Predicted 0", "Predicted 1"]
        )
        
    def plot_roc_curve(self, save_output=False):
        fpr, tpr, _ = roc_curve(self.results["y_test"], self.results["y_proba"])
        auc_score = auc(fpr, tpr)

        plt.figure()
        plt.plot(fpr, tpr, linewidth=2.5, label=f"ROC (AUC = {auc_score:.3f})")
        plt.plot([0, 1], [0, 1], "k--", linewidth=1.5, label="Random")
        plt.xlabel("False positive rate")
        plt.ylabel("True positive rate")
        plt.legend(loc="lower right")
        plt.grid(alpha=0.5)
        plt.tight_layout()

        if save_output:
            out_path = Path(f"../images/{self.results['tag']}")
            out_path.mkdir(parents=True, exist_ok=True)
            file_path = out_path / "roc_curve.png"
            plt.savefig(file_path)
            print(f"✅ Wrote {file_path}")
        plt.show()

        return {
            "fpr": fpr,
            "tpr": tpr,
            "auc": auc_score
        }

    def df_classification_report(self):
        report_dict = classification_report(self.results["y_test"], self.results["y_pred"], output_dict=True)
        report = pd.DataFrame(report_dict).transpose()
        return report

    def df_feature_importance(self, X, top_n=15):
        """
        Extract feature importance from the model.
        """
        model = self.results["model"]
        feature_names = X.columns
        
        # Try different methods depending on model type
        if hasattr(model, 'coef_'):
            # Linear models
            importance = np.abs(model.coef_[0])
        elif hasattr(model, "feature_importances_"):
            # Tree-based models (RandomForest, XGBoost)
            importance = model.feature_importances_
        else:
            raise ValueError(f"Model type {type(model).__name__} doesn't support feature importance extraction")
        
        # Create DataFrame
        df_importance = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importance
        }).sort_values("Importance", ascending=False)
        
        if top_n:
            df_importance = df_importance.head(top_n)
        
        return df_importance
    
    def plot_feature_importance(self, X, top_n=15, save_output=False):
        """
        Plot feature importance 
        """
        df_importance = self.df_feature_importance(X, top_n=top_n)
        
        # Adjust figure height based on number of features
        fig_height = max(6, len(df_importance) * 0.4)
        plt.figure(figsize=(10, fig_height))
        plt.barh(range(len(df_importance)), df_importance["Importance"])
        plt.yticks(range(len(df_importance)), df_importance["Feature"], fontsize=10)
        plt.xlabel("Importance")
        # plt.title(f"Top {top_n} features")
        plt.gca().invert_yaxis()
        plt.tight_layout()
        
        if save_output:
            out_path = Path(f"../images/{self.results['tag']}")
            out_path.mkdir(parents=True, exist_ok=True)
            file_path = out_path / "feature_importance.png"
            plt.savefig(file_path, bbox_inches='tight')
            print(f"✅ Wrote {file_path}")
        plt.show()
        
        # cm = self.confusion_matrix()
        # tn, fp, fn, tp = cm.ravel()
        # # Positive precision: "out of those predicted positive, what fraction were actually positive?" 
        # precision_pos = tp / (tp + fp) if (tp + fp) > 0 else 0
        # # Positive recall: "out of those actually positive, what fraction were predicted positive?"
        # recall_pos = tp / (tp + fn) if (tp + fn) > 0 else 0
        # # Positive F1-score: "harmonic mean of positive precision and recall"
        # f1_pos = 2 * (precision_pos * recall_pos) / (precision_pos + recall_pos) if (precision_pos + recall_pos) > 0 else 0
        # # Negative precision: "out of those predicted negative, what fraction were actually negative?"
        # precision_neg = tn / (tn + fn) if (tn + fn) > 0 else 0
        # recall_neg = tn / (tn + fp) if (tn + fp) > 0 else 0
        # # Negative F1-score: "harmonic mean of negative precision and recall"
        # f1_neg = 2 * (precision_neg * recall_neg) / (precision_neg + recall_neg) if (precision_neg + recall_neg) > 0 else 0

        # # Overall accuracy: the fraction of correct predictions
        # accuracy = (tp + tn) / (tp + tn + fp + fn)
        # 
        # report = pd.DataFrame({
        #     "0": [precision_neg, recall_neg, f1_neg, ""],
        #     "1": [precision_pos, recall_pos, f1_pos, ""]
        # }, index=["Precision", "Recall", "F1-Score", ""])
        # 
        # # Add accuracy on the last row
        # report.loc["Accuracy", "0"] = ""
        # report.loc["Accuracy", "1"] = accuracy
        # 
        # return report

    def execute(self, X=None, save_output=False):
        # display only works interactively
        print("⭐ Confusion matrix:\n")
        cm = self.df_confusion_matrix()
        display(cm)
        print("\n⭐ Classification report:\n")
        report = self.df_classification_report()
        display(report)
        print("\n⭐ ROC curve & feature importance:\n")
        roc = self.plot_roc_curve(save_output=save_output)
        
        # Feature importance if X is provided
        feature_importance = None
        if X is not None:
            try:
                feature_importance = self.plot_feature_importance(X, top_n=15, save_output=save_output)
                display(feature_importance)
            except ValueError as e:
                print(f"⚠️ Feature importance not available: {e}")
        
        # Return all results
        return {
            "cm": cm,
            "report": report,
            "roc": roc,
            "feature_importance": feature_importance
        }
    # def confusion_matrix(self, out_name=None):
    #    # tp = len(self.results["y_test"][(self.results["y_pred"] == 1) & (self.results["y_test"] == 1)]
    #    # tn = len(self.results["y_test"][(self.results["y_pred"] == 0) & (self.results["y_test"] == 0)])
    #    # fp = len(self.results["y_test"][(self.results["y_pred"] == 1) & (self.results["y_test"] == 0)])
    #    # fn = len(self.results["y_test"][(self.results["y_pred"] == 0) & (self.results["y_test"] == 1)])
    #    # cm = [[tn, fp], [fn, tp]]
    #    # return cm
    
# tests
# if __name__ == "__main__":
#     import joblib
#     results = joblib.load("../results/lr-base/results.pkl")
#     print("Results loaded successfully.")
#     ana = Analyse(results)    
#     print(ana.df_confusion_matrix())
#     ana.plot_roc_curve()
#     report = ana.df_classification_report()
#     print("\nReport:\n", report)
# 
