import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.style.use("dark_background")

# metrics_df: Dataframe with columns: model, accuracy, macro_f1, weighted_f1


def plot_metrics_bar(metrics_df: pd.DataFrame) -> plt.Figure:
    df = metrics_df.set_index("model")[["accuracy", "macro_f1", "weighted_f1"]]

    fig, ax = plt.subplots(figsize=(6, 3.5))
    x = np.arange(len(df.index))
    width = 0.25
    colors = ["#4C72B0", "#5DA5DA", "#9EDAE5"]

    ax.bar(x - width, df["accuracy"].values, width, label="accuracy", color=colors[0])
    ax.bar(x, df["macro_f1"].values, width, label="macro_f1", color=colors[1])
    ax.bar(
        x + width, df["weighted_f1"].values, width, label="weighted_f1", color=colors[2]
    )

    ax.set_title("Model Metrics Comparison")
    ax.set_xlabel("Model")
    ax.set_ylabel("Score")
    ax.set_xticks(x)
    ax.set_xticklabels(df.index.tolist(), rotation=0)
    ax.legend(loc="best")
    fig.tight_layout()
    return fig


# cm_df: Dataframe with index=true labels, columns=pred labels
def plot_confusion_matrix_heatmap(cm_df: pd.DataFrame) -> plt.Figure:

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm_df.values, cmap="Blues")

    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")

    ax.set_xticks(np.arange(cm_df.shape[1]))
    ax.set_yticks(np.arange(cm_df.shape[0]))
    ax.set_xticklabels(cm_df.columns, rotation=45, ha="right")
    ax.set_yticklabels(cm_df.index)

    for i in range(cm_df.shape[0]):
        for j in range(cm_df.shape[1]):
            ax.text(j, i, int(cm_df.iat[i, j]), ha="center", va="center", fontsize=8)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


# fi_df columns: feature, importance (or mean_abs_coef)
def plot_feature_importance(fi_df: pd.DataFrame, top_n: int = 20) -> plt.Figure:

    df = fi_df.head(top_n).copy()

    if "importance" in df.columns:
        score_col = "importance"
        title = "Random Forest Feature Importance"
    elif "mean_abs_coef" in df.columns:
        score_col = "mean_abs_coef"
        title = "Logistic Regression Coefficients (mean abs)"
    else:
        raise ValueError(
            "Expected 'importance' or 'mean_abs_coef' in feature importance DF."
        )

    df = df.sort_values(score_col, ascending=True)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(df["feature"], df[score_col], color="#4C72B0")
    ax.grid(axis="x", alpha=0.2)

    ax.set_title(title)
    ax.set_xlabel(score_col)
    ax.set_ylabel("feature")
    fig.tight_layout()
    return fig


# pca_df: Dataframe with columns: PC1, PC2, subtype
def plot_pca_scatter(
    pca_df: pd.DataFrame,
    label_col: str = "subtype",
    x_col: str = "PC1",
    y_col: str = "PC2",
) -> plt.Figure:

    fig, ax = plt.subplots(figsize=(6, 4))

    labels = sorted(pca_df[label_col].astype(str).unique().tolist())
    for lab in labels:
        sub = pca_df[pca_df[label_col].astype(str) == lab]
        ax.scatter(sub[x_col], sub[y_col], s=20, alpha=0.7, label=str(lab))

    ax.set_title("PCA (2D) of Gene Expression Features")
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    return fig
