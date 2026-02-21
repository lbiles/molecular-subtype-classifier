import argparse
import logging
import warnings
from pathlib import Path

from src.config import CLEAN_DATA_PATH, TARGET_COLUMN
from src.data.load_data import load_raw_metabric_data
from src.data.preprocess import (
    PreprocessArtifacts,
    clean_metabric_df,
    fit_preprocess,
    transform_features,
    transform_labels,
)
from src.data.split import split_df_train_test
from src.modeling.evaluate import evaluate_models, save_metrics
from src.modeling.explain import (
    save_logistic_regression_coefficients,
    save_random_forest_feature_importance,
)
from src.modeling.train import load_models, save_models, train_models

TRAIN_TAG = "[TRAIN]"
EVAL_TAG = "[EVAL]"
LOG_PATH = "artifacts/logs/app.log"
Path("artifacts/logs").mkdir(parents=True, exist_ok=True)

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(LOG_PATH),
        ],
    )


def _load_clean_and_split(seed: int, test_size: float, stratify: bool):
    """
    Load raw data, clean, split into train/test dataframes.
    """
    df = load_raw_metabric_data()
    df = clean_metabric_df(df, target_column=TARGET_COLUMN)

    df_train, df_test = split_df_train_test(
        df,
        target_column=TARGET_COLUMN,
        test_size=test_size,
        random_state=seed,
        stratify=stratify,
    )
    return df_train, df_test


def _artifacts_from_prep(prep: dict) -> PreprocessArtifacts:
    """
    Reconstruct PreprocessArtifacts from saved preprocessing metadata.
    """
    return PreprocessArtifacts(
        feature_columns=prep["feature_columns"],
        label_encoder=prep["label_encoder"],
        scaler=prep["scaler"],
        medians=prep.get("medians", {}),
    )


def _save_clean_splits(df_train, df_test) -> dict:
    """
    Save cleaned train/test dataframes so raw vs cleaned outputs are explicit.
    """
    out_dir = CLEAN_DATA_PATH.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    train_path = out_dir / "metabric_clean_train.csv"
    test_path = out_dir / "metabric_clean_test.csv"

    df_train.to_csv(train_path, index=False)
    df_test.to_csv(test_path, index=False)

    return {"clean_train": train_path, "clean_test": test_path}


def cmd_train(args) -> int:
    logger.info(
        "CLI train started: seed=%s test_size=%s stratify=%s",
        args.seed,
        args.test_size,
        not args.no_stratify,
    )
    df_train, _df_test = _load_clean_and_split(
        seed=args.seed,
        test_size=args.test_size,
        stratify=not args.no_stratify,
    )

    Xtr, ytr, artifacts = fit_preprocess(df_train, target_column=TARGET_COLUMN)
    models = train_models(Xtr, ytr, random_state=args.seed)
    paths = save_models(models, artifacts)

    print(f"{TRAIN_TAG} Models saved successfully.")
    for k, p in paths.items():
        print(f"{TRAIN_TAG}   {k}: {p}")

    return 0


def cmd_eval(args) -> int:
    logger.info(
        "CLI eval started: seed=%s test_size=%s stratify=%s model=%s explain=%s",
        args.seed,
        args.test_size,
        not args.no_stratify,
        getattr(args, "model", "all"),
        args.explain,
    )
    _df_train, df_test = _load_clean_and_split(
        seed=args.seed,
        test_size=args.test_size,
        stratify=not args.no_stratify,
    )

    try:
        models, prep = load_models()
    except FileNotFoundError:
        print(f"{EVAL_TAG} No saved models found. Run: python3 -m src.main train")
        return 1

    artifacts = _artifacts_from_prep(prep)

    Xte = transform_features(df_test, artifacts)
    yte = transform_labels(df_test, artifacts, target_column=TARGET_COLUMN)

    class_names = artifacts.label_encoder.classes_.tolist()

    # Evaluate only one model if requested
    if hasattr(args, "model") and args.model != "all":
        models = {args.model: models[args.model]}

    results = evaluate_models(models, Xte, yte, class_names=class_names)
    metrics_path = save_metrics(results)

    print(f"{EVAL_TAG} Metrics saved to: {metrics_path}")
    for name, m in results.items():
        print(
            f"{EVAL_TAG} {name}: "
            f"acc={m['accuracy']:.4f} "
            f"macro_f1={m['macro_f1']:.4f}"
        )

    if args.explain:
        rf_fi_path = save_random_forest_feature_importance(top_n=args.top_n)
        lr_fi_path = save_logistic_regression_coefficients(top_n=args.top_n)
        print(f"{EVAL_TAG} Saved feature importance to: {rf_fi_path}")
        print(f"{EVAL_TAG} Saved feature importance to: {lr_fi_path}")

    return 0


def cmd_all(args) -> int:
    logger.info("CLI all started.")
    # Export cleaned splits once for reproducibility/reporting.
    df_train, df_test = _load_clean_and_split(
        seed=args.seed,
        test_size=args.test_size,
        stratify=not args.no_stratify,
    )
    split_paths = _save_clean_splits(df_train, df_test)
    print(f"{TRAIN_TAG} Cleaned splits saved.")
    for k, p in split_paths.items():
        print(f"{TRAIN_TAG}   {k}: {p}")

    train_rc = cmd_train(args)
    if train_rc != 0:
        return train_rc
    return cmd_eval(args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="molecular-subtype-classifier",
        description="Train/evaluate models for METABRIC PAM50 subtype classification.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common_flags(p: argparse.ArgumentParser) -> None:
        p.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
        p.add_argument(
            "--test-size",
            type=float,
            default=0.2,
            help="Test split fraction (default: 0.2)",
        )
        p.add_argument(
            "--no-stratify", action="store_true", help="Disable stratified split"
        )
        p.add_argument(
            "--explain", action="store_true", help="Also write feature importance CSV"
        )
        p.add_argument(
            "--top-n", type=int, default=20, help="Top-N features to save (default: 20)"
        )

    p_train = sub.add_parser("train", help="Train models and save artifacts")
    add_common_flags(p_train)
    p_train.set_defaults(func=cmd_train)

    p_eval = sub.add_parser(
        "eval", help="Evaluate saved models and write metrics/reports"
    )
    add_common_flags(p_eval)
    p_eval.add_argument(
        "--model",
        choices=["all", "logistic_regression", "random_forest"],
        default="all",
        help="Which model to evaluate (default: all)",
    )
    p_eval.set_defaults(func=cmd_eval)

    p_all = sub.add_parser("all", help="Train + eval (one-shot)")
    add_common_flags(p_all)
    p_all.add_argument(
        "--model",
        choices=["all", "logistic_regression", "random_forest"],
        default="all",
        help="Which model to evaluate (default: all)",
    )
    p_all.set_defaults(func=cmd_all)

    return parser


def main() -> int:
    # Suppress noisy numerical warnings emitted inside scikit-learn
    warnings.filterwarnings("ignore", category=RuntimeWarning, module=r"sklearn\..*")
    warnings.filterwarnings("ignore", category=FutureWarning, module=r"sklearn\..*")

    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
