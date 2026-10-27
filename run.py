"""Run all experiments and regenerate GitHub-ready tables and figures."""
from pathlib import Path
from src.pipeline import run_experiment

if __name__ == "__main__":
    cv_table, test_metrics, metadata = run_experiment(Path(__file__).resolve().parent)
    print("\nTraining-only five-fold CV results:")
    print(cv_table.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nSelected by CV ROC-AUC: {metadata['selected_model']}")
    print("\nHeld-out test metrics:")
    for key, value in test_metrics.items():
        print(f"  {key}: {value}")
    print("\nWrote results/ and figures/.")
