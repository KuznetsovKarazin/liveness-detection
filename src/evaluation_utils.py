#!/usr/bin/env python3
"""
Comprehensive Model Evaluation Utilities

This module provides advanced evaluation metrics, visualization tools,
and biometric-specific performance analysis for liveness detection models.
Now with robust guards against degenerate cases (single-class tests,
constant scores, empty arrays) so cross-evaluation never crashes.
"""

import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, average_precision_score, f1_score,
    matthews_corrcoef, cohen_kappa_score, balanced_accuracy_score
)
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import logging
from datetime import datetime
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

# NOTE: keep import name compatible with your project structure
# (_json_safe is used for JSON dumping)
from src.training_utils import _json_safe


class BiometricMetrics:
    """Calculate biometric-specific metrics for liveness detection"""

    @staticmethod
    def _is_single_class(y_true: np.ndarray) -> bool:
        return len(np.unique(y_true)) < 2

    @staticmethod
    def _is_constant_scores(y_scores: np.ndarray) -> bool:
        return y_scores.size == 0 or np.allclose(y_scores, y_scores[0])

    @staticmethod
    def calculate_apcer(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        APCER = FP / (FP + TN) for attack class (attack == 1)
        """
        attack_mask = (y_true == 1)
        if not np.any(attack_mask):
            return 0.0
        attack_predictions = y_pred[attack_mask]
        apcer = np.sum(attack_predictions == 0) / len(attack_predictions)
        return float(apcer)

    @staticmethod
    def calculate_bpcer(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        BPCER = FN / (FN + TP) for bonafide class (bonafide == 0)
        """
        bonafide_mask = (y_true == 0)
        if not np.any(bonafide_mask):
            return 0.0
        bonafide_predictions = y_pred[bonafide_mask]
        bpcer = np.sum(bonafide_predictions == 1) / len(bonafide_predictions)
        return float(bpcer)

    @staticmethod
    def calculate_acer(apcer: float, bpcer: float) -> float:
        return (apcer + bpcer) / 2

    @staticmethod
    def calculate_hter(far: float, frr: float) -> float:
        return (far + frr) / 2

    @staticmethod
    def calculate_eer(y_true: np.ndarray, y_scores: np.ndarray) -> Tuple[float, float]:
        """
        Equal Error Rate (EER) and corresponding threshold.
        Returns (nan, nan) if EER cannot be computed (single-class test,
        constant scores, or degenerate ROC).
        """
        # Guards
        if y_scores.size == 0 or BiometricMetrics._is_single_class(y_true) or BiometricMetrics._is_constant_scores(y_scores):
            return float('nan'), float('nan')

        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        if fpr.size == 0 or tpr.size == 0 or thresholds.size == 0:
            return float('nan'), float('nan')

        frr = 1 - tpr
        diff = np.abs(fpr - frr)
        if diff.size == 0:
            return float('nan'), float('nan')

        idx = int(np.argmin(diff))
        eer = (fpr[idx] + frr[idx]) / 2
        eer_threshold = thresholds[idx]
        return float(eer), float(eer_threshold)

    @staticmethod
    def calculate_bpcer_at_apcer(y_true: np.ndarray, y_scores: np.ndarray,
                                 target_apcer: float = 0.1) -> float:
        """
        BPCER at specific APCER (e.g. BPCER@APCER=10%).
        Returns 0.0 if not computable (no attack/bonafide samples, constant scores).
        """
        if y_scores.size == 0 or BiometricMetrics._is_constant_scores(y_scores):
            return 0.0

        attack_mask = (y_true == 1)
        bonafide_mask = (y_true == 0)
        attack_scores = y_scores[attack_mask]
        bonafide_scores = y_scores[bonafide_mask]

        if attack_scores.size == 0 or bonafide_scores.size == 0:
            return 0.0

        # Threshold that lets ~target_apcer fraction of attack pass as bonafide
        sorted_scores = np.sort(attack_scores)
        # clamp index to valid range
        threshold_idx = int(np.clip(len(sorted_scores) * target_apcer, 0, max(len(sorted_scores) - 1, 0)))
        threshold = sorted_scores[threshold_idx]

        # Bonafide rejected under this threshold (bonafide should have low attack score)
        bpcer = np.sum(bonafide_scores <= threshold) / len(bonafide_scores)
        return float(bpcer)


class ModelEvaluator:
    """Comprehensive model evaluation with visualization"""

    def __init__(self, save_dir: Path = Path("results/evaluation")):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(self.__class__.__name__)
        self._setup_logging()

    def _setup_logging(self):
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
            self.logger.propagate = False

    def evaluate_model(self, model: tf.keras.Model, test_dataset: tf.data.Dataset,
                       model_name: str, dataset_name: str,
                       save_predictions: bool = True) -> Dict[str, Any]:
        """
        Comprehensive model evaluation
        """
        self.logger.info(f"Evaluating {model_name} on {dataset_name}")

        # Predictions
        y_true_onehot, y_pred_proba = self._get_predictions(model, test_dataset)
        if y_pred_proba.ndim == 1:
            # Ensure shape (N,2) for binary; if model outputs only attack prob, create complementary
            y_pred_proba = np.stack([1 - y_pred_proba, y_pred_proba], axis=1)

        y_pred = np.argmax(y_pred_proba, axis=1)
        y_true = np.argmax(y_true_onehot, axis=1)

        # Log score distribution quick stats (helps diagnose degenerate cases)
        attack_scores = y_pred_proba[:, 1]
        unique_vals, unique_counts = np.unique(np.round(attack_scores, 6), return_counts=True)
        self.logger.info(
            f"Scores stats — min:{attack_scores.min():.6f} max:{attack_scores.max():.6f} "
            f"unique:{len(unique_vals)}"
        )
        if len(unique_vals) <= 3:
            self.logger.warning(f"Scores look near-constant: {dict(zip(unique_vals.tolist(), unique_counts.tolist()))}")

        # Standard metrics (with guards)
        standard_metrics = self._calculate_standard_metrics(y_true, y_pred, y_pred_proba)

        # Biometric metrics (with guards)
        biometric_metrics = self._calculate_biometric_metrics(y_true, y_pred, attack_scores)

        # Visualizations (robust to NaNs)
        viz_paths = self._create_visualizations(y_true, y_pred, y_pred_proba, model_name, dataset_name)

        # Results
        results = {
            "model": model_name,
            "dataset": dataset_name,
            "timestamp": datetime.now().isoformat(),
            "standard_metrics": standard_metrics,
            "biometric_metrics": biometric_metrics,
            "visualizations": viz_paths,
            "class_distribution": {
                "bonafide": int(np.sum(y_true == 0)),
                "attack": int(np.sum(y_true == 1))
            }
        }

        if save_predictions:
            pred_path = self._save_predictions(y_true, y_pred, y_pred_proba, model_name, dataset_name)
            results["predictions_file"] = str(pred_path)

        # Persist
        results_path = self.save_dir / f"{model_name}_{dataset_name}_evaluation.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2, default=_json_safe)

        self.logger.info(f"Evaluation complete. Results saved to {results_path}")
        return results

    def _get_predictions(self, model: tf.keras.Model, dataset: tf.data.Dataset) -> Tuple[np.ndarray, np.ndarray]:
        """Run model on dataset and collect (y_true_onehot, y_pred_proba)"""
        y_true = []
        y_pred = []
        for x_batch, y_batch in tqdm(dataset, desc="Getting predictions"):
            predictions = model.predict(x_batch, verbose=0)
            y_pred.extend(predictions)
            y_true.extend(y_batch.numpy())
        return np.array(y_true), np.array(y_pred)

    def _calculate_standard_metrics(self, y_true: np.ndarray, y_pred: np.ndarray,
                                    y_pred_proba: np.ndarray) -> Dict[str, Any]:
        """Standard classification metrics with robust ROC/PR handling"""

        # Confusion matrix (guard: if some class missing, sklearn still returns 2x2 if labels present)
        labels_present = np.unique(y_true)
        if len(labels_present) == 1:
            # Force both classes for consistent cm shape
            cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        else:
            cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        specificity = tn / max(tn + fp, 1)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        mcc = matthews_corrcoef(y_true, y_pred) if len(np.unique(y_pred)) > 1 else 0.0
        kappa = cohen_kappa_score(y_true, y_pred) if len(np.unique(y_pred)) > 1 else 0.0
        balanced_acc = balanced_accuracy_score(y_true, y_pred)

        # ROC / PR: only if both classes in y_true and scores not constant
        attack_scores = y_pred_proba[:, 1]
        can_curve = (len(np.unique(y_true)) > 1) and (not BiometricMetrics._is_constant_scores(attack_scores))

        if can_curve:
            fpr, tpr, _ = roc_curve(y_true, attack_scores)
            roc_auc = float(auc(fpr, tpr)) if fpr.size and tpr.size else float('nan')

            prec_curve, rec_curve, _ = precision_recall_curve(y_true, attack_scores)
            pr_auc = float(average_precision_score(y_true, attack_scores)) if rec_curve.size and prec_curve.size else float('nan')
        else:
            roc_auc = float('nan')
            pr_auc = float('nan')

        report = classification_report(
            y_true, y_pred, target_names=['Bonafide', 'Attack'], output_dict=True, zero_division=0
        )

        return {
            "confusion_matrix": cm.tolist(),
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "specificity": float(specificity),
            "f1_score": float(f1),
            "matthews_corrcoef": float(mcc),
            "cohen_kappa": float(kappa),
            "balanced_accuracy": float(balanced_acc),
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "classification_report": report,
            "true_positives": int(tp),
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn)
        }

    def _calculate_biometric_metrics(self, y_true: np.ndarray, y_pred: np.ndarray,
                                     y_scores: np.ndarray) -> Dict[str, float]:
        """Biometric metrics with robust EER computation"""
        apcer = BiometricMetrics.calculate_apcer(y_true, y_pred)
        bpcer = BiometricMetrics.calculate_bpcer(y_true, y_pred)
        acer = BiometricMetrics.calculate_acer(apcer, bpcer)

        far = np.sum((y_true == 0) & (y_pred == 1)) / max(np.sum(y_true == 0), 1)
        frr = np.sum((y_true == 1) & (y_pred == 0)) / max(np.sum(y_true == 1), 1)
        hter = BiometricMetrics.calculate_hter(far, frr)

        eer, eer_threshold = BiometricMetrics.calculate_eer(y_true, y_scores)

        bpcer_at_apcer = {
            "0.1": BiometricMetrics.calculate_bpcer_at_apcer(y_true, y_scores, 0.1),
            "0.05": BiometricMetrics.calculate_bpcer_at_apcer(y_true, y_scores, 0.05),
            "0.01": BiometricMetrics.calculate_bpcer_at_apcer(y_true, y_scores, 0.01)
        }

        return {
            "apcer": float(apcer),
            "bpcer": float(bpcer),
            "acer": float(acer),
            "far": float(far),
            "frr": float(frr),
            "hter": float(hter),
            "eer": float(eer) if not np.isnan(eer) else float('nan'),
            "eer_threshold": float(eer_threshold) if not np.isnan(eer_threshold) else float('nan'),
            "bpcer_at_apcer": bpcer_at_apcer
        }

    # ---------- Visualization helpers (robust to NaNs) ----------

    def _create_visualizations(self, y_true: np.ndarray, y_pred: np.ndarray,
                               y_pred_proba: np.ndarray, model_name: str,
                               dataset_name: str) -> Dict[str, str]:

        viz_dir = self.save_dir / "visualizations" / f"{model_name}_{dataset_name}"
        viz_dir.mkdir(parents=True, exist_ok=True)

        paths: Dict[str, str] = {}
        paths["confusion_matrix"] = str(self._plot_confusion_matrix(y_true, y_pred, viz_dir, model_name, dataset_name))
        paths["roc_curve"] = str(self._plot_roc_curve(y_true, y_pred_proba[:, 1], viz_dir, model_name, dataset_name))
        paths["pr_curve"] = str(self._plot_pr_curve(y_true, y_pred_proba[:, 1], viz_dir, model_name, dataset_name))
        paths["score_distribution"] = str(self._plot_score_distribution(y_true, y_pred_proba[:, 1], viz_dir, model_name, dataset_name))
        paths["error_analysis"] = str(self._plot_error_analysis(y_true, y_pred, y_pred_proba[:, 1], viz_dir, model_name, dataset_name))
        paths["det_curve"] = str(self._plot_det_curve(y_true, y_pred_proba[:, 1], viz_dir, model_name, dataset_name))
        return paths

    def _plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray,
                               save_dir: Path, model_name: str, dataset_name: str) -> Path:
        plt.figure(figsize=(10, 8))
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

        cm_norm = cm.astype('float') / np.maximum(cm.sum(axis=1)[:, np.newaxis], 1)
        annotations = np.array([[f"{cm[i, j]}\n({cm_norm[i, j]:.1%})" for j in range(2)] for i in range(2)])

        sns.heatmap(cm, annot=annotations, fmt='', cmap='Blues', cbar=True,
                    xticklabels=['Bonafide', 'Attack'], yticklabels=['Bonafide', 'Attack'])

        plt.title(f'Confusion Matrix - {model_name} on {dataset_name}', fontsize=16)
        plt.ylabel('True Label', fontsize=14)
        plt.xlabel('Predicted Label', fontsize=14)

        tn, fp, fn, tp = cm.ravel()
        accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        metrics_text = f'Accuracy: {accuracy:.3f}\nPrecision: {precision:.3f}\nRecall: {recall:.3f}'
        plt.text(0.02, 0.98, metrics_text, transform=plt.gca().transAxes,
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                 verticalalignment='top', fontsize=10)

        plt.tight_layout()
        save_path = save_dir / "confusion_matrix.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        return save_path

    def _plot_roc_curve(self, y_true: np.ndarray, y_scores: np.ndarray,
                        save_dir: Path, model_name: str, dataset_name: str) -> Path:
        plt.figure(figsize=(10, 8))

        can_curve = (len(np.unique(y_true)) > 1) and (not BiometricMetrics._is_constant_scores(y_scores))
        if not can_curve:
            # Render placeholder
            plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
            plt.title(f'ROC Curve - {model_name} on {dataset_name}\n(insufficient variance / single class)')
            plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
            plt.legend(loc='lower right'); plt.grid(True, alpha=0.3)
            save_path = save_dir / "roc_curve.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight'); plt.close()
            return save_path

        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr) if fpr.size and tpr.size else float('nan')
        eer, thr = BiometricMetrics.calculate_eer(y_true, y_scores)

        plt.plot(fpr, tpr, linewidth=2, label=f'ROC (AUC = {roc_auc:.3f})')
        if not np.isnan(eer) and thresholds.size:
            # pick closest threshold index safely
            idx = np.argmin(np.abs(thresholds - thr))
            if 0 <= idx < len(fpr):
                plt.plot(fpr[idx], tpr[idx], 'ro', markersize=8, label=f'EER = {eer:.3f}')

        plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
        plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {model_name} on {dataset_name}')
        plt.legend(loc='lower right'); plt.grid(True, alpha=0.3)
        plt.xlim([0, 1]); plt.ylim([0, 1])

        save_path = save_dir / "roc_curve.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        return save_path

    def _plot_pr_curve(self, y_true: np.ndarray, y_scores: np.ndarray,
                       save_dir: Path, model_name: str, dataset_name: str) -> Path:
        plt.figure(figsize=(10, 8))

        can_curve = (len(np.unique(y_true)) > 1) and (not BiometricMetrics._is_constant_scores(y_scores))
        if not can_curve:
            plt.title(f'Precision-Recall - {model_name} on {dataset_name}\n(insufficient variance / single class)')
            plt.xlabel('Recall'); plt.ylabel('Precision')
            plt.grid(True, alpha=0.3)
            save_path = save_dir / "pr_curve.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight'); plt.close()
            return save_path

        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        pr_auc = average_precision_score(y_true, y_scores)

        plt.plot(recall, precision, linewidth=2, label=f'PR (AP = {pr_auc:.3f})')
        baseline = np.sum(y_true) / len(y_true) if len(y_true) else 0.0
        plt.axhline(y=baseline, color='k', linestyle='--', linewidth=1, label=f'Baseline (AP = {baseline:.3f})')

        plt.xlabel('Recall'); plt.ylabel('Precision')
        plt.title(f'Precision-Recall - {model_name} on {dataset_name}')
        plt.legend(loc='lower left'); plt.grid(True, alpha=0.3)
        plt.xlim([0, 1]); plt.ylim([0, 1])

        save_path = save_dir / "pr_curve.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        return save_path

    def _plot_score_distribution(self, y_true: np.ndarray, y_scores: np.ndarray,
                                 save_dir: Path, model_name: str, dataset_name: str) -> Path:
        plt.figure(figsize=(12, 8))

        bonafide_scores = y_scores[y_true == 0]
        attack_scores = y_scores[y_true == 1]

        bins = np.linspace(0, 1, 50)
        plt.hist(bonafide_scores, bins=bins, alpha=0.5, label='Bonafide', density=True)
        plt.hist(attack_scores, bins=bins, alpha=0.5, label='Attack', density=True)

        eer, thr = BiometricMetrics.calculate_eer(y_true, y_scores)
        if not np.isnan(thr):
            plt.axvline(x=thr, linestyle='--', linewidth=2, label=f'EER thr = {thr:.3f}')

        plt.xlabel('Prediction Score'); plt.ylabel('Density')
        plt.title(f'Score Distribution - {model_name} on {dataset_name}')
        plt.legend(); plt.grid(True, alpha=0.3)

        save_path = save_dir / "score_distribution.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        return save_path

    def _plot_error_analysis(self, y_true: np.ndarray, y_pred: np.ndarray,
                             y_scores: np.ndarray, save_dir: Path,
                             model_name: str, dataset_name: str) -> Path:
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        errors = (y_true != y_pred)

        # 1) Error rate by confidence
        ax = axes[0, 0]
        confidence_bins = np.linspace(0, 1, 11)
        error_rates = []
        for i in range(len(confidence_bins) - 1):
            mask = (y_scores >= confidence_bins[i]) & (y_scores < confidence_bins[i + 1])
            total = np.sum(mask)
            error_rates.append(np.sum(errors[mask]) / total if total > 0 else 0.0)
        ax.bar(confidence_bins[:-1], error_rates, width=0.08, alpha=0.7)
        ax.set_xlabel('Confidence Score'); ax.set_ylabel('Error Rate'); ax.set_title('Error Rate by Confidence')
        ax.grid(True, alpha=0.3)

        # 2) Confidence distribution for correct vs errors
        ax = axes[0, 1]
        correct_scores = y_scores[~errors]
        error_scores = y_scores[errors]
        ax.hist(correct_scores, bins=30, alpha=0.5, label='Correct', density=True)
        ax.hist(error_scores, bins=30, alpha=0.5, label='Errors', density=True)
        ax.set_xlabel('Confidence Score'); ax.set_ylabel('Density'); ax.set_title('Confidence Distribution')
        ax.legend(); ax.grid(True, alpha=0.3)

        # 3) False Positives
        ax = axes[1, 0]
        fp_mask = (y_true == 0) & (y_pred == 1)
        fp_scores = y_scores[fp_mask]
        if fp_scores.size:
            ax.hist(fp_scores, bins=20, alpha=0.7)
            ax.axvline(x=np.mean(fp_scores), linestyle='--', label=f'Mean: {np.mean(fp_scores):.3f}')
        ax.set_xlabel('Confidence Score'); ax.set_ylabel('Count'); ax.set_title(f'False Positives (n={len(fp_scores)})')
        ax.legend(); ax.grid(True, alpha=0.3)

        # 4) False Negatives
        ax = axes[1, 1]
        fn_mask = (y_true == 1) & (y_pred == 0)
        fn_scores = y_scores[fn_mask]
        if fn_scores.size:
            ax.hist(fn_scores, bins=20, alpha=0.7)
            ax.axvline(x=np.mean(fn_scores), linestyle='--', label=f'Mean: {np.mean(fn_scores):.3f}')
        ax.set_xlabel('Confidence Score'); ax.set_ylabel('Count'); ax.set_title(f'False Negatives (n={len(fn_scores)})')
        ax.legend(); ax.grid(True, alpha=0.3)

        plt.suptitle(f'Error Analysis - {model_name} on {dataset_name}')
        plt.tight_layout()

        save_path = save_dir / "error_analysis.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        return save_path

    def _plot_det_curve(self, y_true: np.ndarray, y_scores: np.ndarray,
                        save_dir: Path, model_name: str, dataset_name: str) -> Path:
        from scipy.stats import norm

        plt.figure(figsize=(10, 8))

        can_curve = (len(np.unique(y_true)) > 1) and (not BiometricMetrics._is_constant_scores(y_scores))
        if not can_curve:
            plt.title(f'DET - {model_name} on {dataset_name}\n(insufficient variance / single class)')
            save_path = save_dir / "det_curve.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight'); plt.close()
            return save_path

        fpr, tpr, _ = roc_curve(y_true, y_scores)
        fnr = 1 - tpr

        # Convert to normal deviate; filter infinities
        with np.errstate(divide='ignore'):
            fpr_norm = norm.ppf(fpr)
            fnr_norm = norm.ppf(fnr)
        mask = np.isfinite(fpr_norm) & np.isfinite(fnr_norm)
        fpr_norm = fpr_norm[mask]; fnr_norm = fnr_norm[mask]
        fpr = fpr[mask]; fnr = fnr[mask]

        if fpr_norm.size and fnr_norm.size:
            plt.plot(fpr_norm, fnr_norm, linewidth=2)

            # EER point if possible
            if fpr.size and fnr.size:
                idx = int(np.argmin(np.abs(fpr - fnr)))
                plt.plot(fpr_norm[idx], fnr_norm[idx], 'ro', markersize=8,
                         label=f'EER ≈ {(fpr[idx] + fnr[idx]) / 2:.3f}')

        # Axis ticks in % space
        ticks = np.array([0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50])
        tick_locs = norm.ppf(ticks / 100)
        plt.xticks(tick_locs, [str(t) for t in ticks])
        plt.yticks(tick_locs, [str(t) for t in ticks])

        plt.xlabel('False Positive Rate (%)'); plt.ylabel('False Negative Rate (%)')
        plt.title(f'DET Curve - {model_name} on {dataset_name}')
        plt.legend(); plt.grid(True, alpha=0.3)

        plt.xlim([norm.ppf(0.001), norm.ppf(0.5)])
        plt.ylim([norm.ppf(0.001), norm.ppf(0.5)])

        save_path = save_dir / "det_curve.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        return save_path

    def _save_predictions(self, y_true: np.ndarray, y_pred: np.ndarray,
                          y_pred_proba: np.ndarray, model_name: str,
                          dataset_name: str) -> Path:
        pred_dir = self.save_dir / "predictions"
        pred_dir.mkdir(exist_ok=True)

        df = pd.DataFrame({
            'true_label': y_true,
            'predicted_label': y_pred,
            'bonafide_prob': y_pred_proba[:, 0],
            'attack_prob': y_pred_proba[:, 1],
            'confidence': np.max(y_pred_proba, axis=1),
            'correct': y_true == y_pred
        })

        save_path = pred_dir / f"{model_name}_{dataset_name}_predictions.csv"
        df.to_csv(save_path, index=False)
        return save_path

    def compare_models(self, evaluation_results: List[Dict[str, Any]],
                       save_path: Optional[Path] = None) -> Dict[str, Any]:
        if not evaluation_results:
            return {}

        comparison_data = []
        for result in evaluation_results:
            comparison_data.append({
                'model': result['model'],
                'dataset': result['dataset'],
                'accuracy': result['standard_metrics']['accuracy'],
                'precision': result['standard_metrics']['precision'],
                'recall': result['standard_metrics']['recall'],
                'f1_score': result['standard_metrics']['f1_score'],
                'roc_auc': result['standard_metrics']['roc_auc'],
                'apcer': result['biometric_metrics']['apcer'],
                'bpcer': result['biometric_metrics']['bpcer'],
                'acer': result['biometric_metrics']['acer'],
                'eer': result['biometric_metrics']['eer']
            })

        df = pd.DataFrame(comparison_data)
        self._plot_model_comparison(df, save_path)

        summary = {
            'best_accuracy': df.loc[df['accuracy'].idxmax()].to_dict(),
            'best_acer': df.loc[df['acer'].idxmin()].to_dict(),
            'best_eer': df.loc[df['eer'].idxmin()].to_dict(),
            'comparison_table': df.to_dict('records')
        }

        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(save_path.with_suffix('.csv'), index=False)
            with open(save_path.with_suffix('.json'), 'w') as f:
                json.dump(summary, f, indent=2, default=_json_safe)

        return summary

    def _plot_model_comparison(self, df: pd.DataFrame, save_path: Optional[Path] = None):
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # 1. Accuracy comparison
        ax = axes[0, 0]
        df.pivot(index='model', columns='dataset', values='accuracy').plot(kind='bar', ax=ax)
        ax.set_ylabel('Accuracy'); ax.set_title('Accuracy Comparison'); ax.legend(title='Dataset', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)

        # 2. ACER comparison
        ax = axes[0, 1]
        df.pivot(index='model', columns='dataset', values='acer').plot(kind='bar', ax=ax)
        ax.set_ylabel('ACER (lower better)'); ax.set_title('ACER Comparison'); ax.legend(title='Dataset', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)

        # 3. EER comparison
        ax = axes[1, 0]
        df.pivot(index='model', columns='dataset', values='eer').plot(kind='bar', ax=ax)
        ax.set_ylabel('EER (lower better)'); ax.set_title('EER Comparison'); ax.legend(title='Dataset', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)

        # 4. ROC-AUC comparison
        ax = axes[1, 1]
        df.pivot(index='model', columns='dataset', values='roc_auc').plot(kind='bar', ax=ax)
        ax.set_ylabel('ROC-AUC'); ax.set_title('ROC-AUC Comparison'); ax.legend(title='Dataset', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(Path(save_path), dpi=300, bbox_inches='tight')
        else:
            # default save inside evaluation dir
            out = self.save_dir / "comparison" / "model_comparison.png"
            out.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(out, dpi=300, bbox_inches='tight')
        plt.close()
