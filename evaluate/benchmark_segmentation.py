"""Benchmark and tune 3D tumor segmentation against labeled NIfTI studies."""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import SimpleITK as sitk

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import CHECKPOINTS_DIR, MODEL_PATHS, SEGMENTATION_DEVICE
from modules.segmentation import (
    MonaiBraTSSegmentationEngine,
    NnUnet3DStrongEngine,
    _brain_foreground_mask,
    _dice_score,
    _iou_score,
    _postprocess_mask,
    _reference_image,
    _surface_hausdorff,
)

BRATS_ORDER = ("t1c", "t1", "t2", "flair")
VOLUME_EXTENSIONS = (".nii.gz", ".nii", ".mha", ".mhd", ".nrrd")
ROLE_ALIASES = {
    "t1c": ("t1c", "t1ce"),
    "t1": ("t1",),
    "t2": ("t2",),
    "flair": ("flair", "t2f", "fla"),
    "seg": ("seg", "mask", "label", "labels"),
}
DEFAULT_OUTPUT_DIR = CHECKPOINTS_DIR / "benchmark"
DEFAULT_SUMMARY_PATH = DEFAULT_OUTPUT_DIR / "segmentation_eval_summary.json"
DEFAULT_TUNING_PATH = DEFAULT_OUTPUT_DIR / "segmentation_tuning.json"


@dataclass
class BenchmarkCase:
    case_id: str
    volume: np.ndarray
    label: np.ndarray
    spacing_mm: Tuple[float, float, float]
    source_files: Dict[str, str]


@dataclass
class CasePrediction:
    case_id: str
    label: np.ndarray
    spacing_mm: Tuple[float, float, float]
    reference_image: np.ndarray
    brain_mask: np.ndarray
    probabilities: Dict[str, np.ndarray]
    default_thresholds: Dict[str, float]
    default_masks: Dict[str, np.ndarray]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def strip_volume_extension(name: str) -> str:
    lower = name.lower()
    for extension in VOLUME_EXTENSIONS:
        if lower.endswith(extension):
            return name[: -len(extension)]
    return Path(name).stem


def detect_role(stem: str) -> Optional[str]:
    normalized = stem.lower()
    tokens = [token for token in re.split(r"[^a-z0-9]+", normalized) if token]
    if not tokens:
        return None

    for role, aliases in ROLE_ALIASES.items():
        if any(token in aliases for token in tokens):
            return role
        if any(normalized.endswith(f"-{alias}") or normalized.endswith(f"_{alias}") for alias in aliases):
            return role
    return None


def remove_role_suffix(stem: str, role: str) -> str:
    normalized = stem
    lower = stem.lower()
    aliases = ROLE_ALIASES.get(role, ())
    for alias in sorted(aliases, key=len, reverse=True):
        for separator in ("-", "_", "."):
            suffix = f"{separator}{alias}"
            if lower.endswith(suffix):
                return normalized[: -len(suffix)].rstrip("-_. ")
    return normalized


def read_volume(path: Path) -> Tuple[np.ndarray, Tuple[float, float, float]]:
    image = sitk.ReadImage(str(path))
    spacing_xyz = tuple(float(item) for item in image.GetSpacing())
    spacing_zyx = tuple(reversed(spacing_xyz))
    volume = sitk.GetArrayFromImage(image).astype(np.float32)
    return volume, spacing_zyx


def collect_case_files(dataset_root: Path) -> Dict[str, Dict[str, Path]]:
    grouped: Dict[str, Dict[str, Path]] = {}
    for path in sorted(dataset_root.rglob("*")):
        if not path.is_file():
            continue
        if not any(path.name.lower().endswith(extension) for extension in VOLUME_EXTENSIONS):
            continue

        stem = strip_volume_extension(path.name)
        role = detect_role(stem)
        if role is None:
            continue

        case_id = remove_role_suffix(stem, role)
        if not case_id:
            continue

        group_key = f"{path.parent.resolve()}::{case_id.lower()}"
        group = grouped.setdefault(group_key, {"case_id": case_id, "dir": path.parent})
        group.setdefault(role, path)
    return grouped


def build_benchmark_cases(dataset_root: Path, limit: Optional[int] = None) -> List[BenchmarkCase]:
    grouped = collect_case_files(dataset_root)
    cases: List[BenchmarkCase] = []

    for group in grouped.values():
        label_path = group.get("seg")
        if not label_path:
            continue

        modality_paths = [group[role] for role in BRATS_ORDER if role in group]
        if not modality_paths:
            continue

        modality_volumes = []
        spacing_reference: Optional[Tuple[float, float, float]] = None
        valid_modalities = []
        for role in BRATS_ORDER:
            path = group.get(role)
            if path is None:
                continue
            volume, spacing = read_volume(path)
            if spacing_reference is None:
                spacing_reference = spacing
            if volume.ndim != 3:
                continue
            modality_volumes.append(volume)
            valid_modalities.append(role)

        if not modality_volumes:
            continue

        label_volume, label_spacing = read_volume(label_path)
        if label_volume.ndim != 3:
            continue

        volume_shape = modality_volumes[0].shape
        if any(volume.shape != volume_shape for volume in modality_volumes):
            continue
        if label_volume.shape != volume_shape:
            continue

        spacing = spacing_reference or label_spacing
        volume = np.stack(modality_volumes, axis=0).astype(np.float32)
        label = (label_volume > 0).astype(np.uint8)

        source_files = {role: str(group[role]) for role in valid_modalities if role in group}
        source_files["seg"] = str(label_path)
        cases.append(
            BenchmarkCase(
                case_id=str(group["case_id"]),
                volume=volume,
                label=label,
                spacing_mm=tuple(float(item) for item in spacing),
                source_files=source_files,
            )
        )

    cases.sort(key=lambda item: item.case_id)
    if limit is not None:
        return cases[:limit]
    return cases


def volume_cm3(mask: np.ndarray, spacing_mm: Sequence[float]) -> float:
    voxel_volume_mm3 = float(np.prod(np.asarray(spacing_mm, dtype=np.float32)))
    return float(mask.astype(bool).sum() * voxel_volume_mm3 / 1000.0)


def hausdorff_mm(mask_a: np.ndarray, mask_b: np.ndarray, spacing_mm: Sequence[float]) -> float:
    return float(_surface_hausdorff(mask_a, mask_b) * float(np.mean(np.asarray(spacing_mm, dtype=np.float32))))


def build_mask_from_probability(
    probability: np.ndarray,
    threshold: float,
    reference_image: np.ndarray,
    brain_mask: np.ndarray,
    uncertainty: Optional[np.ndarray] = None,
) -> np.ndarray:
    raw_mask = (probability >= threshold).astype(np.uint8)
    support_core = probability >= min(threshold + 0.12, 0.90)
    cleaned, _ = _postprocess_mask(
        raw_mask,
        probability=probability.astype(np.float32),
        uncertainty=None if uncertainty is None else uncertainty.astype(np.float32),
        reference_image=reference_image.astype(np.float32),
        support_core=support_core.astype(np.uint8),
        brain_mask=brain_mask.astype(np.uint8),
    )
    if int(cleaned.sum()) == 0 and int(raw_mask.sum()) > 0:
        return raw_mask.astype(np.uint8)
    return cleaned.astype(np.uint8)


def summarize_metrics(case_metrics: List[Dict[str, float]]) -> Dict[str, float]:
    if not case_metrics:
        return {
            "mean_dice": 0.0,
            "mean_iou": 0.0,
            "mean_hausdorff_mm": 0.0,
            "mean_volume_mae_cm3": 0.0,
            "mean_volume_mape_percent": 0.0,
        }

    keys = [
        "dice",
        "iou",
        "hausdorff_mm",
        "volume_mae_cm3",
        "volume_mape_percent",
    ]
    summary = {}
    for key in keys:
        summary[f"mean_{key}"] = float(np.mean([item[key] for item in case_metrics]))
        summary[f"median_{key}"] = float(np.median([item[key] for item in case_metrics]))
    return summary


def score_summary(summary: Dict[str, float]) -> Tuple[float, float, float, float]:
    return (
        float(summary.get("mean_dice", 0.0)),
        float(summary.get("mean_iou", 0.0)),
        -float(summary.get("mean_hausdorff_mm", 0.0)),
        -float(summary.get("mean_volume_mae_cm3", 0.0)),
    )


def evaluate_masks(cases: Sequence[CasePrediction], mask_builder) -> Tuple[List[Dict[str, float]], Dict[str, float], List[Dict[str, float]]]:
    case_metrics: List[Dict[str, float]] = []
    worst_cases: List[Dict[str, float]] = []

    for case in cases:
        predicted_mask = mask_builder(case)
        gt_mask = case.label.astype(np.uint8)
        pred_volume = volume_cm3(predicted_mask, case.spacing_mm)
        gt_volume = volume_cm3(gt_mask, case.spacing_mm)
        mape = abs(pred_volume - gt_volume) / max(gt_volume, 1e-6) * 100.0
        metrics = {
            "dice": float(_dice_score(predicted_mask, gt_mask)),
            "iou": float(_iou_score(predicted_mask, gt_mask)),
            "hausdorff_mm": hausdorff_mm(predicted_mask, gt_mask, case.spacing_mm),
            "volume_mae_cm3": float(abs(pred_volume - gt_volume)),
            "volume_mape_percent": float(mape),
        }
        case_metrics.append(metrics)
        worst_cases.append(
            {
                "case_id": case.case_id,
                "dice": metrics["dice"],
                "iou": metrics["iou"],
                "hausdorff_mm": metrics["hausdorff_mm"],
            }
        )

    summary = summarize_metrics(case_metrics)
    worst_cases.sort(key=lambda item: (item["dice"], -item["hausdorff_mm"]))
    return case_metrics, summary, worst_cases[:10]


def instantiate_engines(device: str):
    engines = {}
    monai_path = MODEL_PATHS.get("monai_brats_mri_segmentation")
    if monai_path and monai_path.exists():
        engines["monai_brats_mri_segmentation"] = MonaiBraTSSegmentationEngine(monai_path, device=device)

    nnunet_path = MODEL_PATHS.get("nnunet3d_strong")
    if nnunet_path and nnunet_path.exists():
        engines["nnunet3d_strong"] = NnUnet3DStrongEngine(nnunet_path, device=device)

    return engines


def collect_predictions(cases: Sequence[BenchmarkCase], engines: Dict[str, object]) -> List[CasePrediction]:
    predictions: List[CasePrediction] = []
    for case in cases:
        reference_image = _reference_image(case.volume)
        brain_mask = _brain_foreground_mask(case.volume)
        probability_maps: Dict[str, np.ndarray] = {}
        thresholds: Dict[str, float] = {}
        masks: Dict[str, np.ndarray] = {}

        for model_key, engine in engines.items():
            prediction = engine.segment(case.volume)
            probability_maps[model_key] = prediction.probability.astype(np.float32)
            thresholds[model_key] = float(prediction.threshold)
            masks[model_key] = prediction.mask.astype(np.uint8)

        predictions.append(
            CasePrediction(
                case_id=case.case_id,
                label=case.label.astype(np.uint8),
                spacing_mm=case.spacing_mm,
                reference_image=reference_image.astype(np.float32),
                brain_mask=brain_mask.astype(np.uint8),
                probabilities=probability_maps,
                default_thresholds=thresholds,
                default_masks=masks,
            )
        )
    return predictions


def tune_single_model(
    cases: Sequence[CasePrediction],
    model_key: str,
    threshold_grid: Iterable[float],
) -> Dict[str, object]:
    best_payload: Optional[Dict[str, object]] = None

    for threshold in threshold_grid:
        _, summary, worst_cases = evaluate_masks(
            cases,
            lambda case, threshold=threshold: build_mask_from_probability(
                case.probabilities[model_key],
                threshold=threshold,
                reference_image=case.reference_image,
                brain_mask=case.brain_mask,
            ),
        )
        payload = {
            "model_key": model_key,
            "threshold": float(threshold),
            "summary": summary,
            "worst_cases": worst_cases,
        }
        if best_payload is None or score_summary(summary) > score_summary(best_payload["summary"]):
            best_payload = payload

    assert best_payload is not None
    best_payload["threshold_bias"] = float(best_payload["threshold"]) - (0.50 if model_key == "monai_brats_mri_segmentation" else 0.46)
    return best_payload


def tune_ensemble(
    cases: Sequence[CasePrediction],
    threshold_grid: Iterable[float],
    weight_grid: Iterable[float],
    consensus_grid: Iterable[float],
) -> Dict[str, object]:
    best_payload: Optional[Dict[str, object]] = None

    for monai_weight in weight_grid:
        nnunet_weight = 1.0 - float(monai_weight)
        for threshold in threshold_grid:
            for consensus_weight in consensus_grid:
                def build(case, threshold=threshold, monai_weight=monai_weight, nnunet_weight=nnunet_weight, consensus_weight=consensus_weight):
                    monai_probability = case.probabilities["monai_brats_mri_segmentation"]
                    nnunet_probability = case.probabilities["nnunet3d_strong"]
                    weighted_probability = (monai_probability * float(monai_weight)) + (nnunet_probability * float(nnunet_weight))
                    consensus_map = (
                        ((monai_probability >= threshold).astype(np.float32) + (nnunet_probability >= threshold).astype(np.float32)) / 2.0
                    )
                    blended_probability = np.clip(
                        (weighted_probability * (1.0 - float(consensus_weight))) + (consensus_map * float(consensus_weight)),
                        0.0,
                        1.0,
                    ).astype(np.float32)
                    uncertainty = np.var(np.stack([monai_probability, nnunet_probability], axis=0), axis=0).astype(np.float32)
                    return build_mask_from_probability(
                        blended_probability,
                        threshold=threshold,
                        reference_image=case.reference_image,
                        brain_mask=case.brain_mask,
                        uncertainty=uncertainty,
                    )

                _, summary, worst_cases = evaluate_masks(cases, build)
                payload = {
                    "model_key": "hybrid_segmentation_ensemble",
                    "threshold": float(threshold),
                    "threshold_bias": float(threshold) - 0.47,
                    "monai_weight": float(monai_weight),
                    "nnunet_weight": float(nnunet_weight),
                    "consensus_weight": float(consensus_weight),
                    "summary": summary,
                    "worst_cases": worst_cases,
                }
                if best_payload is None or score_summary(summary) > score_summary(best_payload["summary"]):
                    best_payload = payload

    assert best_payload is not None
    return best_payload


def default_model_metrics(cases: Sequence[CasePrediction], model_key: str) -> Dict[str, object]:
    _, summary, worst_cases = evaluate_masks(cases, lambda case: case.default_masks[model_key])
    return {
        "model_key": model_key,
        "threshold": float(np.mean([case.default_thresholds[model_key] for case in cases])),
        "summary": summary,
        "worst_cases": worst_cases,
    }


def save_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark and tune segmentation thresholds/weights against labeled NIfTI cases.")
    parser.add_argument("--dataset-root", required=True, help="Dataset root with labeled NIfTI cases.")
    parser.add_argument("--device", default=SEGMENTATION_DEVICE, help="Inference device: auto/cpu/cuda.")
    parser.add_argument("--max-cases", type=int, default=None, help="Optional limit for a quick benchmark pass.")
    parser.add_argument("--summary-json", default=str(DEFAULT_SUMMARY_PATH), help="Output path for evaluation summary JSON.")
    parser.add_argument("--tuning-json", default=str(DEFAULT_TUNING_PATH), help="Output path for tuning JSON.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root).resolve()
    summary_path = Path(args.summary_json).resolve()
    tuning_path = Path(args.tuning_json).resolve()

    cases = build_benchmark_cases(dataset_root, limit=args.max_cases)
    if not cases:
        raise SystemExit(f"Không tìm thấy case hợp lệ trong {dataset_root}")

    engines = instantiate_engines(args.device)
    if "monai_brats_mri_segmentation" not in engines:
        raise SystemExit("Không tìm thấy checkpoint MONAI để benchmark.")

    predictions = collect_predictions(cases, engines)

    monai_default = default_model_metrics(predictions, "monai_brats_mri_segmentation")
    monai_tuned = tune_single_model(predictions, "monai_brats_mri_segmentation", np.arange(0.36, 0.61, 0.02))

    nnunet_default = None
    nnunet_tuned = None
    ensemble_tuned = None
    if "nnunet3d_strong" in engines:
        nnunet_default = default_model_metrics(predictions, "nnunet3d_strong")
        nnunet_tuned = tune_single_model(predictions, "nnunet3d_strong", np.arange(0.30, 0.57, 0.02))
        ensemble_tuned = tune_ensemble(
            predictions,
            threshold_grid=np.arange(0.34, 0.59, 0.02),
            weight_grid=np.arange(0.45, 0.86, 0.05),
            consensus_grid=(0.00, 0.06, 0.10, 0.14, 0.18),
        )

    preferred = ensemble_tuned or monai_tuned
    summary_payload = {
        "generated_at": utc_now(),
        "dataset_root": str(dataset_root),
        "case_count": len(cases),
        "cases": [case.case_id for case in cases],
        "models": {
            "monai_default": monai_default,
            "monai_tuned": monai_tuned,
            "nnunet_default": nnunet_default,
            "nnunet_tuned": nnunet_tuned,
            "ensemble_tuned": ensemble_tuned,
        },
        "metrics": {
            "clean": {
                "mae_cm3": preferred["summary"]["mean_volume_mae_cm3"],
                "mape_percent": preferred["summary"]["mean_volume_mape_percent"],
                "mean_dice": preferred["summary"]["mean_dice"],
                "mean_iou": preferred["summary"]["mean_iou"],
                "mean_hausdorff_mm": preferred["summary"]["mean_hausdorff_mm"],
            },
            "raw": {
                "mae_cm3": monai_default["summary"]["mean_volume_mae_cm3"],
                "mape_percent": monai_default["summary"]["mean_volume_mape_percent"],
                "mean_dice": monai_default["summary"]["mean_dice"],
                "mean_iou": monai_default["summary"]["mean_iou"],
                "mean_hausdorff_mm": monai_default["summary"]["mean_hausdorff_mm"],
            },
        },
        "preferred_model_key": preferred["model_key"],
    }

    tuning_payload = {
        "generated_at": utc_now(),
        "dataset_root": str(dataset_root),
        "case_count": len(cases),
        "best_configs": {
            "monai_brats_mri_segmentation": {
                "threshold": monai_tuned["threshold"],
                "threshold_bias": monai_tuned["threshold_bias"],
                "mean_dice": monai_tuned["summary"]["mean_dice"],
                "mean_iou": monai_tuned["summary"]["mean_iou"],
                "mean_hausdorff_mm": monai_tuned["summary"]["mean_hausdorff_mm"],
            }
        },
    }
    if nnunet_tuned:
        tuning_payload["best_configs"]["nnunet3d_strong"] = {
            "threshold": nnunet_tuned["threshold"],
            "threshold_bias": nnunet_tuned["threshold_bias"],
            "mean_dice": nnunet_tuned["summary"]["mean_dice"],
            "mean_iou": nnunet_tuned["summary"]["mean_iou"],
            "mean_hausdorff_mm": nnunet_tuned["summary"]["mean_hausdorff_mm"],
        }
    if ensemble_tuned:
        tuning_payload["best_configs"]["hybrid_segmentation_ensemble"] = {
            "threshold": ensemble_tuned["threshold"],
            "threshold_bias": ensemble_tuned["threshold_bias"],
            "monai_weight": ensemble_tuned["monai_weight"],
            "nnunet_weight": ensemble_tuned["nnunet_weight"],
            "consensus_weight": ensemble_tuned["consensus_weight"],
            "mean_dice": ensemble_tuned["summary"]["mean_dice"],
            "mean_iou": ensemble_tuned["summary"]["mean_iou"],
            "mean_hausdorff_mm": ensemble_tuned["summary"]["mean_hausdorff_mm"],
        }

    save_json(summary_path, summary_payload)
    save_json(tuning_path, tuning_payload)

    print(f"Evaluated cases: {len(cases)}")
    print(f"Summary JSON: {summary_path}")
    print(f"Tuning JSON: {tuning_path}")
    print(f"Preferred config: {preferred['model_key']}")
    print(
        "Best mean Dice / IoU / Hausdorff(mm): "
        f"{preferred['summary']['mean_dice']:.4f} / "
        f"{preferred['summary']['mean_iou']:.4f} / "
        f"{preferred['summary']['mean_hausdorff_mm']:.3f}"
    )


if __name__ == "__main__":
    main()
