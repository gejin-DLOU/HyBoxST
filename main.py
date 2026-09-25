import os

import torch.utils
import torch.utils.data

import numpy as np 
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F

import torchmetrics
from torchmetrics.regression import (
    PearsonCorrCoef,
    ConcordanceCorrCoef,
    MeanSquaredError,
    MeanAbsoluteError,
    ExplainedVariance
)


from datetime import datetime
import json
import argparse

from HyBoxST.datasets.hyboxst_dataset import HyBoxSTDataset
from HyBoxST.model import HyBoxSTBase, HyBoxST
from HyBoxST.utils.utils import fix_seed, split_list_numpy, EarlyStopping
from pretrained.UNI.uni import get_encoder


MODEL_NAME_HYBOXST_BASE = "hyboxst_base"
MODEL_NAME_HYBOXST = "hyboxst"
SUPPORTED_MODEL_NAMES = {MODEL_NAME_HYBOXST_BASE, MODEL_NAME_HYBOXST}


def validate_model_name(model_name: str) -> str:
    model_key = str(model_name).lower()
    if model_key not in SUPPORTED_MODEL_NAMES:
        supported = ", ".join(sorted(SUPPORTED_MODEL_NAMES))
        raise ValueError(f"model {model_name} is not supported. Supported names: {supported}")
    return model_key

def json_serializer(obj):
    if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                      np.int16, np.int32, np.int64, np.uint8,
                      np.uint16, np.uint32, np.uint64)):
        return int(obj)
    elif isinstance(obj, (np.float_, np.float16, np.float32, 
                          np.float64)):
        return float(obj)
    elif isinstance(obj, (np.ndarray,)):  
        return obj.tolist()
    elif isinstance(obj, (torch.Tensor)):  
        return obj.cpu().numpy().tolist()
    raise TypeError(f"Type {type(obj)} not serializable")


class GeneWiseLinearCalibrator(nn.Module):
    def __init__(self, num_genes: int):
        super().__init__()
        self.register_buffer("center", torch.zeros(num_genes))
        self.register_buffer("scale", torch.ones(num_genes))
        self.register_buffer("bias", torch.zeros(num_genes))

    @torch.no_grad()
    def fit(
            self,
            preds: torch.Tensor,
            labels: torch.Tensor,
            mode: str = "scale_only",
            shrinkage: float = 0.9,
            min_scale: float = 0.8,
            max_scale: float = 1.25,
            eps: float = 1e-6
    ):
        preds = preds.float()
        labels = labels.float()
        pred_mean = preds.mean(dim=0)
        label_mean = labels.mean(dim=0)
        pred_centered = preds - pred_mean
        label_centered = labels - label_mean
        variance = pred_centered.pow(2).mean(dim=0)
        covariance = (pred_centered * label_centered).mean(dim=0)

        if mode == "identity":
            scale = torch.ones_like(pred_mean)
            bias = torch.zeros_like(pred_mean)
        else:
            raw_scale = covariance / variance.clamp_min(eps)
            raw_scale = torch.where(raw_scale > 0, raw_scale, torch.ones_like(raw_scale))
            raw_scale = torch.nan_to_num(raw_scale, nan=1.0, posinf=max_scale, neginf=min_scale)
            raw_scale = raw_scale.clamp(min=min_scale, max=max_scale)
            scale = 1.0 + (raw_scale - 1.0) * (1.0 - shrinkage)

            if mode == "scale_bias":
                bias = (label_mean - pred_mean) * (1.0 - shrinkage)
            elif mode == "scale_only":
                bias = torch.zeros_like(pred_mean)
            else:
                raise ValueError(f"Unsupported calibration mode: {mode}")

        self.center.copy_(pred_mean)
        self.scale.copy_(scale)
        self.bias.copy_(bias)
        return self

    def forward(self, preds: torch.Tensor):
        center = self.center.to(preds.device)
        scale = self.scale.to(preds.device)
        bias = self.bias.to(preds.device)
        return center + (preds - center) * scale + bias


class TrainableParameterEMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad and param.data.is_floating_point():
                self.shadow[name] = param.detach().clone()

    @torch.no_grad()
    def update(self, model: nn.Module):
        for name, param in model.named_parameters():
            if name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(param.detach(), alpha=1.0 - self.decay)

    @torch.no_grad()
    def apply_to(self, model: nn.Module):
        self.backup = {}
        for name, param in model.named_parameters():
            if name in self.shadow:
                self.backup[name] = param.detach().clone()
                param.data.copy_(self.shadow[name].to(param.device, dtype=param.dtype))

    @torch.no_grad()
    def restore(self, model: nn.Module):
        for name, param in model.named_parameters():
            if name in self.backup:
                param.data.copy_(self.backup[name].to(param.device, dtype=param.dtype))
        self.backup = {}

    def state_dict(self):
        return {
            "decay": self.decay,
            "shadow": {name: value.cpu() for name, value in self.shadow.items()},
        }

    def load_state_dict(self, state_dict):
        self.decay = state_dict.get("decay", self.decay)
        self.shadow = {
            name: value.detach().clone()
            for name, value in state_dict.get("shadow", {}).items()
        }
        self.backup = {}


def load_dataset(
        args:argparse.Namespace, 
        train_samples: list,
        val_samples: list,
        test_samples: list,
        selected_genes: list,
        preprocess=None,
        base_width=None
    ):

    model_name = validate_model_name(args.model_name)
    pathway_resource_dir = args.pathway_resource_dir if model_name == MODEL_NAME_HYBOXST else None

    if model_name in [MODEL_NAME_HYBOXST_BASE, MODEL_NAME_HYBOXST]:
        train_dataset = HyBoxSTDataset(
            slidename_lst=train_samples, selected_genes=selected_genes, phase='train',
            data_path=args.data_path, process_path=args.process_path, 
            preprocess=preprocess, base_width=base_width,
            pathway_resource_dir=pathway_resource_dir
        )
        val_dataset = HyBoxSTDataset(
            slidename_lst=val_samples, selected_genes=selected_genes, phase='test', 
            data_path=args.data_path, process_path=args.process_path, 
            preprocess=preprocess, base_width=base_width,
            pathway_resource_dir=pathway_resource_dir
        )
        test_dataset = HyBoxSTDataset(
            slidename_lst=test_samples, selected_genes=selected_genes, phase='test',
            data_path=args.data_path, process_path=args.process_path, 
            preprocess=preprocess, base_width=base_width,
            pathway_resource_dir=pathway_resource_dir
        )
    else:
        raise ValueError(f'model {args.model_name} is not supported')
    
    return train_dataset, val_dataset, test_dataset


def load_model(
        args:argparse.Namespace,
        selected_genes:list,
        image_encoder=None
):
    if args.img_pretrained_model == 'uni':
        image_dim=1024
    else:
        raise ValueError(f'img_pretrained_model {args.img_pretrained_model} is not supported')

    model_kwargs = dict(
            image_dim=image_dim,
            gene_dim=len(selected_genes),
            emb_dim=args.emb_dim,
            num_outputs=len(selected_genes),
            mlp_ratio=2.0,
            image_dropout=0.1,
            gene_dropout=args.gene_dropout,
            decoder_dropout=args.image_dropout,
            predict_norm=args.predict_norm,
            entail_weight=args.entail_weight,
            alignment_beta=args.alignment_beta,
            image_encoder=image_encoder,
            lora_rank=8,
            lora_alpha=16,
            image_encoder_name=args.img_pretrained_model,
            last_layer=args.last_layer,
            is_lora_ffn=args.is_lora_ffn
        )
    model_name = validate_model_name(args.model_name)
    if model_name == MODEL_NAME_HYBOXST_BASE:
        model = HyBoxSTBase(**model_kwargs)
    elif model_name == MODEL_NAME_HYBOXST:
        mask_path = os.path.join(args.pathway_resource_dir, 'gene_pathway_mask.npy')
        if not os.path.exists(mask_path):
            raise FileNotFoundError(
                f"Pathway mask not found: {mask_path}. "
                "Run scripts/build_pathway_resources.py before training HyBoxST."
            )
        gene_pathway_mask = torch.from_numpy(np.load(mask_path)).float()
        model = HyBoxST(
            gene_pathway_mask=gene_pathway_mask,
            pathway_box_dim=args.pathway_box_dim,
            pathway_loss_weight=args.pathway_loss_weight,
            box_loss_weight=args.box_loss_weight,
            prediction_mae_weight=args.prediction_mae_weight,
            prediction_mean_weight=args.prediction_mean_weight,
            prediction_std_weight=args.prediction_std_weight,
            prediction_corr_weight=args.prediction_corr_weight,
            prediction_corr_min_label_std=args.prediction_corr_min_label_std,
            prediction_corr_train_only=not args.prediction_corr_eval_loss,
            high_expression_loss_weight=args.high_expression_loss_weight,
            direct_pathway_loss_weight=args.direct_pathway_loss_weight,
            use_query_box=not args.disable_query_box,
            query_box_temperature=args.query_box_temperature,
            min_query_box_size=args.min_query_box_size,
            query_mix_init=args.query_mix_init,
            use_gene_calibration=not args.disable_gene_calibration,
            **model_kwargs
        )
    else:
        raise ValueError(f'model {args.model_name} is not supported')

    return model
class Trainer:
    def __init__(
            self,
            model:torch.nn.Module,
            model_name:str,
            optimizer:torch.optim.Optimizer,
            scaler,
            total_epoch:int,
            train_loader:torch.utils.data.DataLoader,
            val_loader:torch.utils.data.DataLoader,
            test_loader:torch.utils.data.DataLoader,
            train_dataset:torch.utils.data.Dataset,
            val_dataset:torch.utils.data.Dataset,
            test_dataset:torch.utils.data.Dataset,
            train_metrics,
            val_metrics,
            test_metrics,
            device,
            early_stopping,
            args:argparse.Namespace
        ) -> None:

        self.model = model
        self.model_name = model_name
        self.optimizer = optimizer
        self.scaler = scaler
        self.total_epoch = total_epoch
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.train_metrics = train_metrics
        self.val_metrics = val_metrics
        self.test_metrics = test_metrics
        self.args = args
        self.device = device
        self.early_stopping = early_stopping

        self.ema = None
        self.use_ema_for_test = False

        self.is_train_metric = True

        self.predict_samples = []
        self.predict_labels = []
        self.gene_calibrator = None
        self.distribution_reference = None
        self.distribution_calibration_last_stats = None
        self.distribution_calibration_gate_stats = None

        if not self.args.disable_ema:
            self.ema = TrainableParameterEMA(self.model, decay=self.args.ema_decay)


    def _preprocess_inputs(self, batch, dataset):

        for i in batch:
            batch[i] = batch[i].to(self.device)
        return batch

    def _apply_gene_calibration(self, preds):
        if self.gene_calibrator is None:
            return preds
        return self.gene_calibrator(preds.float())

    def _fit_distribution_calibration(self, labels):
        self.distribution_reference = None
        self.distribution_calibration_last_stats = None
        self.distribution_calibration_gate_stats = None
        if not self.args.use_distribution_calibration:
            return

        labels = labels.detach().float().cpu()
        self.distribution_reference = {
            "mean": labels.mean(),
            "std": labels.std(unbiased=False).clamp_min(1e-6),
            "count": int(labels.numel()),
            "spot_count": int(labels.shape[0]),
            "gene_count": int(labels.shape[1]),
        }

    def _adjust_distribution_block(self, preds):
        ref_mean = self.distribution_reference["mean"].to(preds.device, dtype=preds.dtype)
        ref_std = self.distribution_reference["std"].to(preds.device, dtype=preds.dtype)
        pred_mean = preds.mean()
        pred_std = preds.std(unbiased=False).clamp_min(1e-6)

        mean_strength = min(max(float(self.args.distribution_calibration_mean_strength), 0.0), 1.0)
        std_strength = min(max(float(self.args.distribution_calibration_std_strength), 0.0), 1.0)
        target_mean = pred_mean + mean_strength * (ref_mean - pred_mean)
        target_std = pred_std + std_strength * (ref_std - pred_std)
        scale = (target_std / pred_std).clamp(
            min=float(self.args.distribution_calibration_min_scale),
            max=float(self.args.distribution_calibration_max_scale),
        )
        adjusted = (preds - pred_mean) * scale + target_mean
        stats = {
            "pred_mean_before": pred_mean.item(),
            "pred_std_before": pred_std.item(),
            "pred_mean_after": adjusted.mean().item(),
            "pred_std_after": adjusted.std(unbiased=False).item(),
            "scale": scale.item(),
            "target_mean": target_mean.item(),
            "target_std": target_std.item(),
        }
        return adjusted, stats

    def _distribution_calibration_base_stats(self, applied=False):
        stats = {
            "enabled": bool(self.args.use_distribution_calibration),
            "applied": bool(applied),
        }
        if self.distribution_reference is not None:
            stats.update({
                "scope": self.args.distribution_calibration_scope,
                "mean_strength": float(self.args.distribution_calibration_mean_strength),
                "std_strength": float(self.args.distribution_calibration_std_strength),
                "min_scale": float(self.args.distribution_calibration_min_scale),
                "max_scale": float(self.args.distribution_calibration_max_scale),
                "reference_mean": self.distribution_reference["mean"].item(),
                "reference_std": self.distribution_reference["std"].item(),
                "reference_spot_count": self.distribution_reference["spot_count"],
                "reference_gene_count": self.distribution_reference["gene_count"],
            })
        if self.distribution_calibration_gate_stats is not None:
            stats["gate"] = self.distribution_calibration_gate_stats
        return stats

    def _apply_distribution_calibration(
        self,
        preds,
        sample_names=None,
        record_stats=True,
        ignore_gate=False,
        default_group_name="test_all",
    ):
        if not self.args.use_distribution_calibration or self.distribution_reference is None:
            if record_stats:
                self.distribution_calibration_last_stats = self._distribution_calibration_base_stats(applied=False)
            return preds

        gate_mode = getattr(self.args, "distribution_calibration_gate_mode", "mean_gap")
        if (
            not ignore_gate
            and self.args.distribution_calibration_auto_gate
            and gate_mode == "validation"
            and self.distribution_calibration_gate_stats is not None
            and not self.distribution_calibration_gate_stats.get("accepted", True)
        ):
            if record_stats:
                self.distribution_calibration_last_stats = self._distribution_calibration_base_stats(applied=False)
                self.distribution_calibration_last_stats["disabled_by_gate"] = True
            return preds

        preds = preds.float()
        adjusted = preds.clone()
        scope = self.args.distribution_calibration_scope
        groups = []
        if scope == "per_slide" and sample_names is not None and len(sample_names) == preds.shape[0]:
            sample_names_array = np.asarray(sample_names)
            for slide_name in dict.fromkeys(sample_names):
                indices = np.where(sample_names_array == slide_name)[0]
                groups.append((slide_name, indices))
        else:
            groups.append((default_group_name, np.arange(preds.shape[0])))

        group_stats = []
        applied_count = 0
        for group_name, indices in groups:
            if len(indices) == 0:
                continue
            index_tensor = torch.as_tensor(indices, dtype=torch.long, device=preds.device)
            block = preds.index_select(0, index_tensor)
            if (
                not ignore_gate
                and self.args.distribution_calibration_auto_gate
                and gate_mode == "mean_gap"
            ):
                ref_mean = self.distribution_reference["mean"].to(block.device, dtype=block.dtype)
                pred_mean = block.mean()
                pred_std = block.std(unbiased=False).clamp_min(1e-6)
                mean_gap = torch.abs(pred_mean - ref_mean).item()
                threshold = float(self.args.distribution_calibration_mean_gap_threshold)
                if mean_gap < threshold:
                    group_stats.append({
                        "group": str(group_name),
                        "spot_count": int(len(indices)),
                        "applied": False,
                        "disabled_by_gate": True,
                        "gate_mode": gate_mode,
                        "mean_gap": mean_gap,
                        "mean_gap_threshold": threshold,
                        "pred_mean_before": pred_mean.item(),
                        "pred_std_before": pred_std.item(),
                    })
                    continue

            block_adjusted, stats = self._adjust_distribution_block(block)
            adjusted.index_copy_(0, index_tensor, block_adjusted)
            applied_count += 1
            stats["group"] = str(group_name)
            stats["spot_count"] = int(len(indices))
            stats["applied"] = True
            if self.args.distribution_calibration_auto_gate and gate_mode == "mean_gap":
                stats["gate_mode"] = gate_mode
                stats["mean_gap"] = torch.abs(
                    block.mean() - self.distribution_reference["mean"].to(block.device, dtype=block.dtype)
                ).item()
                stats["mean_gap_threshold"] = float(self.args.distribution_calibration_mean_gap_threshold)
            group_stats.append(stats)

        if record_stats:
            self.distribution_calibration_last_stats = self._distribution_calibration_base_stats(
                applied=applied_count > 0
            )
            if self.args.distribution_calibration_auto_gate:
                self.distribution_calibration_last_stats["gate_mode"] = gate_mode
                if gate_mode == "mean_gap":
                    self.distribution_calibration_last_stats["mean_gap_threshold"] = float(
                        self.args.distribution_calibration_mean_gap_threshold
                    )
                    self.distribution_calibration_last_stats["disabled_by_gate"] = applied_count == 0
            self.distribution_calibration_last_stats["groups"] = group_stats
        return adjusted

    def _fit_distribution_calibration_gate(self, val_preds, val_labels):
        if (
            not self.args.use_distribution_calibration
            or not self.args.distribution_calibration_auto_gate
            or self.distribution_reference is None
        ):
            return

        gate_mode = getattr(self.args, "distribution_calibration_gate_mode", "mean_gap")
        if gate_mode == "mean_gap":
            self.distribution_calibration_gate_stats = {
                "enabled": True,
                "mode": gate_mode,
                "accepted": None,
                "auto_disabled": None,
                "mean_gap_threshold": float(self.args.distribution_calibration_mean_gap_threshold),
            }
            return

        val_preds = val_preds.detach().float().cpu()
        val_labels = val_labels.detach().float().cpu()
        before = self._prediction_summary(val_preds, val_labels)
        adjusted_val_preds = self._apply_distribution_calibration(
            val_preds,
            record_stats=False,
            ignore_gate=True,
            default_group_name="val_all",
        )
        after = self._prediction_summary(adjusted_val_preds, val_labels)
        before_score = self._calibration_score(before["MSE"], before["MAE"])
        after_score = self._calibration_score(after["MSE"], after["MAE"])
        score_accepted = after_score <= before_score * (
            1.0 + self.args.distribution_calibration_accept_tolerance
        )
        pcc_200_drop = before["PCC_200"] - after["PCC_200"]
        pcc_accepted = pcc_200_drop <= self.args.distribution_calibration_pcc_tolerance
        accepted = bool(score_accepted and pcc_accepted)

        self.distribution_calibration_gate_stats = {
            "enabled": True,
            "mode": gate_mode,
            "accepted": accepted,
            "auto_disabled": not accepted,
            "accept_metric": self.args.calibration_accept_metric,
            "accept_tolerance": float(self.args.distribution_calibration_accept_tolerance),
            "pcc_200_tolerance": float(self.args.distribution_calibration_pcc_tolerance),
            "before": before,
            "after": after,
            "before_score": before_score,
            "after_score": after_score,
            "score_delta": after_score - before_score,
            "pcc_200_drop": pcc_200_drop,
        }

    def _make_eval_loader(self, dataset):
        return DataLoader(
            dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=self.args.num_workers,
            pin_memory=True,
            drop_last=False,
        )

    def _eval_aug_indices(self, dataset):
        tta_num = max(1, int(self.args.tta_num))
        max_aug = int(getattr(dataset, "random_num", tta_num))
        return list(range(min(tta_num, max_aug)))

    def _collect_predictions_once(
            self,
            loader,
            dataset,
            aug_idx=0,
            apply_calibration=False
    ):
        self.model.eval()
        preds_list = []
        labels_list = []
        old_phase = getattr(dataset, "phase", None)
        old_eval_aug_idx = getattr(dataset, "eval_aug_idx", 0)
        if old_phase is not None:
            dataset.phase = "test"
        if hasattr(dataset, "eval_aug_idx"):
            dataset.eval_aug_idx = aug_idx
        try:
            with torch.no_grad():
                for batch in loader:
                    batch = self._preprocess_inputs(batch, dataset=dataset)
                    with torch.autocast(enabled=self.args.amp, device_type='cuda', dtype=torch.float16):
                        results_dict = self.model(**batch)
                    preds = results_dict['logits']
                    if apply_calibration:
                        preds = self._apply_gene_calibration(preds)
                    preds_list.append(preds.detach().float().cpu())
                    labels_list.append(batch['label'].detach().float().cpu())
        finally:
            if old_phase is not None:
                dataset.phase = old_phase
            if hasattr(dataset, "eval_aug_idx"):
                dataset.eval_aug_idx = old_eval_aug_idx
        preds = torch.cat(preds_list, dim=0)
        labels = torch.cat(labels_list, dim=0)
        return preds, labels

    def _collect_predictions(self, loader, dataset, apply_calibration=False):
        aug_indices = self._eval_aug_indices(dataset)
        preds_sum = None
        labels = None
        for aug_idx in aug_indices:
            preds, current_labels = self._collect_predictions_once(
                loader, dataset, aug_idx=aug_idx, apply_calibration=apply_calibration,
            )
            preds_sum = preds if preds_sum is None else preds_sum + preds
            if labels is None:
                labels = current_labels
        return preds_sum / len(aug_indices), labels

    def _calibration_score(self, mse, mae):
        metric = self.args.calibration_accept_metric
        if metric == "mse":
            return mse
        if metric == "mae":
            return mae
        if metric == "mse_mae":
            return mse + self.args.calibration_accept_mae_weight * mae
        raise ValueError(f"Unsupported calibration_accept_metric: {metric}")

    def _calibration_gene_score(self, mse_gene, mae_gene):
        metric = self.args.calibration_accept_metric
        if metric == "mse":
            return mse_gene
        if metric == "mae":
            return mae_gene
        if metric == "mse_mae":
            return mse_gene + self.args.calibration_accept_mae_weight * mae_gene
        raise ValueError(f"Unsupported calibration_accept_metric: {metric}")

    @torch.no_grad()
    def _apply_per_gene_calibration_acceptance(self, calibrator, val_preds, val_labels):
        calibrated_val_preds = calibrator(val_preds)
        stats = {
            "enabled": bool(self.args.calibration_per_gene_auto_disable),
            "accepted_count": None,
            "rejected_count": None,
            "accepted_ratio": None,
        }
        if not self.args.calibration_per_gene_auto_disable:
            return calibrated_val_preds, stats

        before_error = val_preds.float() - val_labels.float()
        after_error = calibrated_val_preds.float() - val_labels.float()
        before_mse_gene = before_error.pow(2).mean(dim=0)
        before_mae_gene = before_error.abs().mean(dim=0)
        after_mse_gene = after_error.pow(2).mean(dim=0)
        after_mae_gene = after_error.abs().mean(dim=0)
        before_score_gene = self._calibration_gene_score(before_mse_gene, before_mae_gene)
        after_score_gene = self._calibration_gene_score(after_mse_gene, after_mae_gene)
        accept_mask = after_score_gene <= before_score_gene * (1.0 + self.args.calibration_accept_tolerance)
        accept_mask = torch.nan_to_num(accept_mask.float(), nan=0.0).bool()

        reject_mask = ~accept_mask
        if reject_mask.any():
            calibrator.scale[reject_mask] = 1.0
            calibrator.bias[reject_mask] = 0.0
        calibrated_val_preds = calibrator(val_preds)

        accepted_count = int(accept_mask.sum().item())
        total_count = int(accept_mask.numel())
        stats = {
            "enabled": True,
            "accepted_count": accepted_count,
            "rejected_count": total_count - accepted_count,
            "accepted_ratio": accepted_count / max(total_count, 1),
            "accept_metric": self.args.calibration_accept_metric,
            "accept_tolerance": self.args.calibration_accept_tolerance,
        }
        return calibrated_val_preds, stats


    def _calibration_output_dirs(self):
        return [self.early_stopping.save_flod] if self.early_stopping.save_flod else []

    def _test_sample_names(self, num_rows):
        names = []
        sample_lengths = getattr(self.test_dataset, "sample_lengths", None)
        slidename_lst = getattr(self.test_dataset, "slidename_lst", None)
        if sample_lengths is not None and slidename_lst is not None:
            for slide_name, length in zip(slidename_lst, sample_lengths):
                names.extend([str(slide_name)] * int(length))

        if len(names) != num_rows:
            names = ["test_all"] * num_rows
        return names

    @staticmethod
    def _gene_pearson(preds, labels, eps=1e-8):
        preds = preds.float()
        labels = labels.float()
        pred_centered = preds - preds.mean(dim=0, keepdim=True)
        label_centered = labels - labels.mean(dim=0, keepdim=True)
        numerator = (pred_centered * label_centered).sum(dim=0)
        denominator = torch.sqrt(
            pred_centered.pow(2).sum(dim=0) * label_centered.pow(2).sum(dim=0)
        ).clamp_min(eps)
        corr = numerator / denominator
        return torch.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)

    @staticmethod
    def _topk_corr(corr, k):
        if corr.numel() == 0:
            return 0.0
        k = min(int(k), int(corr.numel()))
        return torch.sort(corr, descending=True).values[:k].mean().item()


    def _prediction_summary(self, preds, labels):
        preds = preds.float().cpu()
        labels = labels.float().cpu()
        error = preds - labels
        corr = self._gene_pearson(preds, labels)
        return {
            "MSE": error.pow(2).mean().item(),
            "MAE": error.abs().mean().item(),
            "PCC_10": self._topk_corr(corr, 10),
            "PCC_50": self._topk_corr(corr, 50),
            "PCC_200": self._topk_corr(corr, 200),
            "pred_mean": preds.mean().item(),
            "label_mean": labels.mean().item(),
            "mean_bias": error.mean().item(),
            "pred_std": preds.std(unbiased=False).item(),
            "label_std": labels.std(unbiased=False).item(),
        }


    def fit_gene_calibrator(self):
        if self.args.disable_post_calibration:
            return

        val_preds, val_labels = self._collect_predictions(self.val_loader, self.val_dataset)
        before_mse = F.mse_loss(val_preds, val_labels).item()
        before_mae = F.l1_loss(val_preds, val_labels).item()

        if self.args.calibration_fit_split == "val":
            fit_preds, fit_labels = val_preds, val_labels
        elif self.args.calibration_fit_split == "train":
            fit_preds, fit_labels = self._collect_predictions(
                self._make_eval_loader(self.train_dataset),
                self.train_dataset,
            )
        elif self.args.calibration_fit_split == "train_val":
            train_preds, train_labels = self._collect_predictions(
                self._make_eval_loader(self.train_dataset),
                self.train_dataset,
            )
            fit_preds = torch.cat([train_preds, val_preds], dim=0)
            fit_labels = torch.cat([train_labels, val_labels], dim=0)
        else:
            raise ValueError(f"Unsupported calibration_fit_split: {self.args.calibration_fit_split}")

        self._fit_distribution_calibration(fit_labels)

        calibrator = GeneWiseLinearCalibrator(num_genes=fit_preds.shape[1])
        calibrator.fit(
            fit_preds,
            fit_labels,
            mode=self.args.calibration_mode,
            shrinkage=self.args.calibration_shrinkage,
            min_scale=self.args.calibration_min_scale,
            max_scale=self.args.calibration_max_scale,
        )
        calibrated_val_preds, per_gene_acceptance_stats = self._apply_per_gene_calibration_acceptance(
            calibrator,
            val_preds,
            val_labels,
        )
        after_mse = F.mse_loss(calibrated_val_preds, val_labels).item()
        after_mae = F.l1_loss(calibrated_val_preds, val_labels).item()

        self.gene_calibrator = calibrator.to(self.device)
        before_score = self._calibration_score(before_mse, before_mae)
        after_score = self._calibration_score(after_mse, after_mae)
        accepted = after_score <= before_score * (1.0 + self.args.calibration_accept_tolerance)
        if self.args.calibration_auto_disable and not accepted:
            self.gene_calibrator = None
        active_val_preds = calibrated_val_preds if self.gene_calibrator is not None else val_preds
        self._fit_distribution_calibration_gate(active_val_preds, val_labels)

        stats = {
            "accepted": accepted,
            "auto_disabled": bool(self.args.calibration_auto_disable and not accepted),
            "mode": self.args.calibration_mode,
            "fit_split": self.args.calibration_fit_split,
            "tta_num": self.args.tta_num,
            "tta_aug_indices": self._eval_aug_indices(self.val_dataset),
            "ema_enabled": self.ema is not None,
            "test_ema": self.use_ema_for_test,
            "ema_decay": self.args.ema_decay,
            "fit_count": int(fit_preds.shape[0]),
            "val_count": int(val_preds.shape[0]),
            "before_mse": before_mse,
            "before_mae": before_mae,
            "after_mse": after_mse,
            "after_mae": after_mae,
            "before_score": before_score,
            "after_score": after_score,
            "accept_metric": self.args.calibration_accept_metric,
            "scale_mean": calibrator.scale.mean().item(),
            "scale_min": calibrator.scale.min().item(),
            "scale_max": calibrator.scale.max().item(),
            "bias_mean": calibrator.bias.mean().item(),
            "bias_min": calibrator.bias.min().item(),
            "bias_max": calibrator.bias.max().item(),
            "shrinkage": self.args.calibration_shrinkage,
            "per_gene_acceptance": per_gene_acceptance_stats,
        }

        for save_dir in self._calibration_output_dirs():
            os.makedirs(save_dir, exist_ok=True)
            torch.save(
                {
                    "center": calibrator.center.cpu(),
                    "scale": calibrator.scale.cpu(),
                    "bias": calibrator.bias.cpu(),
                    "stats": stats,
                },
                os.path.join(save_dir, "gene_wise_calibrator.pth"),
            )
            with open(os.path.join(save_dir, "gene_wise_calibration_stats.json"), "w", encoding="utf-8") as f:
                json.dump(stats, f, indent=4, default=json_serializer)
            if self.distribution_reference is not None:
                distribution_stats = {
                    "enabled": True,
                    "fit_split": self.args.calibration_fit_split,
                    "reference_mean": self.distribution_reference["mean"].item(),
                    "reference_std": self.distribution_reference["std"].item(),
                    "reference_count": self.distribution_reference["count"],
                    "reference_spot_count": self.distribution_reference["spot_count"],
                    "reference_gene_count": self.distribution_reference["gene_count"],
                    "scope": self.args.distribution_calibration_scope,
                    "mean_strength": self.args.distribution_calibration_mean_strength,
                    "std_strength": self.args.distribution_calibration_std_strength,
                    "min_scale": self.args.distribution_calibration_min_scale,
                    "max_scale": self.args.distribution_calibration_max_scale,
                }
                if self.distribution_calibration_gate_stats is not None:
                    distribution_stats["gate"] = self.distribution_calibration_gate_stats
                with open(os.path.join(save_dir, "distribution_calibration_stats.json"), "w", encoding="utf-8") as f:
                    json.dump(distribution_stats, f, indent=4, default=json_serializer)

    def train_step(self, batch_data):
        self.model.train()

        self.optimizer.zero_grad()
        # ---> Forward
        with torch.autocast(enabled=self.args.amp, device_type='cuda', dtype=torch.float16):

            results_dict = self.model(**batch_data)
            # ---> Loss 
            preds = results_dict['logits']
            loss = results_dict['loss']
            label = batch_data['label']
            
        # ---> backward

        self.scaler.scale(loss).backward()
        if self.model_name in SUPPORTED_MODEL_NAMES:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        if self.ema is not None:
            self.ema.update(self.model)
        batch_size = label.shape[0]
        if self.is_train_metric:
            self.train_metrics.update(preds, label)
        return loss.item() * batch_size, batch_size
    
    def test_step(self, batch_data, test_metrics, final_test=False, not_generate:bool=True):
        self.model.eval()
        with torch.no_grad():
            with torch.autocast(enabled=self.args.amp, device_type='cuda', dtype=torch.float16):
                # ---> Forward
                results_dict = self.model(**batch_data)
                # ---> Loss 
                loss = results_dict['loss']
                preds = self._apply_gene_calibration(results_dict['logits'])
                label = batch_data['label']
                if final_test:
                    self.predict_samples.append(preds.detach().float().cpu())
                    self.predict_labels.append(label.detach().float().cpu())
                # ---> metrics
                batch_size = label.shape[0] 
                if self.is_train_metric or final_test or not not_generate:
                    test_metrics.update(preds, label)

        return loss.item() * batch_size, batch_size

    def train_epoch(self, epoch):

        self.model.train()
        self.train_metrics.reset()

        total_loss = 0
        total_count = 0

        for batch in self.train_loader:
            batch = self._preprocess_inputs(batch, dataset=self.train_dataset)
            
            batch_loss, batch_size = self.train_step(batch_data=batch)
            total_loss += batch_loss
            total_count += batch_size

        train_avg_loss = total_loss / total_count

        if self.is_train_metric:
            train_epoch_result = self.train_metrics.compute()
            self.train_metrics.reset()
            train_metrics_result = {k: y.detach().cpu() for k, y in train_epoch_result.items()}
            train_PCC_sort = sorted(torch.nan_to_num(train_metrics_result['PearsonCorrCoef']))[::-1]
            train_MSE = train_metrics_result['MeanSquaredError'].mean().item()
            train_MAE = train_metrics_result['MeanAbsoluteError'].mean().item()
            train_PCC_10 = np.mean(sorted(train_PCC_sort)[::-1][:10])
            train_PCC_50 = np.mean(sorted(train_PCC_sort)[::-1][:50])
            train_PCC_200 = np.mean(sorted(train_PCC_sort)[::-1][:200])

        if self.is_train_metric:
            metrics_dict = {
                'train_loss' : train_avg_loss,
                'PCC_10' : train_PCC_10,
                'PCC_50' : train_PCC_50,
                'PCC_200' : train_PCC_200,
                'MSE' : train_MSE,
                'MAE' : train_MAE,
                'epoch' : epoch
            }
        else:
            metrics_dict = {
                'train_loss' : train_avg_loss,
                'epoch' : epoch
            }

        return train_avg_loss, metrics_dict


    def val_epoch(self, epoch):

        self.model.eval()
        self.val_metrics.reset()

        val_total_loss = 0
        val_total_count = 0

        for batch in self.val_loader:
            # load data to gpu
            batch = self._preprocess_inputs(batch, dataset=self.val_dataset)
            
            batch_loss, batch_size = self.test_step(batch_data=batch, test_metrics=self.val_metrics)
            val_total_loss += batch_loss
            val_total_count += batch_size

        val_avg_loss = val_total_loss / val_total_count
        if self.is_train_metric:
            val_epoch_result = self.val_metrics.compute()
            self.val_metrics.reset()
            val_metrics_result = {k: y.detach().cpu() for k, y in val_epoch_result.items()}
            val_PCC_sort = sorted(torch.nan_to_num(val_metrics_result['val_PearsonCorrCoef']))[::-1]
            val_CCC_sort = sorted(torch.nan_to_num(val_metrics_result['val_ConcordanceCorrCoef']))[::-1]
            val_MSE = val_metrics_result['val_MeanSquaredError'].mean().item()
            val_MAE = val_metrics_result['val_MeanAbsoluteError'].mean().item()
            val_PCC_10 = np.mean(sorted(val_PCC_sort)[::-1][:10])
            val_PCC_50 = np.mean(sorted(val_PCC_sort)[::-1][:50])
            val_PCC_200 = np.mean(sorted(val_PCC_sort)[::-1][:200])


        if self.is_train_metric:
            metrics_dict = {
                'val_loss' : val_avg_loss,
                'PCC_10' : val_PCC_10,
                'PCC_50' : val_PCC_50,
                'PCC_200' : val_PCC_200,
                'MSE' : val_MSE,
                'MAE' : val_MAE,
                'epoch' : epoch
            }
        else:
            metrics_dict = {
                'val_loss' : val_avg_loss,
                'epoch' : epoch
            }

        return val_avg_loss, metrics_dict


    def test_epoch(self, epoch, final_test=False):
        self.model.eval()
        self.test_metrics.reset()

        not_generate = True

        if final_test and (self.args.tta_num > 1 or self.args.use_distribution_calibration):
            preds, labels = self._collect_predictions(
                self.test_loader,
                self.test_dataset,
                apply_calibration=True,
            )
            sample_names = self._test_sample_names(preds.shape[0])
            preds = self._apply_distribution_calibration(preds, sample_names=sample_names)
            preds = preds.to(self.device)
            labels = labels.to(self.device)
            test_avg_loss = F.mse_loss(preds, labels).item()
            self.predict_samples.append(preds.detach().cpu())
            self.predict_labels.append(labels.detach().cpu())
            self.test_metrics.update(preds, labels)
            test_epoch_result = self.test_metrics.compute()
            self.test_metrics.reset()
            test_metrics_result = {k: y.detach().cpu() for k, y in test_epoch_result.items()}
            test_PCC_sort = sorted(torch.nan_to_num(test_metrics_result['test_PearsonCorrCoef']))[::-1]
            test_MSE = test_metrics_result['test_MeanSquaredError'].mean().item()
            test_MAE = test_metrics_result['test_MeanAbsoluteError'].mean().item()
            test_PCC_10 = np.mean(sorted(test_PCC_sort)[::-1][:10])
            test_PCC_50 = np.mean(sorted(test_PCC_sort)[::-1][:50])
            test_PCC_200 = np.mean(sorted(test_PCC_sort)[::-1][:200])
            return test_avg_loss, {
                'test_loss': test_avg_loss,
                'PCC_10': test_PCC_10,
                'PCC_50': test_PCC_50,
                'PCC_200': test_PCC_200,
                'MSE': test_MSE,
                'MAE': test_MAE,
                'epoch': epoch,
            }
        
        test_total_loss = 0
        test_total_count = 0

        for batch in self.test_loader:
            # load data to gpu
            batch = self._preprocess_inputs(batch, dataset=self.test_dataset)
            
            batch_loss, batch_size = self.test_step(batch_data=batch, test_metrics=self.test_metrics, final_test=final_test, not_generate=not_generate)
            test_total_loss += batch_loss
            test_total_count += batch_size

        test_avg_loss = test_total_loss / test_total_count
        if self.is_train_metric or final_test or not not_generate:
            test_epoch_result = self.test_metrics.compute()
            self.test_metrics.reset()
            test_metrics_result = {k: y.detach().cpu() for k, y in test_epoch_result.items()}
            test_PCC_sort = sorted(torch.nan_to_num(test_metrics_result['test_PearsonCorrCoef']))[::-1]
            test_MSE = test_metrics_result['test_MeanSquaredError'].mean().item()
            test_MAE = test_metrics_result['test_MeanAbsoluteError'].mean().item()
            test_PCC_10 = np.mean(sorted(test_PCC_sort)[::-1][:10])
            test_PCC_50 = np.mean(sorted(test_PCC_sort)[::-1][:50])
            test_PCC_200 = np.mean(sorted(test_PCC_sort)[::-1][:200])

        if self.is_train_metric or final_test or not not_generate:
            metrics_dict = {
                'test_loss' : test_avg_loss,
                'PCC_10' : test_PCC_10,
                'PCC_50' : test_PCC_50,
                'PCC_200' : test_PCC_200,
                'MSE' : test_MSE,
                'MAE' : test_MAE,
                'epoch' : epoch
            }
        else:
            metrics_dict = {
                'test_loss' : test_avg_loss,
                'epoch' : epoch
            }

        return test_avg_loss, metrics_dict


    def train(self):      


        for epoch in range(self.total_epoch):
            train_avg_loss, train_metrics_dict = self.train_epoch(epoch=epoch)

            val_avg_loss, val_metrics_dict = self.val_epoch(epoch=epoch)

            test_avg_loss, test_metrics_dict = self.test_epoch(epoch=epoch)

            self.early_stopping(val_avg_loss, self.model, epoch, train_metrics_dict, val_metrics_dict, test_metrics_dict, self.ema)
            if self.early_stopping.early_stop:
                break


    def test(self):
        self.model, self.ema = self.early_stopping.load_from_disk(
            model=self.model,
            ema=self.ema,
            device=self.device,
            best=self.args.test_checkpoint == 'best',
        )
        self.use_ema_for_test = self.ema is not None and self.args.test_ema
        if self.use_ema_for_test:
            self.ema.apply_to(self.model)

        self.fit_gene_calibrator()
        test_avg_loss, test_metrics_dict = self.test_epoch(epoch=-1, final_test=True)
        test_metrics_dict['ema_enabled'] = self.ema is not None
        test_metrics_dict['test_ema'] = self.use_ema_for_test
        test_metrics_dict['ema_decay'] = self.args.ema_decay
        test_metrics_dict['tta_num'] = self.args.tta_num
        test_metrics_dict['distribution_calibration'] = self.distribution_calibration_last_stats or {
            "enabled": bool(self.args.use_distribution_calibration),
            "applied": False,
        }
        a = self.args.fold
        with open(os.path.join(self.args.experiment_result_path, f"final_test_result_{a}.json"), "w", encoding='utf-8') as f:
            json.dump(test_metrics_dict, f, indent=4, default=json_serializer)

        predict_samples = torch.cat(self.predict_samples, dim=0).cpu()
        predict_labels = torch.cat(self.predict_labels, dim=0).cpu()


        torch.save(predict_samples, os.path.join(self.args.samples_dir, 'predict_samples.pt'))
        torch.save(predict_labels, os.path.join(self.args.samples_dir, 'label_samples.pt'))
        print(json.dumps(test_metrics_dict, indent=4, default=json_serializer))
        print(f'Results saved to: {self.args.experiment_result_path}')
        if self.use_ema_for_test:
            self.ema.restore(self.model)

def main(args):
    folder_list_path = os.path.join(args.process_path, args.folder_list_filename)
    slidename_lst = list(np.genfromtxt(folder_list_path, dtype=str))
    for out in args.slide_out.split(','):
        if out not in slidename_lst:
            continue
        slidename_lst.remove(out)
    # load selected gene list
    gene_list_path = os.path.join(args.process_path, args.gene_list_filename)
    selected_genes = list(np.genfromtxt(gene_list_path, dtype=str))
    input_gene_size = len(selected_genes)
    fix_seed(args.seed)
    args.model_name = validate_model_name(args.model_name)
    if args.model_name == MODEL_NAME_HYBOXST and args.pathway_resource_dir is None:
        args.pathway_resource_dir = os.path.join('pathway_resources', args.dataset)

    # split train and val
    if args.split_dir:
        with open(args.split_dir, 'r', encoding='utf-8') as f:
            dataset_split_json = json.load(f)
    else:
        slidename_length = len(slidename_lst)
        test_length = max(round(slidename_length * 0.1), 1)
        val_length =  max(round(slidename_length * 0.1), 1)
        train_length = slidename_length - test_length - val_length
        split_dir = os.path.join('./split', args.dataset)
        os.makedirs(split_dir, exist_ok=True)
        dataset_split = split_list_numpy(slidename_lst, [train_length, val_length, test_length])
        dataset_split_json = {
            'train_sample' : [str(i) for i in dataset_split[0]],
            'val_sample' : [str(i) for i in dataset_split[1]],
            'test_sample' : [str(i) for i in dataset_split[2]]
        }

        with open(os.path.join(split_dir, args.split_samples_filename), 'w', encoding='utf-8') as f:
            json.dump(dataset_split_json, f, indent=4)

    train_samples = dataset_split_json['train_sample']
    val_samples = dataset_split_json['val_sample']
    test_samples = dataset_split_json['test_sample']

    args.train_samples = train_samples
    args.val_samples = val_samples
    args.test_samples = test_samples
    args.selected_genes = selected_genes
    args.selected_genes_length = input_gene_size


    # load datasets 

    if args.img_pretrained_model == 'uni':
        image_dim=1024
        base_width = 224

        if args.uni_weight_path == None:
            model_image, transform_image = get_encoder(enc_name='uni', device='cpu', assets_dir='./uni_weight', token=args.huggingface_token)
        else:
            model_image, transform_image = get_encoder(enc_name='uni', device='cpu', assets_dir=args.uni_weight_path)
    else:
        raise ValueError()
    
    args.base_width = base_width
    args.image_dim = image_dim

    # save args
    args_dict = {
        k: v for k, v in vars(args).items() 
        if isinstance(v, (str, int, float, bool, list, dict, tuple, type(None)))
    }
    with open(os.path.join(args.experiment_result_path, "config.json"), "w", encoding='utf-8') as f:
        json.dump(args_dict, f, indent=4)

    train_dataset, val_dataset, test_dataset = load_dataset(
        args=args,
        train_samples=train_samples,
        val_samples=val_samples,
        test_samples=test_samples,
        selected_genes=selected_genes,
        preprocess=transform_image,
        base_width=base_width
    )
    
    # load dataloader

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, sampler=None, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=False)

    # load model
    model = load_model(
        args=args,
        selected_genes=selected_genes,
        image_encoder=model_image
    )

    model = model.to(args.device)

    # load optimizer

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)


    train_metrics = torchmetrics.MetricCollection([
        PearsonCorrCoef(num_outputs=len(selected_genes)),
        ConcordanceCorrCoef(num_outputs=len(selected_genes)),
        MeanSquaredError(num_outputs = len(selected_genes)),
        MeanAbsoluteError(num_outputs = len(selected_genes)),
        ExplainedVariance() 
    ]).to(args.device)
    test_metrics = train_metrics.clone(prefix = 'test_')
    val_metrics = train_metrics.clone(prefix = 'val_')

    if args.only_test:
        checkpoint_dir = args.checkpoint_path
    else:
        checkpoint_dir = args.checkpoint_dir

    early_stopping = EarlyStopping(
        patience=args.patience,
        save_path=os.path.join(checkpoint_dir, 'best_model.pth'),
        verbose=True,
    )

    trainer = Trainer(
        model=model,
        model_name=args.model_name,
        optimizer=optimizer,
        scaler=scaler,
        total_epoch=args.epochs,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        train_metrics=train_metrics,
        val_metrics=val_metrics,
        test_metrics=test_metrics,
        device=args.device,
        early_stopping=early_stopping,
        args=args
    )
    if not args.only_test:
        trainer.train()
    trainer.test()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_name', type=str, default='train')
    parser.add_argument('--data_root', type=str, default='./hest1k_datasets')
    parser.add_argument('--dataset', type=str, default='kidney')
    parser.add_argument('--model_name', type=str, default=MODEL_NAME_HYBOXST)
    parser.add_argument('--folder_list_filename', type=str, default='all_slide_lst.txt')
    parser.add_argument('--gene_list_filename', type=str, default='selected_gene_list.txt')
    parser.add_argument('--results_dir', type=str, default='./experiments')
    parser.add_argument('--split_dir', type=str, default=None)
    parser.add_argument('--split_samples_filename', type=str, default='sample_split.json')
    parser.add_argument('--experiment_result_path', type=str, default=None)
    parser.add_argument('--img_pretrained_model', type=str, default='uni')

    parser.add_argument('--image_dropout', type=float, default=0.1)

    parser.add_argument('--gene_dropout', type=float, default=0.1)

    parser.add_argument('--emb_dim', type=int, default=1024)
    parser.add_argument('--predict_norm', action='store_true')
    parser.add_argument('--entail_weight', type=float, default=0.4)
    parser.add_argument('--alignment_beta', type=float, default=0.2)
    parser.add_argument('--pathway_resource_dir', type=str, default=None)
    parser.add_argument('--pathway_loss_weight', type=float, default=0.1)
    parser.add_argument('--box_loss_weight', type=float, default=0.01)
    parser.add_argument('--pathway_box_dim', type=int, default=128)
    parser.add_argument('--prediction_mae_weight', type=float, default=0.1)
    parser.add_argument('--prediction_mean_weight', type=float, default=0.05)
    parser.add_argument('--prediction_std_weight', type=float, default=0.02)
    parser.add_argument('--prediction_corr_weight', type=float, default=0.0)
    parser.add_argument('--prediction_corr_min_label_std', type=float, default=1e-4)
    parser.add_argument('--prediction_corr_eval_loss', action='store_true')
    parser.add_argument('--high_expression_loss_weight', type=float, default=0.0)
    parser.add_argument('--direct_pathway_loss_weight', type=float, default=0.05)
    parser.add_argument('--disable_query_box', action='store_true')
    parser.add_argument('--query_box_temperature', type=float, default=0.1)
    parser.add_argument('--min_query_box_size', type=float, default=1e-3)
    parser.add_argument('--query_mix_init', type=float, default=-1.38629436)
    parser.add_argument('--disable_gene_calibration', action='store_true')
    parser.add_argument('--disable_post_calibration', action='store_true')
    parser.add_argument('--calibration_mode', type=str, default='scale_only', choices=['identity', 'scale_only', 'scale_bias'])
    parser.add_argument('--calibration_fit_split', type=str, default='train_val', choices=['train', 'val', 'train_val'])
    parser.add_argument('--calibration_shrinkage', type=float, default=0.9)
    parser.add_argument('--calibration_min_scale', type=float, default=0.8)
    parser.add_argument('--calibration_max_scale', type=float, default=1.25)
    parser.add_argument('--disable_calibration_auto_disable', dest='calibration_auto_disable', action='store_false')
    parser.set_defaults(calibration_auto_disable=True)
    parser.add_argument('--calibration_accept_metric', type=str, default='mse_mae', choices=['mse', 'mae', 'mse_mae'])
    parser.add_argument('--calibration_accept_mae_weight', type=float, default=0.1)
    parser.add_argument('--calibration_accept_tolerance', type=float, default=0.0)
    parser.add_argument('--calibration_per_gene_auto_disable', action='store_true')
    parser.add_argument('--use_distribution_calibration', action='store_true')
    parser.add_argument('--distribution_calibration_scope', type=str, default='global', choices=['global', 'per_slide'])
    parser.add_argument('--distribution_calibration_mean_strength', type=float, default=0.5)
    parser.add_argument('--distribution_calibration_std_strength', type=float, default=0.5)
    parser.add_argument('--distribution_calibration_min_scale', type=float, default=0.75)
    parser.add_argument('--distribution_calibration_max_scale', type=float, default=1.35)
    parser.add_argument('--distribution_calibration_auto_gate', action='store_true')
    parser.add_argument('--distribution_calibration_gate_mode', type=str, default='mean_gap', choices=['mean_gap', 'validation'])
    parser.add_argument('--distribution_calibration_mean_gap_threshold', type=float, default=0.15)
    parser.add_argument('--distribution_calibration_accept_tolerance', type=float, default=0.0)
    parser.add_argument('--distribution_calibration_pcc_tolerance', type=float, default=0.002)
    parser.add_argument('--tta_num', type=int, default=1)
    parser.add_argument('--disable_ema', action='store_true')
    parser.add_argument('--ema_decay', type=float, default=0.999)
    parser.add_argument('--no_test_ema', dest='test_ema', action='store_false')
    parser.set_defaults(test_ema=True)
    parser.add_argument('--test_checkpoint', type=str, default='final', choices=['final', 'best'])

    parser.add_argument('--last_layer', type=int, default=11)
    parser.add_argument('--is_lora_ffn', action='store_true')
    parser.add_argument('--only_test', action='store_true')
    parser.add_argument('--checkpoint_path', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--amp', action='store_false')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--num_predict_rep', type=int, default=5)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--gpu', type=int, default=0)

    parser.add_argument("--uni_weight_path", type=str, default=None)
    parser.add_argument('--huggingface_token', type=str, default=None)
    parser.add_argument('--fold', type=int, default=0)

    args = parser.parse_args()
    args.model_name = validate_model_name(args.model_name)

    # set up config
    args.data_path = os.path.join(args.data_root, args.dataset)
    args.process_path = os.path.join(args.data_path, 'processed_data')
    args.tif_path = os.path.join(args.data_path, 'wsis')
    args.st_path = os.path.join(args.data_path, 'st')
    args.slide_out = ""
    args.num_aug_ratio = 7
    # Create the experiment output directories.

    args.results_dataset_dir = os.path.join(os.path.join(os.path.join(args.results_dir, args.model_name), args.experiment_name), args.dataset)
    if not args.experiment_result_path:
        os.makedirs(args.results_dataset_dir, exist_ok=True)
        current_time = datetime.now()
        current_time = current_time.strftime('%Y-%m-%d-%H:%M:%S.%f')
        if args.split_dir:
            split_name = args.split_dir.split('.')[1].split('/')[-1]
            args.experiment_result_path = os.path.join(os.path.join(args.results_dataset_dir, split_name), current_time)
            print(args.experiment_result_path)
        else:
            args.experiment_result_path = os.path.join(args.results_dataset_dir, current_time)
        os.makedirs(args.experiment_result_path)
        
        args.samples_dir = os.path.join(args.experiment_result_path, 'samples')
        os.makedirs(args.samples_dir)
    else:
        args.samples_dir = os.path.join(args.experiment_result_path, 'samples')

    args.checkpoint_dir = os.path.join(args.experiment_result_path, 'checkpoints')


    args.device = 'cuda:' + str(args.gpu)

    main(args)
