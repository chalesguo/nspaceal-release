#!/usr/bin/env python3
"""Main entry point for the anonymous N-spaceAL release."""

import os, sys, json, time, random, datetime, argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from train_utils import frozen, free
from models.resnet import ResNet18, WideResNet18
from models.lossnet import LossNet18
from models.torchvision_backbones import TorchvisionResNet50
from dataset.sampler import SubsetSequentialSampler
from dataset.data_transform import get_data
from dataset.custom_dataset import cls_dict, get_num_images
import torch.nn.functional as F

from sklearn.cluster import KMeans
from relative_innovation import (
    select_with_relative_innovation,
    adaptive_nullspace_dim,
    compute_nullspace_projection,
    compute_relative_innovation,
    rank_normalize,
)
from badge import badge_select


def sync_cuda(device=None):
    if torch.cuda.is_available():
        if device is not None and getattr(device, "type", None) == "cuda":
            torch.cuda.synchronize(device)
        else:
            torch.cuda.synchronize()


def riemannian_density(Z, bandwidth, batch_size=1000):
    """Compute exact full-pool Riemannian density with block-wise matrix ops."""
    Z_norm = Z / torch.norm(Z, dim=1, keepdim=True)
    N = Z.shape[0]

    #
    density = torch.zeros(N, device=Z.device)

    #
    for i in range(0, N, batch_size):
        end_i = min(i + batch_size, N)
        batch = Z_norm[i:end_i]

        #
        #
        for j in range(0, N, batch_size):
            end_j = min(j + batch_size, N)
            batch_j = Z_norm[j:end_j]

            #
            dists = torch.cdist(batch, batch_j, p=2) * np.sqrt(2)

            #
            kernel = torch.exp(-dists ** 2 / (2 * bandwidth ** 2))

            #
            density[i:end_i] += kernel.sum(dim=1)

    return density / N

# ============================================================
#
# ============================================================
class ExperimentTracker:
    def __init__(self, save_dir, exp_name=None):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.exp_name = exp_name or f"nullspace_ev_{ts}"
        self.exp_dir = self.save_dir / self.exp_name
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_file = self.exp_dir / "metrics.jsonl"
        self.config_file = self.exp_dir / "config.json"
        self._logger = self._setup_logger()

    def _setup_logger(self):
        import logging
        logger = logging.getLogger(self.exp_name)
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            for h in [logging.FileHandler(self.exp_dir / "tracker.log"),
                      logging.StreamHandler()]:
                h.setFormatter(logging.Formatter('%(asctime)s - %(message)s', datefmt="%H:%M:%S"))
                logger.addHandler(h)
        return logger

    def log_config(self, cfg):
        with open(self.config_file, 'w') as f:
            json.dump(cfg, f, indent=2, default=str)

    def log_cycle(self, trial, cycle, labeled_size, unlabeled_size,
                  acc, svd_time, select_time, extra=None):
        def json_ready(value):
            if isinstance(value, torch.Tensor):
                if value.numel() == 1:
                    return value.item()
                return value.detach().cpu().tolist()
            if isinstance(value, np.ndarray):
                return value.tolist()
            if isinstance(value, (np.floating, np.integer)):
                return value.item()
            if isinstance(value, dict):
                return {k: json_ready(v) for k, v in value.items()}
            if isinstance(value, (list, tuple)):
                return [json_ready(v) for v in value]
            return value
        rec = {
            "trial": trial, "cycle": cycle,
            "labeled_size": labeled_size, "unlabeled_size": unlabeled_size,
            "test_acc": round(acc, 4),
            "svd_time_s": round(svd_time, 4),
            "select_time_s": round(select_time, 4),
            "timestamp": datetime.datetime.now().isoformat(),
        }
        if extra:
            rec.update(json_ready(extra))
        with open(self.metrics_file, 'a') as f:
            f.write(json.dumps(rec) + "\n")
        self._logger.info(
            f"[Trial {trial} Cycle {cycle}] labeled={labeled_size} "
            f"acc={acc:.2f}% svd={svd_time:.2f}s sel={select_time:.2f}s"
        )

    def summary(self):
        import logging
        if not self.metrics_file.exists():
            self._logger.info("No metrics recorded.")
            return
        records = []
        with open(self.metrics_file) as f:
            for line in f:
                records.append(json.loads(line))
        if not records:
            return
        trials = sorted(set(r["trial"] for r in records))
        self._logger.info("=" * 50)
        self._logger.info(f"Experiment: {self.exp_name}")
        for t in trials:
            recs = [r for r in records if r["trial"] == t]
            best = max(recs, key=lambda r: r["test_acc"])
            self._logger.info(
                f"  Trial {t}: best={best['test_acc']:.2f}% "
                f"@ Cycle {best['cycle']} (labeled={best['labeled_size']})"
            )


# ============================================================
#
# ============================================================
def get_args():
    parser = argparse.ArgumentParser(description='Nullspace Evolution Active Learning')
    #
    parser.add_argument('--datapath', default='data/', type=str)
    parser.add_argument('--dataset', default='svhn', type=str,
                        choices=['svhn', 'cifar100', 'caltech101', 'cifar10', 'caltech256'])
    parser.add_argument('--save', default='Result/nullspace_evolution', type=str)
    parser.add_argument('--exp_name', default=None, type=str)
    parser.add_argument('--trial', default=3, type=int)
    parser.add_argument('--budgets', default='base',
                        choices=['small', 'base', 'larget', 'test'])

    #
    parser.add_argument('--use_relative_innovation', dest='use_relative_innovation',
                        action='store_true', default=True,
                        help='Enable relative-innovation acquisition.')
    parser.add_argument('--no_relative_innovation', dest='use_relative_innovation',
                        action='store_false',
                        help='Use the standard nullspace acquisition variant.')
    parser.add_argument('--dim_method', default='spectral_gap', type=str,
                        choices=['spectral_gap', 'energy_ratio', 'parallel_analysis', 'fixed'],
                        help="See README for details.")
    parser.add_argument('--adaptive_dim_threshold', default=0.05, type=float,
                        help="See README for details.")
    parser.add_argument('--relative_eps', default=1e-8, type=float,
                        help='Epsilon used in the relative innovation ratio.')
    parser.add_argument('--ratio_clip_percentile', default=95.0, type=float,
                        help='Percentile used to clip relative innovation ratios; set to 0 to disable.')
    parser.add_argument('--beta', default=0.1, type=float)
    parser.add_argument('--bandwidth', default=1.0, type=float)
    parser.add_argument('--ablation_mode', default='relative', type=str,
                        choices=['relative', 'standard', 'ablation_compare'],
                        help="See README for details.")

    #
    parser.add_argument('--acq_method', default='ours', type=str,
                        choices=['ours', 'badge'],
                        help='Acquisition method: ours or BADGE baseline.')
    parser.add_argument('--badge_proj_dim', default=256, type=int,
                        help='Random projection dimension for BADGE gradient embeddings.')
    parser.add_argument('--badge_chunk_size', default=256, type=int,
                        help='Chunk size used to build projected BADGE embeddings.')
    parser.add_argument('--backbone', default='auto_pdf', type=str,
                        choices=['auto_pdf', 'resnet18', 'wide_resnet18', 'resnet50_imagenet'],
                        help='Backbone. auto_pdf uses the manuscript setting per dataset.')
    parser.add_argument('--wrn_width_mult', default=2, type=int,
                        help='Width multiplier for the SVHN Wide ResNet-18 backbone.')
    parser.add_argument('--image_size', default=None, type=int,
                        help='Override input image size; auto_pdf uses 224 for Caltech-101 ResNet-50.')
    parser.add_argument('--multi_nullspace', action='store_true')
    parser.add_argument('--nullspace_start_cycle', default=3, type=int)
    parser.add_argument('--num_clusters', default=3, type=int)
    parser.add_argument('--cluster_min_size', default=50, type=int)

    #
    parser.add_argument('--w_isd', default=0.6, type=float)
    parser.add_argument('--w_ent', default=0.25, type=float)
    parser.add_argument('--w_pl', default=0.15, type=float)
    parser.add_argument('--use_maximin', action='store_true')
    parser.add_argument('--lambda_div', default=0.7, type=float)

    #
    parser.add_argument('--epochs', default=160, type=int)
    parser.add_argument('--epochl', default=160, type=int)
    parser.add_argument('--ft_epochs', default=30, type=int,
                        help='Number of LossNet fine-tuning epochs after backbone training.')
    parser.add_argument('--max_cycles', default=None, type=int,
                        help='Run only the first N active-learning cycles.')
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--margin', default=1.0, type=float)
    parser.add_argument('--weight', default=1.0, type=float)
    parser.add_argument('--milestones', nargs='+', type=int, default=[30, 60, 90])
    parser.add_argument('--device', default='cuda:0', type=str)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--weight_decay', default=5e-3, type=float)

    return parser.parse_args()


def resolve_backbone_name(args):
    if args.backbone != 'auto_pdf':
        return args.backbone
    if args.dataset == 'svhn':
        return 'wide_resnet18'
    if args.dataset == 'caltech101':
        return 'resnet50_imagenet'
    return 'resnet18'


def resolve_image_size(args, backbone_name):
    if args.image_size is not None:
        return args.image_size
    if args.dataset == 'caltech101' and backbone_name == 'resnet50_imagenet':
        return 224
    return None


def build_models(args, device):
    backbone_name = resolve_backbone_name(args)
    if backbone_name == 'wide_resnet18':
        backbone = WideResNet18(
            num_classes=args.num_classes,
            width_mult=args.wrn_width_mult,
        )
    elif backbone_name == 'resnet50_imagenet':
        backbone = TorchvisionResNet50(
            num_classes=args.num_classes,
            pretrained=True,
        )
    elif backbone_name == 'resnet18':
        backbone = ResNet18(num_classes=args.num_classes)
    else:
        raise ValueError(f"Unsupported backbone: {backbone_name}")

    loss_module = LossNet18(num_channels=getattr(backbone, 'feature_channels', [64, 128, 256, 512]))
    args.resolved_backbone = backbone_name
    return {
        'backbone': backbone.to(device),
        'module': loss_module.to(device),
    }


def submodular_select(nullspace_features, isd_scores, k, lambda_div=0.3):
    "message"
    n_samples = len(nullspace_features)
    selected = []
    remain = set(range(n_samples))

    #
    min_dist_to_selected = torch.full((n_samples,), float('inf'), device=nullspace_features.device)

    for step in range(k):
        best_idx = None
        best_gain = -float('inf')

        #
        for j in remain:
            #
            if selected:
                #
                dist_to_selected = torch.norm(nullspace_features[selected] - nullspace_features[j], dim=1)
                current_min_dist_to_selected = dist_to_selected.min().item()

                #
                #
                #

                #
                coverage_gain = 0
                for u in remain:
                    if u == j:
                        continue
                    #
                    dist_to_selected_original = min_dist_to_selected[u]
                    #
                    dist_to_j = torch.norm(nullspace_features[u] - nullspace_features[j]).item()
                    #
                    new_min_dist = min(dist_to_selected_original, dist_to_j)
                    #
                    coverage_gain += (new_min_dist - dist_to_selected_original)
            else:
                #
                coverage_gain = 0

            #
            innovation_gain = lambda_div * isd_scores[j].item()

            #
            gain = coverage_gain + innovation_gain

            if gain > best_gain:
                best_gain = gain
                best_idx = j

        #
        selected.append(best_idx)
        remain.remove(best_idx)

        #
        if len(selected) > 1:
            for u in remain:
                dist_to_new = torch.norm(nullspace_features[u] - nullspace_features[best_idx]).item()
                min_dist_to_selected[u] = min(min_dist_to_selected[u], dist_to_new)
        else:
            #
            for u in remain:
                min_dist_to_selected[u] = torch.norm(nullspace_features[u] - nullspace_features[best_idx]).item()

    return selected




# ============================================================
#
# ============================================================
def loss_pred_loss(input, target, margin=1.0, reduction='mean'):
    if len(input) % 2 != 0:
        input, target = input[:len(input) // 2 * 2], target[:len(target) // 2 * 2]
    input = (input - input.flip(0))[:len(input) // 2]
    target = (target - target.flip(0))[:len(target) // 2].detach()
    one = 2 * torch.sign(torch.clamp(target, min=0)) - 1
    if reduction == 'mean':
        return torch.sum(torch.clamp(margin - one * input, min=0)) / input.size(0)
    return torch.clamp(margin - one * input, min=0)


def train_epoch(models, criterion, optimizers, dataloaders, epoch, epoch_loss, device, args):
    free(models['module'])
    free(models['backbone'])
    models['module'].train()
    models['backbone'].train()
    for batch_idx, (inputs, labels) in enumerate(dataloaders['train']):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizers['backbone'].zero_grad()
        optimizers['module'].zero_grad()
        scores, features, _ = models['backbone'](inputs)
        target_loss = criterion(scores, labels)
        loss = torch.sum(target_loss) / target_loss.size(0)
        if epoch > epoch_loss:
            for i in range(len(features)):
                features[i] = features[i].detach()
        pred_loss = models['module'](features).view(-1)
        loss = loss + args.weight * loss_pred_loss(pred_loss, target_loss, args.margin)
        loss.backward()
        optimizers['module'].step()
        optimizers['backbone'].step()


def ft_epoch(models, criterion, optimizers, dataloaders, device, args):
    models['backbone'].eval()
    models['module'].train()
    frozen(models['backbone'])
    free(models['module'])
    for batch_idx, (inputs, labels) in enumerate(dataloaders['train']):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizers['ft'].zero_grad()
        scores, features, _ = models['backbone'](inputs)
        target_loss = criterion(scores, labels)
        for i in range(len(features)):
            features[i] = features[i].detach()
        pred_loss = models['module'](features).view(-1)
        loss = loss_pred_loss(pred_loss, target_loss, args.margin)
        loss.backward()
        optimizers['ft'].step()


def test(models, dataloaders, device, cycle, args, logger, best_acc):
    models['backbone'].eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels in dataloaders['test']:
            inputs, labels = inputs.to(device), labels.to(device)
            scores, _, _ = models['backbone'](inputs)
            _, preds = scores.max(1)
            total += labels.size(0)
            correct += preds.eq(labels).sum().item()
    acc = 100. * correct / total
    if acc > best_acc:
        best_acc = acc
        torch.save(models['backbone'].state_dict(),
                   os.path.join(args.save, f"main_{cycle}.pth"))
        logger.info(f'Saved main_{cycle}.pth with acc={acc:.2f}%')
    return best_acc, acc


def get_logits_and_features(model, data_loader, device, batch_size):
    model.eval()
    feats_buf, logits_buf, indices_list = [], [], []
    start = 0
    with torch.no_grad():
        for batch_idx, (inputs, _) in enumerate(data_loader):
            inputs = inputs.to(device)
            scores, _, feature = model(inputs)
            emb = feature[-1] if isinstance(feature, (list, tuple)) else feature
            if emb.dim() > 2:
                emb = torch.flatten(emb, start_dim=1)
            feats_buf.append(emb.cpu())
            logits_buf.append(scores.cpu())
            if hasattr(data_loader.sampler, 'indices'):
                batch_inds = data_loader.sampler.indices[
                    batch_idx * batch_size: (batch_idx + 1) * batch_size]
                indices_list.extend(batch_inds if isinstance(batch_inds[0], int)
                                    else list(batch_inds))
            else:
                indices_list.extend(range(start, start + emb.size(0)))
            start += emb.size(0)
    return torch.cat(feats_buf, 0).to(device),\
        torch.cat(logits_buf, 0).to(device), indices_list


@torch.no_grad()
def predict_loss_for_unlabeled(models, unlabeled_loader, device):
    models['backbone'].eval()
    models['module'].eval()
    pred_list = []
    for images, _ in unlabeled_loader:
        images = images.to(device)
        _, features, _ = models['backbone'](images)
        pred = models['module'](features).view(-1).cpu()
        pred_list.extend(pred.tolist())
    return torch.tensor(pred_list, dtype=torch.float32)


# ============================================================
#
# ============================================================
def compute_density_score(Z_perp, bandwidth=1.0):
    "message"
    n_samples = Z_perp.size(0)
    density_scores = torch.zeros(n_samples, device=Z_perp.device)
    sigma = bandwidth
    for i in range(0, n_samples, 1024):
        batch = Z_perp[i:i + 1024]
        dists = torch.cdist(batch, Z_perp, p=2)
        density_scores[i:i + 1024] = torch.sum(torch.exp(-dists ** 2 / (sigma ** 2 + 1e-12)), dim=1)
    return density_scores


def compute_information_scarcity_entropy(innovation_scores, density_scores):
    "message"
    density_sum = torch.sum(density_scores) + 1e-12
    density_probs = density_scores / density_sum
    density_probs = torch.clamp(density_probs, min=1e-12)
    entropy_terms = -torch.log(density_probs)
    ise_scores = innovation_scores * entropy_terms
    return ise_scores


def compute_distribution_deviation(Z_perp):
    "message"
    mean_pool = torch.mean(Z_perp, dim=0)
    dd = torch.norm(Z_perp - mean_pool.unsqueeze(0), dim=1)
    return dd


# ============================================================
#
# ============================================================

if __name__ == '__main__':
    args = get_args()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    args.resolved_backbone = resolve_backbone_name(args)
    args.resolved_image_size = resolve_image_size(args, args.resolved_backbone)

    #
    exp_name = args.exp_name or (
        f"{args.dataset}_{args.ablation_mode}_{args.dim_method}_{args.budgets}"
        if not args.use_relative_innovation else
        f"{args.dataset}_relative_{args.dim_method}_{args.budgets}"
    )
    tracker = ExperimentTracker(args.save, exp_name)
    tracker.log_config(vars(args))

    print(f"\n{'=' * 60}")
    print(f"  Dataset: {args.dataset}")
    print(f"  Acquisition mode: {'relative innovation' if args.use_relative_innovation else 'standard nullspace'}")
    print(f"  Dimension method: {args.dim_method}")
    print(f"  Experiment directory: {tracker.exp_dir}")
    print(f"{'=' * 60}\n")

    #
    args.num_images = get_num_images(args.dataset)
    args.num_classes = cls_dict[args.dataset]

    if args.budgets == 'test':
        num_samples_list = [50, 100]
    elif args.dataset == 'cifar100':
        num_samples_list = [1000, 2000, 3000, 4000, 5000]
    else:
        num_samples_list = [500, 1000, 1500, 2000, 2500, 3000]
    if args.max_cycles is not None:
        if args.max_cycles < 2:
            raise ValueError("--max_cycles must be at least 2 so one acquisition is measured.")
        num_samples_list = num_samples_list[:args.max_cycles]
    args.cycles = len(num_samples_list)

    for trial in range(args.trial):
        prev_P_null = None
        prev_residual_cache = {}
        random.seed(2022 + trial)
        np.random.seed(2022 + trial)
        torch.manual_seed(2022 + trial)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(2022 + trial)
        cudnn.deterministic = True
        cudnn.benchmark = False

        logger = tracker._logger
        logger.info(f"\n{'#' * 60} Trial {trial + 1}/{args.trial} {'#' * 60}")

        train_ds, test_ds, eval_ds = get_data(
            args.datapath,
            args.dataset,
            image_size=args.resolved_image_size,
        )
        data_idx = list(range(args.num_images))
        random.shuffle(data_idx)

        labeled_set = data_idx[:num_samples_list[0]]
        unlabeled_set = data_idx[num_samples_list[0]:]

        models = build_models(args, device)
        criterion = nn.CrossEntropyLoss(reduction='none').to(device)

        for cycle in range(args.cycles):
            logger.info(f"\n==== Cycle {cycle + 1}/{args.cycles} | Labeled={len(labeled_set)} ====")

            #
            train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                                      sampler=SubsetRandomSampler(labeled_set))
            test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)
            dataloaders = {'train': train_loader, 'test': test_loader}

            optim_b = optim.SGD(models['backbone'].parameters(), lr=args.lr,
                                momentum=args.momentum, weight_decay=args.weight_decay)
            optim_m = optim.SGD(models['module'].parameters(), lr=args.lr,
                                momentum=args.momentum, weight_decay=args.weight_decay)
            optim_ft = optim.SGD(models['module'].parameters(), lr=args.lr,
                                 momentum=args.momentum, weight_decay=args.weight_decay)
            sched_b = lr_scheduler.MultiStepLR(optim_b, milestones=args.milestones)
            sched_m = lr_scheduler.MultiStepLR(optim_m, milestones=args.milestones)

            best_acc = 0
            #
            for ep in range(args.epochs):
                train_epoch(models, criterion,
                            {'backbone': optim_b, 'module': optim_m},
                            dataloaders, ep, args.epochl, device, args)
                sched_b.step()
                sched_m.step()
                if ep % 20 == 0:
                    best_acc, _ = test(models, dataloaders, device, cycle, args, logger, best_acc)

            # fine-tune LossNet
            logger.info("Fine-tuning LossNet...")
            for ft_ep in range(args.ft_epochs):
                ft_epoch(models, criterion, {'ft': optim_ft}, dataloaders, device, args)
            best_acc, final_acc = test(models, dataloaders, device, cycle, args, logger, best_acc)
            current_labeled_size = len(labeled_set)
            current_unlabeled_size = len(unlabeled_set)

            #
            if cycle < args.cycles - 1:
                labeled_loader = DataLoader(eval_ds, batch_size=args.batch_size,
                                            sampler=SubsetSequentialSampler(labeled_set))
                unlabeled_loader = DataLoader(eval_ds, batch_size=args.batch_size,
                                              sampler=SubsetSequentialSampler(unlabeled_set))

                sync_cuda(device)
                acquisition_start = time.time()
                feature_start = time.time()
                L_feat, _, _ = get_logits_and_features(models['backbone'], labeled_loader, device, args.batch_size)
                U_feat, U_logit, U_idx = get_logits_and_features(
                    models['backbone'], unlabeled_loader, device, args.batch_size)
                loss_pred = predict_loss_for_unlabeled(models, unlabeled_loader, device)
                sync_cuda(device)
                feature_time = time.time() - feature_start

                num_select = num_samples_list[cycle + 1] - len(labeled_set)
                prev_residuals = None
                if prev_residual_cache:
                    cached = [prev_residual_cache.get(int(idx)) for idx in U_idx]
                    if all(v is not None for v in cached):
                        prev_residuals = torch.tensor(cached, dtype=torch.float32, device=device)

                #
                if args.acq_method == 'badge':
                    sync_cuda(device)
                    select_start = time.time()
                    selected = badge_select(
                        U_feat,
                        U_logit,
                        num_select,
                        proj_dim=args.badge_proj_dim,
                        chunk_size=args.badge_chunk_size,
                        seed=2022 + trial + cycle,
                    )
                    sync_cuda(device)
                    sync_cuda(device)
                    select_wall_time = time.time() - select_start
                    extra_info = {
                        'acq_method': 'badge',
                        'svd_time_s': 0.0,
                        'density_time_s': 0.0,
                        'select_time_s': select_wall_time,
                        'badge_proj_dim': args.badge_proj_dim,
                    }
                elif args.use_relative_innovation:
                    sync_cuda(device)
                    select_start = time.time()
                    mode = 'relative' if not args.ablation_mode == 'ablation_compare' else 'ablation_compare'
                    selected, info, current_P = select_with_relative_innovation(
                        L_feat, U_feat, U_logit, loss_pred,
                        num_select, args, cycle,
                        prev_P_null=prev_P_null,
                        prev_residuals=prev_residuals,
                        mode = mode
                    )
                    sync_cuda(device)
                    select_wall_time = time.time() - select_start

                    #
                    prev_P_null = current_P
                    prev_P_null = current_P
                    current_residuals = info.pop('current_residuals', None)
                    if current_residuals is not None:
                        prev_residual_cache = {
                            int(idx): float(res)
                            for idx, res in zip(U_idx, current_residuals.tolist())
                        }
                    extra_info = info.copy() if mode == 'ablation_compare' else info
                    logger.info(f"  Nullspace dimension: {info.get('dims_t', 'N/A')}")
                    if mode == 'ablation_compare':
                        logger.info(f"  Relative ISD mean={info['isd_relative'].mean():.4f} "
                                    f"standard ISD mean={info['isd_standard'].mean():.4f}")
                else:
                    #
                    sync_cuda(device)
                    svd_start = time.time()
                    P_null, dims, S = compute_nullspace_projection(L_feat,
                                                                   dim_method='spectral_gap' if args.dim_method != 'fixed' else 'energy_ratio',
                                                                   adaptive_threshold=args.adaptive_dim_threshold)
                    sync_cuda(device)
                    svd_time = time.time() - svd_start

                    logger.info(f"  Nullspace dimension: {dims}")

                    Z_perp = U_feat @ P_null.t()

                    #
                    innovation = torch.norm(Z_perp, dim=1)
                    #density_scores = compute_density_score(Z_perp, args.bandwidth)
                    sync_cuda(device)
                    density_start = time.time()
                    density_scores = riemannian_density(Z_perp, args.bandwidth)
                    sync_cuda(device)
                    density_time = time.time() - density_start
                    ise_scores = compute_information_scarcity_entropy(innovation, density_scores)
                    dd_scores = compute_distribution_deviation(Z_perp)
                    isd = ise_scores * torch.exp(args.beta * dd_scores)

                    entropy_scores = -torch.sum(torch.softmax(U_logit, 1) *
                                                torch.log(torch.clamp(torch.softmax(U_logit, 1), 1e-12, 1)), 1)

                    #
                    isd_n = rank_normalize(isd).to(device)
                    ent_n = rank_normalize(entropy_scores).to(device)
                    pl_n = rank_normalize(loss_pred).to(device)

                    #
                    if cycle >= 4:
                        w_isd_adj = args.w_isd * 0.7
                        w_ent_adj = args.w_ent * 1.3
                        w_pl_adj = args.w_pl * 1.0
                    else:
                        w_isd_adj, w_ent_adj, w_pl_adj = args.w_isd, args.w_ent, args.w_pl

                    w_sum = max(1e-12, w_isd_adj + w_ent_adj + w_pl_adj)
                    final_scores = (w_isd_adj / w_sum) * isd_n + (w_ent_adj / w_sum) * ent_n + (w_pl_adj / w_sum) * pl_n

                    if args.use_maximin:
                        sync_cuda(device)
                        select_start = time.time()
                        #
                        #
                        #
                        lambda_0 = args.lambda_div  #
                        T = args.cycles  #
                        t = cycle  #

                        #
                        adaptive_lambda = lambda_0 * np.exp(-t / T)

                        #
                        # adaptive_lambda = lambda_0 * (1 - t / T)

                        #
                        adaptive_lambda = max(0.05, adaptive_lambda)

                        logger.info(f"  Adaptive lambda_t = {adaptive_lambda:.4f} (cycle {t + 1}/{T})")

                        selected = submodular_select(
                            Z_perp,
                            final_scores,
                            num_select,
                            lambda_div=adaptive_lambda  #
                        )
                    else:
                        sync_cuda(device)
                        select_start = time.time()
                        _, topk = torch.topk(final_scores, num_select)
                        selected = topk.cpu().numpy().tolist()
                        sync_cuda(device)
                    sync_cuda(device)
                    select_wall_time = time.time() - select_start

                    prev_P_null = P_null
                    prev_residual_cache = {
                        int(idx): float(res)
                        for idx, res in zip(U_idx, innovation.detach().cpu().tolist())
                    }
                    info = {'dims_t': dims, 'svd_time_s': svd_time, 'density_time_s': density_time}
                    extra_info = info

                sync_cuda(device)
                acquisition_time = time.time() - acquisition_start
                svd_time = float(extra_info.get('svd_time_s', 0.0))
                select_time = float(extra_info.get('select_time_s', select_wall_time))
                extra_info.update({
                    'feature_time_s': round(feature_time, 4),
                    'select_wall_time_s': round(select_wall_time, 4),
                    'acquisition_time_s': round(acquisition_time, 4),
                    'prev_residual_cache_hits': int(prev_residuals is not None),
                })

                new_samples = [U_idx[i] for i in selected]
                labeled_set += new_samples
                unlabeled_set = list(set(unlabeled_set) - set(new_samples))

                tracker.log_cycle(
                    trial=trial, cycle=cycle,
                    labeled_size=current_labeled_size, unlabeled_size=current_unlabeled_size,
                    acc=final_acc, svd_time=svd_time, select_time=select_time,
                    extra={**extra_info, 'num_select': num_select}
                )
            else:
                tracker.log_cycle(
                    trial=trial, cycle=cycle,
                    labeled_size=current_labeled_size, unlabeled_size=current_unlabeled_size,
                    acc=final_acc, svd_time=0.0, select_time=0.0,
                    extra={'num_select': 0, 'acquisition_time_s': 0.0}
                )

    tracker.summary()
    print(f"\nExperiment completed. Results saved to: {tracker.exp_dir}")
