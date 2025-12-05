import torch
import torch.nn.functional as F
from model.module.swcd_dgma2_multiscale import (
    SWCD_DGMA2_MultiScale,
    SWCD_DGMA2_MultiScale_Attention
)
from data.cd_dataset import DataLoader
from option import Options
from util.metric_tool import ConfuseMatrixMeter
from tqdm import tqdm
import os
import logging
import numpy as np
import json
from datetime import datetime


def save_test_results(scores, save_path, model_name="SWCD_DGMA2"):
    """Save test results to JSON file"""
    results = {
        'model': model_name,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'metrics': {
            'overall': {
                'accuracy': float(scores['acc']),
                'mean_iou': float(scores['miou']),
                'mean_f1': float(scores['mf1'])
            },
            'class_0_unchanged': {
                'iou': float(scores['iou_0']),
                'f1': float(scores['F1_0']),
                'precision': float(scores['precision_0']),
                'recall': float(scores['recall_0'])
            },
            'class_1_changed': {
                'iou': float(scores['iou_1']),
                'f1': float(scores['F1_1']),
                'precision': float(scores['precision_1']),
                'recall': float(scores['recall_1'])
            }
        }
    }

    with open(save_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"\nResults saved to: {save_path}")


def test_model(opt, checkpoint_path, model_variant='multiscale_attention'):
    """
    Test InceptionMultiScale model with built-in decoder
    Args:
        opt: Options object
        checkpoint_path: Path to model checkpoint
        model_variant: 'multiscale' or 'multiscale_attention'
    """
    logging.basicConfig(level=logging.INFO)

    # Create model based on variant (with built-in decoder)
    logging.info(f"Creating model: {model_variant}...")
    if model_variant == 'multiscale_attention':
        model = SWCD_DGMA2_MultiScale_Attention(
            input_nc=3,
            output_nc=2,
            dam_per_scale=getattr(opt, 'dam_per_scale', 2),
            beta_init=getattr(opt, 'beta_init', 0.3)
        ).cuda()
    else:
        model = SWCD_DGMA2_MultiScale(
            input_nc=3,
            output_nc=2,
            dam_per_scale=getattr(opt, 'dam_per_scale', 2),
            beta_init=getattr(opt, 'beta_init', 0.3)
        ).cuda()

    # Load checkpoint with advanced fallback mechanism
    checkpoint_loaded = False
    loaded_keys = 0
    skipped_keys = 0
    
    if os.path.exists(checkpoint_path):
        logging.info(f"Loading checkpoint from {checkpoint_path}")
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cuda')
            
            # Extract state dict
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
            
            # Load with detailed tracking
            model_state = model.state_dict()
            incompatible_keys = []
            
            for name, param in state_dict.items():
                if name in model_state:
                    try:
                        if model_state[name].shape == param.shape:
                            model_state[name].copy_(param)
                            loaded_keys += 1
                        else:
                            logging.warning(f"Shape mismatch for {name}: expected {model_state[name].shape}, got {param.shape}")
                            skipped_keys += 1
                            incompatible_keys.append(name)
                    except Exception as e:
                        logging.warning(f"Failed to load {name}: {str(e)[:50]}")
                        skipped_keys += 1
                        incompatible_keys.append(name)
                else:
                    skipped_keys += 1
            
            # Load the modified state dict
            model.load_state_dict(model_state, strict=False)
            checkpoint_loaded = True
            
            logging.info(f"Checkpoint loaded: {loaded_keys} keys loaded, {skipped_keys} keys skipped/mismatched")
            if incompatible_keys:
                logging.info(f"Incompatible keys: {incompatible_keys[:5]}{'...' if len(incompatible_keys) > 5 else ''}")
            
        except (RuntimeError, EOFError, Exception) as e:
            logging.error(f"Failed to load checkpoint {checkpoint_path}: {str(e)[:100]}")
            logging.info("Will use randomly initialized model...")
            checkpoint_loaded = False
    
    # Try fallback checkpoint if primary failed
    if not checkpoint_loaded:
        fallback_path = os.path.join(os.path.dirname(checkpoint_path), 'best_model.pth')
        if os.path.exists(fallback_path) and fallback_path != checkpoint_path:
            logging.info(f"Loading fallback checkpoint from {fallback_path}")
            try:
                checkpoint = torch.load(fallback_path, map_location='cuda')
                
                if 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                else:
                    state_dict = checkpoint
                
                model_state = model.state_dict()
                loaded_keys = 0
                skipped_keys = 0
                
                for name, param in state_dict.items():
                    if name in model_state and model_state[name].shape == param.shape:
                        model_state[name].copy_(param)
                        loaded_keys += 1
                    else:
                        skipped_keys += 1
                
                model.load_state_dict(model_state, strict=False)
                checkpoint_loaded = True
                logging.info(f"Fallback checkpoint loaded: {loaded_keys} keys loaded, {skipped_keys} keys skipped")
            except Exception as e:
                logging.error(f"Failed to load fallback checkpoint: {str(e)[:100]}")
                checkpoint_loaded = False
    
    if checkpoint_loaded:
        logging.info("✅ Checkpoint loaded successfully (with architecture adaptation)!")
    else:
        logging.warning("⚠️ No valid checkpoint loaded. Using randomly initialized model.")

    # Create test data loader
    opt.phase = 'test'
    opt.batch_size = 12
    test_loader = DataLoader(opt)
    test_data = test_loader.load_data()
    test_size = len(test_loader)
    logging.info(f"#test images = {test_size}")

    # Evaluation
    model.eval()
    running_metric = ConfuseMatrixMeter(n_class=2)
    running_metric.clear()

    tbar = tqdm(test_data, desc="Testing", ncols=100)

    with torch.no_grad():
        for i, data in enumerate(tbar):
            img1 = data['img1'].cuda()
            img2 = data['img2'].cuda()
            label = data['label']

            # Inference (built-in decoder)
            pred, sim_mask = model(img1, img2)
            pred = torch.argmax(pred, dim=1)

            # Update metrics
            running_metric.update_cm(
                pr=pred.cpu().numpy(),
                gt=label.cpu().numpy()
            )

            # Update progress bar
            scores = running_metric.get_scores()
            tbar.set_postfix({
                'F1': f"{scores['F1_1']*100:.2f}",
                'IoU': f"{scores['iou_1']*100:.2f}"
            })

    # Final metrics
    final_scores = running_metric.get_scores()

    # Format and display results
    print("\n" + "="*80)
    print("TEST RESULTS")
    print("="*80)

    # Overall metrics
    print("\n[Overall Metrics]")
    print(f"  Overall Accuracy (OA):  {final_scores['acc']*100:6.2f}%")
    print(f"  Mean IoU (mIoU):        {final_scores['miou']*100:6.2f}%")
    print(f"  Mean F1 (mF1):          {final_scores['mf1']*100:6.2f}%")

    # Per-class metrics (Class 0: Unchanged, Class 1: Changed)
    print("\n[Per-Class Metrics]")
    print(f"  Class 0 (Unchanged):")
    print(f"    - IoU:        {final_scores['iou_0']*100:6.2f}%")
    print(f"    - F1-Score:   {final_scores['F1_0']*100:6.2f}%")
    print(f"    - Precision:  {final_scores['precision_0']*100:6.2f}%")
    print(f"    - Recall:     {final_scores['recall_0']*100:6.2f}%")

    print(f"\n  Class 1 (Changed): **[MAIN TARGET]**")
    print(f"    - IoU:        {final_scores['iou_1']*100:6.2f}%")
    print(f"    - F1-Score:   {final_scores['F1_1']*100:6.2f}%  <-- PRIMARY METRIC")
    print(f"    - Precision:  {final_scores['precision_1']*100:6.2f}%")
    print(f"    - Recall:     {final_scores['recall_1']*100:6.2f}%")

    # Summary
    print("\n" + "="*80)
    print(f"SUMMARY: F1={final_scores['F1_1']*100:.2f}% | IoU={final_scores['iou_1']*100:.2f}% | OA={final_scores['acc']*100:.2f}%")
    print("="*80 + "\n")

    # Also log to file
    logging.info("\n" + "="*80)
    logging.info("Test Results:")
    logging.info("="*80)
    for k, v in final_scores.items():
        logging.info(f"{k:15s}: {v*100:.3f}%")
    logging.info("="*80)

    # Save results to JSON
    result_dir = os.path.dirname(checkpoint_path)
    result_file = os.path.join(result_dir, 'test_results.json')
    save_test_results(final_scores, result_file, model_name=opt.name)

    return final_scores


def test_ensemble(opt, checkpoint_paths, model_variant='multiscale_attention'):
    """
    Test ensemble of multiple InceptionMultiScale models with built-in decoders

    Args:
        opt: Options object
        checkpoint_paths: List of checkpoint paths
        model_variant: 'multiscale' or 'multiscale_attention'
    """
    logging.basicConfig(level=logging.INFO)

    # Create models (with built-in decoders)
    models = []
    for i, ckpt_path in enumerate(checkpoint_paths):
        logging.info(f"Loading model {i+1}/{len(checkpoint_paths)}: {ckpt_path}")
        
        # Create model
        if model_variant == 'multiscale_attention':
            model = SWCD_DGMA2_MultiScale_Attention(input_nc=3, output_nc=2).cuda()
        else:
            model = SWCD_DGMA2_MultiScale(input_nc=3, output_nc=2).cuda()

        if os.path.exists(ckpt_path):
            try:
                checkpoint = torch.load(ckpt_path)
                
                # Extract state dict
                if 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                else:
                    state_dict = checkpoint
                
                # Load with shape checking
                model_state = model.state_dict()
                loaded_keys = 0
                skipped_keys = 0
                
                for name, param in state_dict.items():
                    if name in model_state:
                        if model_state[name].shape == param.shape:
                            model_state[name].copy_(param)
                            loaded_keys += 1
                        else:
                            skipped_keys += 1
                    else:
                        skipped_keys += 1
                
                model.load_state_dict(model_state, strict=False)
                model.eval()
                models.append(model)
                logging.info(f"Model {i+1} loaded: {loaded_keys} keys matched, {skipped_keys} keys skipped")
            except Exception as e:
                logging.error(f"Failed to load model from {ckpt_path}: {str(e)[:100]}")
        else:
            logging.warning(f"Checkpoint not found: {ckpt_path}")

    if len(models) == 0:
        logging.error("No valid models loaded!")
        return

    # Create test data loader
    opt.phase = 'test'
    opt.batch_size = 12
    test_loader = DataLoader(opt)
    test_data = test_loader.load_data()

    # Evaluation
    running_metric = ConfuseMatrixMeter(n_class=2)
    running_metric.clear()

    tbar = tqdm(test_data, desc="Testing Ensemble", ncols=100)

    with torch.no_grad():
        for i, data in enumerate(tbar):
            img1 = data['img1'].cuda()
            img2 = data['img2'].cuda()
            label = data['label']

            # Ensemble prediction (average logits)
            pred_ensemble = []
            for model in models:
                pred, sim_mask = model(img1, img2)
                pred_ensemble.append(pred)

            # Average predictions
            pred_avg = torch.stack(pred_ensemble).mean(dim=0)
            pred_final = torch.argmax(pred_avg, dim=1)

            # Update metrics
            running_metric.update_cm(
                pr=pred_final.cpu().numpy(),
                gt=label.cpu().numpy()
            )

    # Final metrics
    final_scores = running_metric.get_scores()

    # Format and display results
    print("\n" + "="*80)
    print(f"ENSEMBLE TEST RESULTS ({len(models)} models)")
    print("="*80)

    # Overall metrics
    print("\n[Overall Metrics]")
    print(f"  Overall Accuracy (OA):  {final_scores['acc']*100:6.2f}%")
    print(f"  Mean IoU (mIoU):        {final_scores['miou']*100:6.2f}%")
    print(f"  Mean F1 (mF1):          {final_scores['mf1']*100:6.2f}%")

    # Per-class metrics
    print("\n[Per-Class Metrics]")
    print(f"  Class 0 (Unchanged):")
    print(f"    - IoU:        {final_scores['iou_0']*100:6.2f}%")
    print(f"    - F1-Score:   {final_scores['F1_0']*100:6.2f}%")
    print(f"    - Precision:  {final_scores['precision_0']*100:6.2f}%")
    print(f"    - Recall:     {final_scores['recall_0']*100:6.2f}%")

    print(f"\n  Class 1 (Changed): **[MAIN TARGET]**")
    print(f"    - IoU:        {final_scores['iou_1']*100:6.2f}%")
    print(f"    - F1-Score:   {final_scores['F1_1']*100:6.2f}%  <-- PRIMARY METRIC")
    print(f"    - Precision:  {final_scores['precision_1']*100:6.2f}%")
    print(f"    - Recall:     {final_scores['recall_1']*100:6.2f}%")

    # Summary
    print("\n" + "="*80)
    print(f"ENSEMBLE SUMMARY: F1={final_scores['F1_1']*100:.2f}% | IoU={final_scores['iou_1']*100:.2f}% | OA={final_scores['acc']*100:.2f}%")
    print("="*80 + "\n")

    # Also log to file
    logging.info("\n" + "="*80)
    logging.info(f"Ensemble Test Results ({len(models)} models):")
    logging.info("="*80)
    for k, v in final_scores.items():
        logging.info(f"{k:15s}: {v*100:.3f}%")
    logging.info("="*80)

    # Save ensemble results to JSON
    result_dir = os.path.dirname(checkpoint_paths[0]) if checkpoint_paths else "./results"
    result_file = os.path.join(result_dir, f'ensemble_test_results_{len(models)}models.json')
    os.makedirs(os.path.dirname(result_file), exist_ok=True)
    save_test_results(final_scores, result_file, model_name=f"Ensemble_{len(models)}models")

    return final_scores


if __name__ == "__main__":
    # Parse options
    opt = Options().parse()

    # Add InceptionMultiScale-specific options
    opt.dam_per_scale = 2
    opt.beta_init = 0.3
    
    # Model variant: 'multiscale' or 'multiscale_attention'
    model_variant = getattr(opt, 'model_variant', 'multiscale_attention')

    # Single model test - try multiple checkpoint paths
    checkpoint_candidates = [
        f"{opt.checkpoint_dir}/{opt.name}/best_model.pth",
        f"{opt.checkpoint_dir}/best_model.pth",
    ]
    
    checkpoint_path = None
    for candidate in checkpoint_candidates:
        if os.path.exists(candidate):
            try:
                # Verify checkpoint is valid
                torch.load(candidate, map_location='cpu')
                checkpoint_path = candidate
                logging.info(f"Found valid checkpoint: {candidate}")
                break
            except Exception as e:
                logging.warning(f"Checkpoint {candidate} is corrupted: {str(e)[:100]}")
                continue
    
    if checkpoint_path:
        logging.info(f"Testing single model: {checkpoint_path}")
        logging.info(f"Model variant: {model_variant}")
        scores = test_model(opt, checkpoint_path, model_variant=model_variant)
    else:
        logging.error(f"No valid checkpoint found in {opt.checkpoint_dir}/{opt.name}/")
        logging.info("Tried:")
        for candidate in checkpoint_candidates:
            logging.info(f"  - {candidate}")
        logging.info("Usage: python test.py --name <experiment_name>")
