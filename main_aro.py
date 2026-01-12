import argparse
import os
import pandas as pd
from model_zoo import get_model
from dataset_zoo import get_dataset
from misc import seed_all, _default_collate, save_scores
import numpy as np
import random
from torch.utils.data import DataLoader
import torch


def config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--batch-size", default=1, type=int)
    parser.add_argument("--num_workers", default=16, type=int)
    parser.add_argument("--model-name", default="llava1.5", type=str,
                        choices=["llava1.5", "llava1.6"])
    parser.add_argument("--dataset", default="Controlled_Images_A", type=str,
                        choices=["Controlled_Images_A", "Controlled_Images_B",
                                "COCO_QA_one_obj", "COCO_QA_two_obj",
                                "VG_QA_one_obj", "VG_QA_two_obj", "VSR"])
    parser.add_argument("--seed", default=1, type=int)
    
    # Method selection
    parser.add_argument("--method", type=str, default="base",
                        choices=[
                            # Baseline
                            "base",
                            # Visual Scaling
                            "scaling_vis", "adapt_vis", "adapt_vis_jsd",
                            "adapt_vis_entropy", "adapt_vis_obj",
                            # Reasoning
                            "reasoning_absolute_4directions",
                            "reasoning_relative_location",
                            "reasoning_relative_relationship",
                            "chain_of_thought",
                            # Research
                            "adapt_vis_for_oracle_research"
                        ],
                        help="Experimental method to use")
    
    # Legacy arguments (kept for compatibility)
    parser.add_argument("--dola-decoding", action="store_true")
    parser.add_argument("--info-layer", type=int)
    
    # Dataset download
    parser.add_argument("--download", action="store_true",
                        help="Whether to download the dataset if it doesn't exist.")
    parser.add_argument("--save-scores", action="store_true",
                        help="Whether to save the scores for the retrieval to analyze later.")
    parser.add_argument("--output-dir", default="./output", type=str)
    
    # Visual scaling parameters
    parser.add_argument("--weight", default=1.0, type=float,
                        help="Visual feature weight for scaling_vis method")
    parser.add_argument("--weight1", default=1.0, type=float,
                        help="Weight when confident (for adaptive methods)")
    parser.add_argument("--weight2", default=1.0, type=float,
                        help="Weight when uncertain (for adaptive methods)")
    parser.add_argument("--threshold", default=0.4, type=float,
                        help="Uncertainty threshold for adaptive methods")
    
    # Options
    parser.add_argument("--option", default='four', type=str,
                        choices=['two', 'four', 'six'],
                        help="Number of answer options")

    return parser.parse_args()


def validate_args(args):
    """Validate arguments based on the selected method"""
    
    # Methods that require Scal model
    methods_need_scal = [
        'scaling_vis', 'adapt_vis', 'adapt_vis_jsd',
        'adapt_vis_entropy', 'adapt_vis_obj', 'adapt_vis_for_oracle_research'
    ]
    
    # Methods that require weight parameter
    methods_need_weight = ['scaling_vis']
    
    # Methods that require weight1, weight2, threshold
    methods_need_adaptive_params = [
        'adapt_vis', 'adapt_vis_jsd', 'adapt_vis_entropy',
        'adapt_vis_obj', 'adapt_vis_for_oracle_research'
    ]
    
    if args.method in methods_need_weight:
        if args.weight == 1.0:
            print(f"⚠️  Warning: Using default weight=1.0 for {args.method}. "
                  f"Consider setting --weight to a different value (e.g., 0.5, 0.8, 1.2, 1.5)")
    
    if args.method in methods_need_adaptive_params:
        if args.weight1 == 1.0 and args.weight2 == 1.0:
            print(f"⚠️  Warning: Using default weight1=1.0 and weight2=1.0 for {args.method}. "
                  f"Consider setting different values (e.g., --weight1 0.5 --weight2 1.5)")
        if args.method == 'adapt_vis_jsd' and args.threshold == 0.4:
            print(f"⚠️  Warning: Default threshold=0.4 might be too high for JSD method. "
                  f"Consider using a lower value (e.g., --threshold 0.04)")
    
    print(f"\n{'='*60}")
    print(f"Experiment Configuration:")
    print(f"{'='*60}")
    print(f"Dataset:    {args.dataset}")
    print(f"Model:      {args.model_name}")
    print(f"Method:     {args.method}")
    print(f"Option:     {args.option}")
    if args.method in methods_need_weight:
        print(f"Weight:     {args.weight}")
    if args.method in methods_need_adaptive_params:
        print(f"Weight1:    {args.weight1} (when confident)")
        print(f"Weight2:    {args.weight2} (when uncertain)")
        print(f"Threshold:  {args.threshold}")
    print(f"Test Mode:  {os.getenv('TEST_MODE', 'False')}")
    print(f"{'='*60}\n")


def main(args):
    # Validate arguments
    validate_args(args)
    
    # Set random seed
    seed_all(args.seed)
    
    # Get model and dataset
    model, image_preprocess = get_model(
        args.model_name, 
        args.device, 
        args.method
    )
    dataset = get_dataset(
        args.dataset, 
        image_preprocess=image_preprocess, 
        download=args.download
    )
    
    # Sampling configuration
    SAMPLE = True
    TEST = os.getenv('TEST_MODE', 'False') == 'True'
    sampled_indices = None
    collate_fn = _default_collate if image_preprocess is None else None

    # Split val and test set
    if SAMPLE:
        total_data_count = len(dataset)
        idx_file_path = f'./output/sampled_idx_{args.dataset}.npy'
        
        if os.path.exists(idx_file_path):
            sampled_indices = np.load(idx_file_path).tolist()
            print(f"Loaded existing sample indices from {idx_file_path}")
        else:
            sampled_indices = random.sample(
                range(total_data_count), 
                int(0.2 * total_data_count)
            )
            sampled_indices.sort()
            np.save(idx_file_path, np.array(sampled_indices))
            print(f"Created new sample indices and saved to {idx_file_path}")
        
        all_indices = set(range(total_data_count))
        
        # Use test set (unsampled 80%)
        if TEST:
            unsampled_indices = list(all_indices - set(sampled_indices))
            unsampled_indices.sort()
            sampled_indices = unsampled_indices
            print(f"Using TEST set: {len(sampled_indices)} samples (80%)")
        else:
            print(f"Using VAL set: {len(sampled_indices)} samples (20%)")
        
        sub_dataset = torch.utils.data.Subset(dataset, sampled_indices)
        joint_loader = DataLoader(
            sub_dataset, 
            batch_size=args.batch_size, 
            shuffle=False, 
            num_workers=args.num_workers, 
            collate_fn=collate_fn
        )
    else:
        # Use full set
        print(f"Using FULL dataset: {len(dataset)} samples")
        joint_loader = DataLoader(
            dataset, 
            batch_size=args.batch_size, 
            shuffle=False, 
            num_workers=args.num_workers, 
            collate_fn=collate_fn
        )

    print(f"\nRunning evaluation on {args.dataset} with {args.model_name}...")
    
    # ========================================================================
    # Evaluation based on dataset type
    # ========================================================================
    
    if args.dataset == 'VSR':
        # VSR dataset evaluation
        labels = dataset.get_labels()
        scores = model.get_judge_scores_vsr_batched(
            args.dataset,
            joint_loader,
            args.method,
            args.weight,
            args.threshold,
            args.weight1,
            args.weight2
        )
        result_records = dataset.evaluate_scores(
            args.model_name,
            scores,
            labels,
            args.output_dir,
            args.dataset
        )
        print(f"\nVSR Results: {result_records}")

    elif args.dataset in ['Controlled_Images_B', 'Controlled_Images_A']:
        # Controlled Images evaluation
        scores, correct_id = model.get_out_scores_wh_batched(
            args.dataset,
            joint_loader,
            args.method,
            args.weight,
            args.option,
            args.threshold,
            args.weight1,
            args.weight2
        )
        print(f"Got scores with shape: {scores.shape}")
        
        # Transpose from (N, K, L) to (N, L, K)
        scores = scores.transpose(0, 2, 1)
        
        dataset.evaluate_scores(
            scores,
            args.output_dir,
            args.dataset,
            args.model_name,
            args.method,
            args.weight,
            sampled_indices,
            args.option
        )

    else:
        # Other datasets (COCO_QA, VG_QA)
        scores, correct_id = model.get_out_scores_wh_batched(
            args.dataset,
            joint_loader,
            args.method,
            args.weight,
            args.option,
            args.threshold,
            args.weight1,
            args.weight2
        )
        
        dataset.save_scores(
            scores,
            correct_id,
            args.output_dir,
            args.dataset,
            args.method,
            args.weight,
            args.model_name,
            args.option
        )
    
    print(f"\n{'='*60}")
    print(f"Evaluation completed!")
    print(f"Results saved to: {args.output_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    args = config()
    main(args)
