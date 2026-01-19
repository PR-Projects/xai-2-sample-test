#!/usr/bin/env python


from pathlib import Path
import sys
import argparse
from argparse import Namespace

from src.eval_utils import (
    plot_faithfulness_results,
    compute_auroc_faithfulness,
    print_auroc_table,
    plot_max_sensitivity_results,
    print_max_sensitivity_table
)

# Add src directory to Python path
project_root = Path(__file__).parent.absolute()
src_dir = project_root / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from src.vis2samtestdr import TestStatisticBackprop

# Base arguments
base_args = {
    'annot_path': '/path/to/adni_data/adni_T1_3T_linear_annotation.csv',
    'sav_gr_np': False,
    'sav_embed_np': False,
    'corrupted': False,
    'deg': 'None',
    'ckp': 'fnt',
    'img_path': '/path/to/adni_data',
    'n': 100, 'm': 100, 'bs': 10,
    'idx': 8,
    'random_state': 42,
    'target_layer': '0.7.2.conv3',
    # Faithfulness evaluation parameters
    'n_superpixels': 10,
    'superpixel_compactness': 10.0,
    'circle_radius': 20,
    'circle_center_offset': -20,
    'circle_grey_value': 128,
    # Robustness evaluation parameters
    'n_perturbations': 100,
    'perturbation_min': -1e-4,
    'perturbation_max': 1e-4,
    # LRP-specific parameters (optional, with defaults)
    'lrp_composite': 'epsilon_plus_flat',
    'lrp_epsilon': 1e-6,
    'lrp_gamma': 0.25,
    'lrp_input_low': 0.0,
    'lrp_input_high': 1.0
}

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Run evaluation for explainability methods')
    parser.add_argument(
        '--methods',
        nargs='+',
        default=["cam", "cam++", "lcam", "lrp"],
        choices=["lrp"],
        help='Explainability methods to evaluate (default: cam cam++ lcam lrp)'
    )
    parser.add_argument(
        '--eval_type',
        type=str,
        default='sensitivity',
        choices=['faithfulness', 'sensitivity'],
        help='Type of evaluation: faithfulness or sensitivity (default: faithfulness)'
    )
    cli_args = parser.parse_args()

    # Methods to evaluate
    METHODS = cli_args.methods
    EVAL_TYPE = cli_args.eval_type

    print(f"Evaluation type: {EVAL_TYPE.upper()}")
    print(f"Evaluating methods: {', '.join([m.upper() for m in METHODS])}")

    if EVAL_TYPE == 'faithfulness':
        # Dictionary to store results
        all_results = {}

        base_args["model_path"] = 'adni_results/ckps/model_finetun_best_16_True.pt'
        base_args["dst"] = 'faithfulness_eval'

        # Run for each method
        for method in METHODS:
            print(f"\n{'='*60}")
            print(f"Processing: {method.upper()}")
            print(f"{'='*60}")

            # Create args for this method
            args = Namespace(**base_args)
            args.expl = method
            args.exp = f'{method}-faithfulness-test-eval'

            # Run experiment
            experiment = TestStatisticBackprop(args)
            group0_attr, group1_attr = experiment.run()
            results = experiment.faithfulness_eval()

            # Store results
            all_results[method] = results

        all_results["random"] = experiment.faithfulness_eval(random_attr=True)

        # Plot all results together
        plot_faithfulness_results(all_results)

        # Compute and print AUROC
        auroc_dict = compute_auroc_faithfulness(all_results)
        print_auroc_table(auroc_dict)

    elif EVAL_TYPE == 'sensitivity':
        sensitivity_results = {}

        base_args["model_path"] = 'adni_results/ckps/model_finetun_best_11_False.pt'
        base_args["dst"] = 'test'

        # Run for each method
        for method in METHODS:
            print(f"\n{'='*60}")
            print(f"Max-Sensitivity: {method.upper()}")
            print(f"{'='*60}")

            # Create args for this method
            args = Namespace(**base_args)
            args.expl = method
            args.exp = f'{method}-sensitivity-test-eval'
            args.dst = 'test'

            # Run experiment
            experiment = TestStatisticBackprop(args)
            group0_attr, group1_attr = experiment.run()
            results = experiment.max_sensitivity_evaluation(args.n_perturbations,
                                                            args.perturbation_min,
                                                            args.perturbation_max)

            # Store results
            sensitivity_results[method] = results

        # Plot and print max-sensitivity results
        plot_max_sensitivity_results(sensitivity_results)
        print_max_sensitivity_table(sensitivity_results)
    