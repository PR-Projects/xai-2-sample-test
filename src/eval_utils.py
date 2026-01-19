import numpy as np
from fast_slic import Slic
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.metrics import auc
import json
import os


def segment_image_with_circle_superpixel(image, center, radius, n_segments=10, compactness=10, visualize=False):
    """
    Segment an image into superpixels using SLIC, where one superpixel is constrained
    to be a circular region defined by center and radius.

    Parameters:
    -----------
    image : np.ndarray
        2D numpy array representing a grayscale image (H, W) or 3-channel image (H, W, 3)
    center : tuple of int
        (y, x) coordinates of the circle center
    radius : int or float
        Radius of the circle in pixels
    n_segments : int, optional
        Approximate number of superpixels to generate (default: 100)
    compactness : float, optional
        Balances color proximity and space proximity. Higher values give more weight to
        space proximity, making superpixel shapes more square/compact (default: 10)
    visualize : bool, optional
        If True, displays the superpixel segmentation with borders (default: False)

    Returns:
    --------
    tuple of (superpixel_labels, circle_label)
        superpixel_labels : np.ndarray
            2D array of same shape as image with integer labels for each superpixel
        circle_label : int
            The label ID of the superpixel corresponding to the circle region
    """

    center_y, center_x = center
    height, width = image.shape[:2]

    # Create coordinate grids
    y, x = np.ogrid[:height, :width]

    # Calculate distance from center for each pixel
    distances = np.sqrt((x - center_x)**2 + (y - center_y)**2)

    # Create mask for pixels within the circle
    circle_mask = distances <= radius

    # Initialize SLIC
    slic = Slic(num_components=n_segments, compactness=compactness)

    # Ensure image is 3-channel for SLIC
    if len(image.shape) == 2:
        # Convert grayscale to 3-channel
        image_3ch = np.stack([image, image, image], axis=-1)
    else:
        image_3ch = image.copy()

    # Perform superpixel segmentation
    superpixel_labels = slic.iterate(image_3ch)

    # Assign the circle region a new unique label
    # Use max + 1 to ensure it's a new label that doesn't conflict
    circle_label = np.max(superpixel_labels) + 1
    superpixel_labels[circle_mask] = circle_label

    # Visualization if requested
    if visualize:
        fig, axes = plt.subplots(1, 2, figsize=(12, 7))

        # Original image
        if len(image.shape) == 2:
            axes[0].imshow(image, cmap='gray')
        else:
            axes[0].imshow(image)
        axes[0].set_title('Original Image')
        axes[0].axis('off')

        # Superpixel segmentation (colored by label)
        axes[1].imshow(superpixel_labels)
        axes[1].legend(loc='upper right')
        axes[1].set_title(f'Superpixel Segmentation ({len(np.unique(superpixel_labels))} segments)')
        axes[1].axis('off')

        plt.tight_layout()
        plt.show()

    return superpixel_labels, circle_label


def insert_grey_circle(image, center, radius, grey_value=128):
    """
    Insert a grey circle in a numpy array representing a grayscale image.

    Parameters:
    -----------
    image : np.ndarray
        2D numpy array representing a grayscale image
    center : tuple of int
        (y, x) coordinates of the circle center
    radius : int or float
        Radius of the circle in pixels
    grey_value : int or float, optional
        Grey value for the circle (default: 128 for mid-grey)
        For uint8 images: 0 (black) to 255 (white)
        For float images: typically 0.0 to 1.0

    Returns:
    --------
    np.ndarray
        Image with the grey circle inserted (modifies in-place and returns)
    """
    center_y, center_x = center
    height, width = image.shape[:2]

    # Create coordinate grids
    y, x = np.ogrid[:height, :width]

    # Calculate distance from center for each pixel
    distances = np.sqrt((x - center_x)**2 + (y - center_y)**2)

    # Create mask for pixels within the circle
    circle_mask = distances <= radius

    # Insert grey circle
    image[circle_mask] = grey_value

    return image

def plot_faithfulness_results(results):
    """Save faithfulness evaluation results and create visualizations."""

    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot test statistic vs number of replacements
    axes[0].plot(results['n_replaced'], results['test_statistics'], 'b-o', linewidth=2, markersize=6)
    axes[0].set_xlabel('Number of Superpixels Replaced', fontsize=12)
    axes[0].set_ylabel('Test Statistic (MMD)', fontsize=12)
    axes[0].set_title('Test Statistic vs Superpixel Replacements', fontsize=14)
    axes[0].grid(True, alpha=0.3)

    # Plot p-value vs number of replacements
    axes[1].plot(results['n_replaced'], results['p_values'], 'r-o', linewidth=2, markersize=6)
    axes[1].axhline(y=0.05, color='k', linestyle='--', label='α = 0.05')
    axes[1].set_xlabel('Number of Superpixels Replaced', fontsize=12)
    axes[1].set_ylabel('P-value', fontsize=12)
    axes[1].set_title('P-value vs Superpixel Replacements', fontsize=14)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    return results


def plot_faithfulness_results(results_dict):
    """
    Plot faithfulness results for multiple explanation methods.

    Parameters:
    -----------
    results_dict : dict
        Dictionary mapping method names to their results
        Format: {'cam': results, 'lrp': results, ...}
        Each results should have 'n_replaced', 'test_statistics', 'p_values'
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    colors = {
        'cam': 'blue',
        'cam++': 'green',
        'lcam': 'orange',
        'lrp': 'red',
        'ig': 'purple'
    }

    # Plot 1: Test Statistics
    for method, results in results_dict.items():
        color = colors.get(method, 'gray')
        axes[0].plot(
            results['n_replaced'],
            results['test_statistics'],
            '-o',
            label=method.upper(),
            color=color,
            linewidth=2,
            markersize=5
        )

    axes[0].set_xlabel('Number of Superpixels Replaced', fontsize=13)
    axes[0].set_ylabel('Test Statistic (MMD)', fontsize=13)
    axes[0].set_title('Test Statistic vs Superpixel Replacements', fontsize=15, fontweight='bold')
    axes[0].legend(fontsize=11, loc='best')
    axes[0].grid(True, alpha=0.3)

    # Plot 2: P-values
    for method, results in results_dict.items():
        color = colors.get(method, 'gray')
        axes[1].plot(
            results['n_replaced'],
            results['p_values'],
            '-o',
            label=method.upper(),
            color=color,
            linewidth=2,
            markersize=5
        )

    axes[1].axhline(y=0.05, color='black', linestyle='--', linewidth=1.5, label='α = 0.05')
    axes[1].set_xlabel('Number of Superpixels Replaced', fontsize=13)
    axes[1].set_ylabel('P-value', fontsize=13)
    axes[1].set_title('P-value vs Superpixel Replacements', fontsize=15, fontweight='bold')
    axes[1].legend(fontsize=11, loc='best')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('faithfulness_comparison_all_methods.png', dpi=150, bbox_inches='tight')
    print("\nComparison plot saved to: faithfulness_comparison_all_methods.png")
    plt.show()


def compute_auroc_faithfulness(results_dict):
    """
    Compute AUROC for faithfulness evaluation of each method.

    The AUROC measures the area under the curve when plotting test statistic
    vs number of replacements. Lower AUROC indicates better faithfulness
    (faster decrease in test statistic when replacing important regions).

    Parameters:
    -----------
    results_dict : dict
        Dictionary mapping method names to their results

    Returns:
    --------
    auroc_dict : dict
        Dictionary mapping method names to their AUROC values
    """
    auroc_dict = {}

    for method, results in results_dict.items():
        n_replaced = np.array(results['n_replaced'])
        test_stats = np.array(results['test_statistics'])

        # Compute AUROC without normalization
        if len(n_replaced) > 1:
            auroc_value = auc(n_replaced, test_stats)
        else:
            auroc_value = test_stats[0] if len(test_stats) > 0 else 0.0

        auroc_dict[method] = auroc_value

    return auroc_dict


def print_auroc_table(auroc_dict):
    """
    Print a formatted table of AUROC values for each method.

    Parameters:
    -----------
    auroc_dict : dict
        Dictionary mapping method names to their AUROC values
    """
    print("\n" + "=" * 60)
    print("FAITHFULNESS EVALUATION - AUROC COMPARISON")
    print("=" * 60)
    print(f"{'Method':<15} {'AUROC':<12} {'Note'}")
    print("-" * 60)

    # Sort by AUROC (ascending - lower is better for faithfulness)
    sorted_methods = sorted(auroc_dict.items(), key=lambda x: x[1])

    best_method = sorted_methods[0][0]

    for method, auroc in sorted_methods:
        note = " <- BEST (Lowest)" if method == best_method else ""
        print(f"{method.upper():<15} {auroc:<12.4f} {note}")

    print("-" * 60)
    print(f"Best method: {best_method.upper()} (AUROC = {auroc_dict[best_method]:.4f})")
    print("Note: Lower AUROC indicates better faithfulness")
    print("=" * 60 + "\n")


def plot_max_sensitivity_results(sensitivity_results_dict):
    """
    Plot max-sensitivity results with grouped bar plots.

    Parameters:
    -----------
    sensitivity_results_dict : dict
        Dictionary mapping method names to their max-sensitivity results
    """
    methods = list(sensitivity_results_dict.keys())
    n_methods = len(methods)

    # Extract robustness coefficients (std/mean) for test stat and p-value
    # (should be same across methods as they measure the same statistical test)
    first_method = methods[0]
    test_stat_robustness = sensitivity_results_dict[first_method]['test_stat_robustness_coef']
    p_value_robustness = sensitivity_results_dict[first_method]['p_value_robustness_coef']

    # Attribution sensitivity per method
    attr_g1_mean = [sensitivity_results_dict[m]['mean_attr_sensitivity_group1'] for m in methods]
    attr_g1_max = [sensitivity_results_dict[m]['max_attr_sensitivity_group1'] for m in methods]

    attr_g2_mean = [sensitivity_results_dict[m]['mean_attr_sensitivity_group2'] for m in methods]
    attr_g2_max = [sensitivity_results_dict[m]['max_attr_sensitivity_group2'] for m in methods]

    # Create figure with 1x4 layout
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    # Bar settings for method comparison
    x = np.arange(n_methods)
    width = 0.35

    # Plot 1: Test Statistic Robustness Coefficient (std/mean)
    ax = axes[0]
    bars = ax.bar(0, test_stat_robustness, color='steelblue', alpha=0.8, width=0.5)
    ax.set_ylabel('Robustness Coefficient (std/mean)', fontsize=12)
    ax.set_title('Test Statistic Robustness', fontsize=14, fontweight='bold')
    ax.set_xticks([0])
    ax.set_xticklabels(['Test Stat'])
    ax.grid(True, alpha=0.3, axis='y')

    # Plot 2: P-value Robustness Coefficient (std/mean)
    ax = axes[1]
    bars = ax.bar(0, p_value_robustness, color='steelblue', alpha=0.8, width=0.5)
    ax.set_ylabel('Robustness Coefficient (std/mean)', fontsize=12)
    ax.set_title('P-value Robustness', fontsize=14, fontweight='bold')
    ax.set_xticks([0])
    ax.set_xticklabels(['P-value'])
    ax.grid(True, alpha=0.3, axis='y')

    # Plot 3: Attribution Group 1
    ax = axes[2]
    ax.bar(x - width/2, attr_g1_mean, width, label='Avg', color='orange', alpha=0.8)
    ax.bar(x + width/2, attr_g1_max, width, label='Max', color='red', alpha=0.8)
    ax.set_ylabel('Attribution Sensitivity (MSE)', fontsize=12)
    ax.set_title('Attribution Sensitivity - Group 1', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([m.upper() for m in methods])
    ax.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Plot 4: Attribution Group 2
    ax = axes[3]
    ax.bar(x - width/2, attr_g2_mean, width, label='Avg', color='orange', alpha=0.8)
    ax.bar(x + width/2, attr_g2_max, width, label='Max', color='red', alpha=0.8)
    ax.set_ylabel('Attribution Sensitivity (MSE)', fontsize=12)
    ax.set_title('Attribution Sensitivity - Group 2', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([m.upper() for m in methods])
    ax.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Set the same y-axis limits for both attribution plots
    all_attr_values = attr_g1_mean + attr_g1_max + attr_g2_mean + attr_g2_max
    y_min = min(all_attr_values) * 0.9
    y_max = max(all_attr_values) * 1.1
    axes[2].set_ylim([y_min, y_max])
    axes[3].set_ylim([y_min, y_max])

    plt.tight_layout()
    plt.savefig('max_sensitivity_comparison.png', dpi=150, bbox_inches='tight')
    print("\nMax-sensitivity plot saved to: max_sensitivity_comparison.png")
    plt.show()


def print_max_sensitivity_table(sensitivity_results_dict):
    """
    Print formatted table of max-sensitivity results.

    Parameters:
    -----------
    sensitivity_results_dict : dict
        Dictionary mapping method names to their max-sensitivity results
    """
    print("\n" + "=" * 90)
    print("MAX-SENSITIVITY EVALUATION")
    print("=" * 90)

    # Print robustness coefficients (same across all methods)
    first_method = list(sensitivity_results_dict.keys())[0]
    res_first = sensitivity_results_dict[first_method]
    print(f"Test Statistic Robustness Coef (std/mean): {res_first['test_stat_robustness_coef']:.4f}")
    print(f"P-value Robustness Coef (std/mean):        {res_first['p_value_robustness_coef']:.4f}")
    print("-" * 90)

    # Print attribution sensitivities per method
    print(f"{'Method':<12} {'Attr G1':<25} {'Attr G2':<25}")
    print(f"{'':12} {'(Avg/Max)':<25} {'(Avg/Max)':<25}")
    print("-" * 90)

    for method in sensitivity_results_dict.keys():
        res = sensitivity_results_dict[method]

        print(f"{method.upper():<12} "
              f"{res['mean_attr_sensitivity_group1']:.4f}/{res['max_attr_sensitivity_group1']:.4f}             "
              f"{res['mean_attr_sensitivity_group2']:.4f}/{res['max_attr_sensitivity_group2']:.4f}")

    print("-" * 90)
    print("Note: Lower robustness coefficient = more stable; Lower attribution sensitivity = more robust")
    print("=" * 90 + "\n")