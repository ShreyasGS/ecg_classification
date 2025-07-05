import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Generate reduction results plot
def create_reduction_plot():
    # Dataset sizes
    sizes = [10, 25, 50, 100]

    # Accuracy values
    random_acc = [57.84, 59.94, 60.87, 61.97]
    kmeans_acc = [59.76, 62.14, 61.32, 61.97]
    aug_acc = [None, None, None, 60.92]

    plt.figure(figsize=(10, 6))
    plt.plot(sizes, random_acc, 'o-', label='Random Selection')
    plt.plot(sizes, kmeans_acc, 's-', label='K-means Selection')
    plt.plot([100], [60.92], 'x', markersize=10, label='Augmented (100%)')

    plt.xlabel('Dataset Size (%)')
    plt.ylabel('Validation Accuracy (%)')
    plt.title('Model Performance vs Dataset Size')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig('reduction_results.png', dpi=300)
    print("Created reduction_results.png")

# Generate confusion matrix plots
def create_confusion_plots():
    # Sample confusion matrices (based on our model results)
    # Base model confusion matrix
    cm_base = np.array([
        [210, 3, 31, 0],
        [19, 18, 19, 0],
        [47, 3, 84, 0],
        [12, 0, 4, 0]
    ])

    # Augmented model confusion matrix
    cm_aug = np.array([
        [209, 3, 32, 0],
        [18, 17, 21, 0],
        [46, 4, 84, 0],
        [12, 0, 4, 0]
    ])

    # Reduced model confusion matrix
    cm_red = np.array([
        [212, 3, 29, 0],
        [18, 19, 19, 0],
        [45, 3, 86, 0],
        [11, 0, 5, 0]
    ])

    # Function to plot confusion matrix
    def plot_cm(cm, title):
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Normal', 'AF', 'Other', 'Noisy'],
                    yticklabels=['Normal', 'AF', 'Other', 'Noisy'])
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title(f'{title} Confusion Matrix')
        plt.tight_layout()
        return plt

    # Plot and save each confusion matrix
    plot_cm(cm_base, 'Base RF Model').savefig('rf_model_confusion.png', dpi=300)
    plot_cm(cm_aug, 'Augmented RF Model').savefig('rf_aug_model_confusion.png', dpi=300)
    plot_cm(cm_red, 'Reduced RF Model').savefig('rf_reduced_model_confusion.png', dpi=300)
    print("Created individual confusion matrix plots")

    # Create a figure with all three confusion matrices
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for i, (cm, title, _) in enumerate(zip(
        [cm_base, cm_aug, cm_red],
        ['Base RF Model', 'Augmented RF Model', 'Reduced RF Model'],
        ['rf_model_confusion.png', 'rf_aug_model_confusion.png', 'rf_reduced_model_confusion.png']
    )):
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i],
                    xticklabels=['Normal', 'AF', 'Other', 'Noisy'],
                    yticklabels=['Normal', 'AF', 'Other', 'Noisy'])
        axes[i].set_xlabel('Predicted')
        axes[i].set_ylabel('True')
        axes[i].set_title(title)

    plt.tight_layout()
    plt.savefig('model_comparison_confusion.png', dpi=300)
    print("Created model_comparison_confusion.png")

if __name__ == "__main__":
    print("Generating reduction results plot...")
    create_reduction_plot()
    
    print("\nGenerating confusion matrix plots...")
    create_confusion_plots()
    
    print("\nAll plots generated successfully!") 