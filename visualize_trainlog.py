import re
import matplotlib.pyplot as plt
import numpy as np

def parse_train_log(file_path):
    """
    Parse train_log.txt and extract epoch, train_loss, val_loss,
    baseline_acc, mvcv_acc, and learning_rate.
    """
    epochs = []
    train_losses = []
    val_losses = []
    baseline_accs = []
    mvcv_accs = []
    lrs = []
    improved_epochs = []  # epochs where best model was saved

    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Pattern for the main epoch line
    epoch_pattern = re.compile(
        r'Epoch\s+(?P<epoch>\d+)/\d+:\s+'
        r'Train Loss:\s+(?P<train_loss>[\d.]+)\s+\|\s+'
        r'Val Loss:\s+(?P<val_loss>[\d.]+)\s+\|\s+'
        r'Baseline Acc:\s+(?P<baseline>[\d.]+)%\s+\|\s+'
        r'MVCP Acc:\s+(?P<mvcv>[\d.]+)%\s+\|\s+'
        r'LR:\s+(?P<lr>[\d.e+-]+)'
    )

    # Pattern for "Accuracy Improved" line
    improved_pattern = re.compile(
        r'✨ Accuracy Improved: .*? -> (?P<acc>[\d.]+)%'
    )

    for line in lines:
        m = epoch_pattern.search(line)
        if m:
            epochs.append(int(m.group('epoch')))
            train_losses.append(float(m.group('train_loss')))
            val_losses.append(float(m.group('val_loss')))
            baseline_accs.append(float(m.group('baseline')))
            mvcv_accs.append(float(m.group('mvcv')))
            lrs.append(float(m.group('lr')))

        # Check for improvement line (optional)
        imp = improved_pattern.search(line)
        if imp:
            # We'll use the epoch from the last parsed epoch (assume improvement line follows immediately)
            if epochs:
                improved_epochs.append(epochs[-1])

    return (epochs, train_losses, val_losses,
            baseline_accs, mvcv_accs, lrs, improved_epochs)


def plot_metrics(epochs, train_loss, val_loss, baseline_acc, mvcv_acc, lr, improved_epochs):
    """Create a 3‑panel figure with loss, accuracy, and LR plots."""
    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

    # 1) Loss
    axes[0].plot(epochs, train_loss, 'b-o', label='Train Loss', markersize=4)
    axes[0].plot(epochs, val_loss, 'r-o', label='Val Loss', markersize=4)
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, linestyle='--', alpha=0.6)
    axes[0].set_title('Training and Validation Loss')

    # 2) Accuracy
    axes[1].plot(epochs, baseline_acc, 'g-o', label='Baseline (fused) Acc', markersize=4)
    axes[1].plot(epochs, mvcv_acc, 'm-o', label='MVCP Acc', markersize=4)
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].legend()
    axes[1].grid(True, linestyle='--', alpha=0.6)
    axes[1].set_title('Validation Accuracy')

    # Mark improved epochs
    for ep in improved_epochs:
        axes[1].axvline(x=ep, color='orange', linestyle='--', alpha=0.5, linewidth=1.5)
        # Add a small text
        axes[1].text(ep, 5, 'best', rotation=90, fontsize=8, color='orange', alpha=0.7)

    # 3) Learning rate
    axes[2].plot(epochs, lr, 'k-o', markersize=4)
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Learning Rate')
    axes[2].grid(True, linestyle='--', alpha=0.6)
    axes[2].set_title('Learning Rate Schedule')

    plt.tight_layout()
    plt.savefig('training_metrics.png', dpi=150)
    plt.show()
    print("Plot saved as 'training_metrics.png'")


if __name__ == "__main__":
    import sys
    log_file = "OCR-MultiFrame-ICPR/train_log.txt"
    if len(sys.argv) > 1:
        log_file = sys.argv[1]

    print(f"Parsing {log_file} ...")
    (epochs, train_loss, val_loss,
     baseline_acc, mvcv_acc, lr, improved_epochs) = parse_train_log(log_file)

    if not epochs:
        print("No epoch data found. Please check the log file format.")
        sys.exit(1)

    print(f"Found data for {len(epochs)} epochs.")
    print(f"Best baseline accuracy: {max(baseline_acc):.2f}% at epoch {epochs[baseline_acc.index(max(baseline_acc))]}")
    print(f"Improvements at epochs: {improved_epochs}")

    plot_metrics(epochs, train_loss, val_loss, baseline_acc, mvcv_acc, lr, improved_epochs)