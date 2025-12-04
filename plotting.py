import csv
import matplotlib.pyplot as plt
import os

# CSV read and plot mAP function
def plot_mAP(args):
    mAP_list = []
    epoch_list = []

    # --- MODIFICATION: Use args.save_dir and standard filename ---
    csv_path = os.path.join(args.save_dir, "results.csv")
    
    if not os.path.exists(csv_path):
        print(f"Warning: Could not find CSV at {csv_path}, skipping mAP plot.")
        return

    with open(csv_path, "r") as file:
        reader = csv.DictReader(file)
        for row in reader:
            epoch_list.append(int(row["epoch"]))
            mAP_list.append(float(row["mAP"]))

    if not mAP_list:
        print("Warning: CSV is empty.")
        return

    # Find the best mAP and corresponding epoch
    best_mAP = max(mAP_list)
    best_epoch = epoch_list[mAP_list.index(best_mAP)]
    last_mAP = mAP_list[-1]
    last_epoch = epoch_list[-1]
    # Plot mAP vs. epochs using Matplotlib
    import matplotlib.pyplot as plt

    # Create subplots
    fig, ax = plt.subplots(1, 1, layout='constrained')
    # Plot mAP vs. epochs
    ax.plot(epoch_list, mAP_list, label=f'mAP (last: {last_mAP:.3f}, best: {best_mAP:.3f})')
    # Highlight best mAP epoch
    ax.scatter(best_epoch, best_mAP, color='red', label=f'Best mAP at epoch {best_epoch}', zorder=3)
    # Set axis limits
    ax.set_xlabel("Epoch")
    ax.set_ylabel("mAP")
    # ax.set_xlim(0, max(epoch_list))
    ax.set_ylim(0, max(mAP_list) * 1.1 if max(mAP_list) > 0 else 1.0) # Add 10% headroom
    ax.grid(True)
    # Set title and subtitle
    # version = "YOLOv11 version n"  # Change this dynamically based on actual version
    # epochs = last_epoch  # Total epochs
    fig.suptitle("mAP vs. Epochs")
    ax.set_title(f"YOLOv11 version {args.version} at {args.epochs} epochs")
    # Position legend above the graph
    # ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=2, frameon=False)
    fig.legend(loc="outside lower center")

    # --- MODIFICATION: Save to the versioned folder ---
    save_path = os.path.join(args.save_dir, "mAP_vs_epochs.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Saved mAP plot to {save_path}")