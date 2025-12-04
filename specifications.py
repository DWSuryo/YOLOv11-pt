import torch
from thop import profile
import csv
import os

# Import your model builders from nn.py
# (Ensure this script is in the same folder as nn.py)
from nets.nn import yolo_v11_n, yolo_v11_t, yolo_v11_s, yolo_v11_m, yolo_v11_l, yolo_v11_x

def get_stats(name, model_builder, input_size=640):
    device = torch.device("cpu")
    
    # 1. Initialize
    model = model_builder(num_classes=80).to(device)
    
    # 2. Fuse (Merge Conv+BN for accurate deployment stats)
    model.eval()
    try:
        model.fuse()
    except AttributeError:
        print(f"Warning: {name} could not be fused (check .fuse() method).")

    # 3. Dummy Input
    dummy_input = torch.randn(1, 3, input_size, input_size).to(device)

    # 4. Profile
    # We switch to train mode temporarily for profiling if the Head 
    # structure is too complex for thop in eval mode (common YOLO issue)
    model.train() 
    
    try:
        flops, params = profile(model, inputs=(dummy_input, ), verbose=False)
    except Exception as e:
        print(f"Error profiling {name}: {e}")
        return None

    # 5. Convert units
    # GFLOPs = operations / 10^9 * 2 (for MACs)
    gflops = flops / 1e9 * 2 
    params_m = params / 1e6
    
    return {
        "Model": name,
        "Input Size": input_size,
        "Params (M)": round(params_m, 3),
        "GFLOPs": round(gflops, 3)
    }

if __name__ == "__main__":
    # --- Configuration ---
    input_size = 640
    csv_filename = "model_stats.csv"
    
    models_to_test = {
        "YOLOv11-n": yolo_v11_n,
        # "YOLOv11-Tiny": yolo_v11_t,
        "YOLOv11-s": yolo_v11_s,
        "YOLOv11-m": yolo_v11_m,
        "YOLOv11-l": yolo_v11_l,
        "YOLOv11-x": yolo_v11_x, 
    }

    results = []

    print(f"Starting Benchmark (Input Size: {input_size}x{input_size})...\n")

    # --- Run Loop ---
    for name, builder in models_to_test.items():
        stats = get_stats(name, builder, input_size)
        if stats:
            results.append(stats)
            # Print a dot to show progress
            print(f". Processed {name}")

    # --- 1. Display Table in Terminal ---
    print("\n" + "="*65)
    print(f"| {'Model Name':<20} | {'Size':<6} | {'Params (M)':>12} | {'GFLOPs':>12} |")
    print("-" * 65)
    
    for row in results:
        print(f"| {row['Model']:<20} | {row['Input Size']:<6} | {row['Params (M)']:>12.2f} | {row['GFLOPs']:>12.2f} |")
    
    print("="*65 + "\n")

    # --- 2. Save to CSV ---
    try:
        with open(csv_filename, mode='w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=["Model", "Input Size", "Params (M)", "GFLOPs"])
            writer.writeheader()
            writer.writerows(results)
        print(f"✅ Successfully saved results to '{csv_filename}'")
        print(f"   path: {os.path.abspath(csv_filename)}")
    except IOError as e:
        print(f"❌ Error saving CSV: {e}")