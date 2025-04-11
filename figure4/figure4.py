# this python file is write for resubmission figure.4
import numpy as np
import os

# Now you are given 5 txt file in each category (truth, prediction), all the values inside the txt are 0 and 1. I want to keep the truth file unchanged, and if there is a mismatch
#in the same stage txt file between prediction and truth
# if truth is 0 and pred is 1 then change the pred txt that index to 2
# if truth is 1 and pred is 0 then change the pred txt that index to 3
# files have the naming format of APET_i, TPET_i, where i=0-4
#figure4

main_path = os.path.join(os.path.dirname(__file__), "data")
categories = ["APET", "TPET"]
num_files = 5

for category in categories:
    for i in range(num_files):
        # File paths
        truth_path = os.path.join(main_path, f"truth/{category}/{category}_{i}.txt")
        pred_path = os.path.join(main_path, f"pred/{category}/{category}_{i}.txt")
        new_pred_path = os.path.join(main_path, f"pred/{category}/{category}_{i}_new.txt")

        # Load as float
        truth_float = np.loadtxt(truth_path)
        pred_float = np.loadtxt(pred_path)

        # Convert to binary for comparison
        truth_bin = (truth_float > 0.5).astype(int)
        pred_bin = (pred_float > 0.5).astype(int)

        # Calculate accuracy
        correct = (truth_bin == pred_bin).sum()
        total = truth_bin.size
        accuracy = correct / total
        print(f"✅ {category}_{i} Accuracy: {accuracy:.4f}")

        # Copy original predictions for modification
        updated_pred = pred_float.copy()

        # Apply mismatch rules
        updated_pred[(truth_bin == 0) & (pred_bin == 1)] = 2.0  # False Positive
        updated_pred[(truth_bin == 1) & (pred_bin == 0)] = 3.0  # False Negative

        # Save result as float (e.g., 0.00, 1.00, 2.00, 3.00)
        np.savetxt(new_pred_path, updated_pred, fmt='%.2f')
        print(f"💾 Saved: {new_pred_path}")