import sleap
from sleap.nn.evals import evaluate
import numpy as np

# 1. Load the "Ground Truth"
gt = sleap.load_file("test.pkg.slp") 

# 2. Load the "Inference Test"
pr = sleap.load_file("test.predictions.slp") 

# 3. Grade the exam!
metrics = evaluate(gt, pr)

# 4. Calculate and print the average pixel error
mean_error = np.nanmean(metrics["dist.dists"])
print(f"SLEAP Final Test Error: {mean_error:.2f} pixels")