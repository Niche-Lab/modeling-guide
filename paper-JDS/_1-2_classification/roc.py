import numpy as np

# True labels and predicted probabilities
true_labels = np.array([0, 1, 1, 0, 1, 0, 1, 0, 1, 0])
predicted_probs = np.array([0.1, 0.9, 0.8, 0.2, 0.7, 0.3, 0.85, 0.15, 0.75, 0.05])

# Sort by predicted probabilities
sorted_indices = np.argsort(-predicted_probs)
sorted_true_labels = true_labels[sorted_indices]

# Initialize variables
TPR_list = []
FPR_list = []
num_pos = sum(true_labels)
num_neg = len(true_labels) - num_pos

TP = 0
FP = 0

# Calculate TPR and FPR
for label in sorted_true_labels:
    if label == 1:
        TP += 1
    else:
        FP += 1
    TPR = TP / num_pos
    FPR = FP / num_neg
    TPR_list.append(TPR)
    FPR_list.append(FPR)

print("TPR: ", TPR_list)
print("FPR: ", FPR_list)

import matplotlib.pyplot as plt

plt.figure()
plt.plot(FPR_list, TPR_list, marker="o")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic Curve")
plt.show()
