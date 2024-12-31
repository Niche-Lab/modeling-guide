from cProfile import label
from matplotlib import markers
import numpy as np
import matplotlib.pyplot as plt


class Evaluation:
    def __init__(self):
        self.predictions = Prediction()
        self.logits = self.predictions.logits
        self.labels = self.predictions.labels
        self.sorted_label = None
        self.ROC = None
        self.PR = None

    def sort(self, descending=True):
        if descending:
            sorted_indices = np.argsort(-self.logits)
        else:
            sorted_indices = np.argsort(self.logits)
        self.sorted_label = self.labels[sorted_indices].copy()

    def calculate_PR(self, pos_is_1=True):
        self.PR = dict(
            {
                "precision": [1],
                "recall": [0],
            }
        )
        POS_LBS = 1 if pos_is_1 else 0
        if pos_is_1:
            self.sort(descending=True)
        else:
            self.sort(descending=False)
        TP = 0
        FP = 0
        FN = len(self.sorted_label[self.sorted_label == POS_LBS])
        for i in range(len(self.sorted_label)):
            if self.sorted_label[i] == POS_LBS:
                TP += 1
                FN -= 1
            else:
                FP += 1
            # update
            self.PR["precision"].append(TP / (TP + FP))
            self.PR["recall"].append(TP / (TP + FN))

    def calculate_ROC(self, pos_is_1=True):
        self.ROC = dict(
            {
                "recall": [],
                "fpr": [],
            }
        )
        POS_LBS = 1 if pos_is_1 else 0
        if pos_is_1:
            self.sort(descending=True)
        else:
            self.sort(descending=False)
        TP = 0
        FP = 0
        N_POS = len(self.sorted_label[self.sorted_label == POS_LBS])
        N_NEG = len(self.sorted_label[self.sorted_label != POS_LBS])
        for i in range(len(self.sorted_label)):
            if self.sorted_label[i] == POS_LBS:
                TP += 1
            else:
                FP += 1
            # update
            self.ROC["recall"].append(TP / N_POS)
            self.ROC["fpr"].append(FP / N_NEG)

    def vis_PR(self):
        plt.figure()
        plt.plot(self.PR["recall"], self.PR["precision"], marker="o")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curve")
        plt.show()

    def vis_ROC(self):
        plt.figure()
        plt.plot(self.ROC["fpr"], self.ROC["recall"], marker="o")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Receiver Operating Characteristic Curve")
        plt.show()


class Prediction:
    def __init__(self):
        # members
        self.TP = None
        self.FP = None
        self.TN = None
        self.FN = None
        self.labels = None
        self.probs = None

        # init
        self.init_categories()
        self.init_labels()
        self.init_probs()
        self.shuffle()

    def init_categories(self):
        self.TP = Category(1, (0.8, 1.0), positive=True)
        self.FP = Category(1, (0.6, 0.8), positive=False)
        self.TN = Category(5, (0.0, 0.2), positive=False)
        self.FN = Category(3, (0.2, 0.4), positive=True)

    def init_labels(self):
        self.labels = np.concatenate(
            [
                self.TP.labels,
                self.FP.labels,
                self.TN.labels,
                self.FN.labels,
            ]
        )

    def init_probs(self):
        self.logits = np.concatenate(
            [
                self.TP.probs,
                self.FP.probs,
                self.TN.probs,
                self.FN.probs,
            ]
        )

    def shuffle(self):
        shuffle_indices = np.random.permutation(len(self.labels))
        self.labels = self.labels[shuffle_indices]
        self.logits = self.logits[shuffle_indices]


class Category:
    def __init__(self, n, prob_range, positive=True):
        self.n = n
        self.probs = np.random.uniform(prob_range[0], prob_range[1], n)
        self.labels = np.ones(n) if positive else np.zeros(n)


np.random.seed(24061)
import seaborn as sns

# calculate metrics
e = Evaluation()


# PLOT ROC --------------------
e.calculate_ROC(pos_is_1=True)
recall, fpr = e.ROC["recall"], e.ROC["fpr"]
e.calculate_ROC(pos_is_1=False)
recall_r, fpr_r = e.ROC["recall"], e.ROC["fpr"]

# get the first two color from the palette Set3
colors = sns.color_palette("Set2", 2)
# plot
fig, ax = plt.subplots(figsize=(8, 8))
ax.plot(
    fpr,
    recall,
    # marker="o",
    markersize=10,
    color=colors[0],
    label="ROC (Original labels)",
)
ax.plot(
    fpr_r,
    recall_r,
    # marker="o",
    markersize=10,
    color=colors[1],
    label="ROC (Inversed labels)",
)
# show legend
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("Receiver Operating Characteristic Curve")
# random guess
ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Random Guess")
# set ticks
ax.set_xticks(np.arange(0, 1.1, 0.1))
ax.set_yticks(np.arange(0, 1.1, 0.1))
ax.set_ylim([-0.05, 1.05])
# set grid
ax.grid(which="major", axis="both", linestyle="--")
ax.grid(which="minor", axis="both", linestyle="--")
ax.legend(loc="lower right")
plt.show()

fig.savefig("ROC.png", dpi=300)
print("ROC AUC: ", np.trapz(e.ROC["recall"], e.ROC["fpr"]))


# PLOT PR --------------------
e.calculate_PR(pos_is_1=True)
rc, pr = e.PR["recall"], e.PR["precision"]
e.calculate_PR(pos_is_1=False)
rc_r, pr_r = e.PR["recall"], e.PR["precision"]
# plot
fig, ax = plt.subplots(figsize=(8, 8))
ax.plot(
    rc,
    pr,
    # marker="o",
    markersize=10,
    color=colors[0],
    label="PR (Original labels)",
)
ax.plot(
    rc_r,
    pr_r,
    # marker="o",
    markersize=10,
    color=colors[1],
    label="PR (Inversed labels)",
)
# show legend
ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
ax.set_title("Precision-Recall Curve")
# set ticks
ax.set_xticks(np.arange(0, 1.1, 0.1))
ax.set_yticks(np.arange(0, 1.1, 0.1))
ax.set_ylim([-0.05, 1.05])
# set grid
ax.grid(which="major", axis="both", linestyle="--")
ax.grid(which="minor", axis="both", linestyle="--")
ax.legend(loc="lower right")
plt.show()
fig.savefig("PR.png", dpi=300)
print("PR AUC: ", np.trapz(e.PR["precision"], e.PR["recall"]))


# MCC
import sklearn.metrics as metrics

lb, lg = e.labels, e.logits
lb_r, lg_r = 1 - lb, 1 - lg

mcc = []
mcc_r = []
for t in np.arange(0.0, 1.0, 0.05):
    mcc.append(metrics.matthews_corrcoef(lb, lg > t))
    mcc_r.append(metrics.matthews_corrcoef(lb_r, lg_r > t))

plt.figure()
plt.plot(
    mcc,
    color=colors[0],
    label="MCC (Original labels)",
)
plt.plot(
    mcc_r,
    color=colors[1],
    label="MCC (Inversed labels)",
)

plt.xlim([1, len(mcc) - 1])
plt.ylim([0.0, 1.0])
plt.xlabel("Threshold")
plt.ylabel("MCC")
plt.title("Matthews Correlation Coefficient")
plt.legend(loc="upper right")
plt.savefig("MCC.png", dpi=300)
