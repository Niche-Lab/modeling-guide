## Classification

Classification models aim to predict categorical outcomes such as 'healthy' or 'sick,' 'susceptible' or 'resistant,' and 'high yield' or 'low yield.’ This section presents a hypothetical example to highlight how the choice of different performance metrics can lead to different interpretations of a model's effectiveness. The example focuses on binary classification, where the outcome is either positive (Y=1) or negative (Y=0). Suppose a binary classification model always outputs a probability between 0 and 1, indicating the likelihood that a sample belongs to the positive class. It assumes that the model has high confidence in correctly predicting 1 out of 4 positive samples, and 5 out of 6 negative samples. This example intends to illustrate a scenario where the positive outcome is rare, such as predicting the onset of a calving event in dairy cows (Ouellet et al 2016 and Borchers 2017). The model performance can be summarized in the following probability distribution:

$$
P(X, Y) = \begin{cases}
\frac{1}{10} \times \text{Uniform}(0.8, 1.0) & \text{if } Y=1 \\
\frac{1}{10} \times \text{Uniform}(0.6, 0.8) & \text{if } Y=1  \\
\frac{5}{10} \times \text{Uniform}(0.0, 0.2) & \text{if } Y=0 \\
\frac{3}{10} \times \text{Uniform}(0.2, 0.4) & \text{if } Y=0
\end{cases}
$$

Where $X$ is a random variable representing the predicted probabilities sampled from a uniform distribution $\text{Uniform}(a, b)$ between $a$ and $b$, and $Y$ representing the ground truth labels. The simulated result is shown in Figure 2

| Ground Truth | Prediction Probability |
|--------------|------------------------|
| (+) | 0.99 |
| (-) | 0.70 |
| (+) | 0.38 |
| (+) | 0.33 |
| (+) | 0.26 |
| (-) | 0.16 |
| (-) | 0.15 |
| (-) | 0.14 |
| (-) | 0.12 |
| (-) | 0.07 |

|  |   |Prediction| Prediction|
|--|---|-----|-----|
|  |   | (+) | (-) |
| Ground Truth | (+) | 1 | 3 |
| Ground Truth | (-) | 1 | 5 |

Figure 2. Simulated hypothetical example of binary classification. Upper: The ground truth and prediction probability. Lower: The confusion matrix of the prediction at a threshold of 0.5, followed by classification metrics of accuracy, precision, recall, MCC, PR curve AUC, and ROC curve AUC. The performance of the original labels serves as a baseline for comparison. Any better performance metrics from the inveted labels are highlighted in bold and underscored.

In addition to the original labels, this example also examines a scenario with inveted labels. Since most classification metrics prioritize positive samples, it is generally advisable to designate the event of interest as the positive class in binary classification problems. Reversing the labels illustrates the potential overestimation of model performance when the more common, but less significant, background event is mistakenly marked as the positive class. It is important to note that reversing the labels affects only the interpretation of model performance, not the model configuration or parameters.

To evaluate classification performance, one must first establish a confidence threshold to dichotomize the prediction probabilities. For instance, if a prediction probability exceeds the threshold, the sample is labeled positive. By default, the threshold is set at 0.5 for its simplicity. Consider the third data row: with a prediction probability of 0.38, falling below the threshold, the sample is deemed negative, resulting in a false negative classification since the ground truth is positive. It is worth mentioning that this threshold is adjustable to fine-tune model performance for particular uses.
A confusion matrix, as shown in Figure 2. Lower, effectively encapsulates prediction outcomes. The rows in this 2x2 matrix correspond to ground truth, while its columns reflect predictions. Correct predictions populate the diagonal cells, and errors fill the off-diagonal ones. This matrix serves as the foundation for computing various metrics to assess model performance, which we will explore in the subsequent sections.

### Accuracy

$$
Accuracy = \frac{TP + TN}{TP + TN + FP + FN} = \frac{Total Correct Predictions}{Total Predictions}
\tag{6}
$$
Accuracy quantifies overall model performance by measuring the rate of correct predictions, as defined in Equation 5, with TP (true positives), TN (true negatives), FP (false positives), and FN (false negatives) defining the respective outcomes. With a 0.5 threshold in this example, the accuracy stands at 0.60. This figure might suggest modest efficacy, marginally surpassing random chance, with an accuracy of 0.50. Nonetheless, the same accuracy level could be achieved by classifying every sample as negative in an imbalanced dataset where negatives are predominant. Hence, relying solely on accuracy to assess models can be misleading, indicating the need for additional metrics to ensure robust evaluation.

### Precision, Recall, and Precision-Recall Curve

$$
Precision = \frac{TP}{TP + FP} = \frac{TP}{Total Predicted Positive}
$$
$$
Recall (Sensitivity) = \frac{TP}{TP + FN} = \frac{TP}{Total Actual Positive}
$$

Precision and recall refine the assessment of a classification model by offering insights that accuracy alone may overlook. Precision calculates the fraction of true positives among all positive predictions, essentially measuring the trustworthiness of positive predictions made by the model. High precision is crucial in scenarios where false positives incur significant cost, and false negatives are more tolerable. For instance, in contexts where clinical treatments and culling are expensive, such as detecting bovine tuberculosis  (referenced by Denholm et al., 2020) or mastitis (cited by Kandeel, 2019) using non-invasive methods, a high-precision model is crucial to minimize unnecessary costs and interventions from false positives.
On the other hand, recall, also knwon as sensitivity, quantifies the ratio of true positives to all actual positives, assessing the model's ability to identify positive cases. High recall is essential where missing a positive case has serious consequences, or where false positives are easily rectifiable. For instance, detecting lameness or abnormal gait is crucial, as these can indicate underlying pathologies (leary 2020) and impact welfare-related transport decisions (stojkov 2018). An automated detection system (alsaaod, 2019, leary 2020, kang 2020) with high recall can mitigate economic losses by flagging at-risk cows. The benefit here lies in the feasibility of re-examining false positives, thus preventing more severe outcomes from undetected cases.

Figure 3. (Left) Precision-recall (PR) curve and (Right) Receiver operating characteristic (ROC) curve for the hypotehtical example are displayed. The performance at confidence thresholds of 0.25 and 0.50 is highlighted by dots. Original labels are marked in green, while inverted labels appear in orange. The Area Under the Curve (AUC) is depicted at the center of each curve.

In the hypothetical example, setting a threshold of 0.5 yields precision and recall values of 0.5 and 0.25, respectively. These metrics deliver a more interpretable information indicating that only half of the positive predictions are correct, and just a quarter of the actual positives are detected. This contrasts with an accuracy of 0.6, which may appear misleadingly high due to the abundance of negative samples. The chosen confidence threshold significantly impacts precision and recall. While the trade-off between these two metrics is not always linear, it is generally observed that a higher threshold increases precision but decreases recall, and vice versa. A high threshold indicates a conservative approach in predicting positives, reducing false positives and thus enhancing precision. However, this often leads to missing actual positive cases, lowering recall. The Precision-Recall (PR) curve is an essential tool for evaluating model performance across various thresholds. Plotted with recall on the x-axis and precision on the y-axis, this curve is derived by computing these metrics at different thresholds (see Figure 3, Left). The Area Under the Curve (AUC) provides a summary measure of the PR curve's overall performance. A model's effectiveness is generally indicated by how close a point on the PR curve is to the top-right corner. For example, at a threshold of 0.25, which is positioned near the top-right of the PR curve, the model demonstrates impressive performance with an accuracy of 0.90, precision at 0.80, and recall at 1.00.

However, it is worth re-emphasizing that precision and recall focus predominantly on positive samples. Inappropriately assigning a predominant background event as the positive class can lead to skewed interpretations. This pitfall is demonstrated in this example by inverting the labels. At a threshold of 0.50, precision increases from 0.50 to 0.63, and recall jumps from 0.25 to 0.83. With the threshold set at 0.25, precision drops to 0.66 from 0.80, while recall remains unchanged. The PR AUC also rises from 0.76 to 0.94. Such shifts in metrics, driven merely by label rearrangement unrelated to the data or model characteristics, underscore the importance of label-invariant metrics that remain unaffected by label assignments.



### Receiver Operating Characteristic (ROC) Curve

The Receiver Operating Characteristic (ROC) curve is a label-invariant and threshold-invariant alternative to the PR curve. The x-axis is the false positive rate (FPR) and the y-axis is the true positive rate (TPR), which is equivalent to recall.

$$
FPR = \frac{FP}{FP + TN} = \frac{FP}{Total Actual Negative}
$$
$$
TPR (recall) = \frac{TP}{TP + FN} = \frac{TP}{Total Actual Positive}
$$

The ROC curve can be intepreted as how much cost is needed to capture true positives. If the curve climbs steeply from the left side, it means that the model can capture most true positives with a low cost of false positives. A random guess, which yield a 50% chance of making a correct prediction, is represented by a diagnoal line in the ROC curve. This curve is widely used in reporting genetics markers in Genome-Wide Association Studies (GWAS) [ref 7-9], as whether the top-associated markers (i.e., prediction with high positive probability) can be identified is more important than considering the entire list of prediction quality. In the hypothetical example, the ROC curve show a labe-invariant AUC of 0.875, which is the same for both the original labels and the reversed labels. However, such high metric may still mislead the conclusion of model evalution, as it does not reflect the poor quality that the model has in predicting positive samples.

###  Matthews Correlation Coefficient (MCC)

The Matthews Correlation Coefficient (MCC) has been proposed as a more robust metric for the evaluation of binary classification models, particularly in the context of imbalanced datasets [refs 10, 11]. The MCC is defined as follows:

$$
MCC=\frac{TP \times TN - FP \times FN}{\sqrt{(TP + FP)(TP + FN)(TN + FP)(TN + FN)}}
$$

This metric comprehensively incorporates all four elements of the confusion matrix, rendering it invariant to changes in label assignments. The MCC ranges from -1 to 1, with 1 denoting perfect classification devoid of false positives and false negatives, and 0 suggesting performance equivalent to random guessing. In a hypothetical case study using a threshold of 0.5, the MCC value was found to be 0.10, signifying suboptimal classification quality. This evaluation aligns more closely with the expected model performance compared to metrics like accuracy, ROC AUC, or precision and recall, particularly when the dataset skews toward positive samples. The characteristics of MCC make it a valuable tool for identifying optimized classification thresholds. By evaluating the MCC across different thresholds, one can pinpoint the threshold that maximizes the MCC value, thereby enhancing the model's overall performance. For instance, in the aforementioned example, the MCC reached its maximum value of 0.82 at a threshold of 0.2. This threshold yielded accuracy, precision, and recall values of 0.90, 0.75, and 1.00, respectively. Interestingly, even when labels were swapped, the metrics remained high, with values of 0.90 for accuracy, 1.00 for precision, and 0.83 for recall. This serves to underscore the MCC's balanced consideration of both positive and negative samples, solidifying its role as a versatile metric for a well-rounded model evaluation.
