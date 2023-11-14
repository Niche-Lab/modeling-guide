## Classification

Classification models aim to predict categorical outcomes such as 'healthy' or 'sick', 'susceptible' or 'resistant', and 'high yield' or 'low yield'. This section presents a hypothetical example to highlight how the choice of performance metrics can lead to misleading model evaluations. Consider a binary classification model predicted on ten samples, of which four are positive (+) and six are negative (-), and the model produces a list of probabilities between 0 and 1, indicating the likelihood that each sample belongs to the positive class. The prediction can be expressed as the following joint probability distribution:

$$
P(X, Y) = \begin{cases}
\frac{1}{10} \times \text{Uniform}(0.8, 1.0) & \text{if } Y=1 \\
\frac{1}{10} \times \text{Uniform}(0.6, 0.8) & \text{if } Y=1  \\
\frac{5}{10} \times \text{Uniform}(0.0, 0.2) & \text{if } Y=0 \\
\frac{3}{10} \times \text{Uniform}(0.2, 0.4) & \text{if } Y=0
\end{cases}
$$

Where $X$ is a random variable representing the predicted probabilities, and $Y$ representing the ground truth labels. $\text{Uniform}(a, b)$ denotes a uniform distribution between $a$ and $b$. Table 1 shows ten samples drawn from this distribution to simulate the model predictions.

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

This example illustrates a scenario where the positive outcome is rare, which is commonly encountered in abnormality detection tasks. Before judging the prediction outcomes, a threshold that defines how the the prediction probability is converted to a binary outcome is needed. If a prediction probability is greater than the threshold, the sample is classified as positive. In this example, the threshold is set to 0.5 for simplicity, the following confusion matrix summarizes the model performance (Figure 2a). Noted that the threshold can be adjusted for specific applications and change the confusion matrix accordingly.

|  |   |Prediction| Prediction|
|--|---|-----|-----|
|  |   | (+) | (-) |
| Ground Truth | (+) | 1 | 3 |
| Ground Truth | (-) | 1 | 5 |

### Accuracy

$$
Accuracy = \frac{TP + TN}{TP + TN + FP + FN} = \frac{Total Correct Predictions}{Total Predictions}
\tag{6}
$$

Accuracy is a straightforward metric in classification problems, as defined in Equation 6. Here, TP, TN, FP, and FN represent true positive, true negative, false positive, and false negative, respectively. Based on this definition (Equation 6) and the confusion matrix, the model performance from the example is 0.60, which may appear to be better than random guesses (0.50). However, this metric can be misleading when applied to imbalanced datasets, warranting the use of additional metrics for a comprehensive evaluation. However, caution must be exercised when applying this metric to imbalanced datasets. In such cases, a model could show an accuracy that is higher than random guessing by predicting all samples as negative in an imbalanced dataset where negatives are predominant. This demonstrates that solely relying on accuracy is insufficient for evaluating a classification model, particularly when dealing with imbalanced datasets. Therefore, it is crucial to consider additional metrics for a more comprehensive and robust evaluation.


### Precision, Recall, and Precision-Recall Curve

$$
Precision = \frac{TP}{TP + FP} = \frac{TP}{Total Predicted Positive}
$$
$$
Recall (Sensitivity) = \frac{TP}{TP + FN} = \frac{TP}{Total Actual Positive}
$$

Precision evaluates the proportion of true positive predictions among all positive predictions. It effectively measures how reliable a positive prediction is. Recall, or sensitivity, measures the proportion of true positives among all actual positives. It gauges how effectively the model identifies positive samples. For instance, if the threshold is set as low as 0.1, the model is prone to making false positives, resulting in low precision. A high rate of false positives could be particularly costly in applications like <example 1>, where unnecessary treatments could be administered based on these incorrect results. Conversely, the same low threshold can yield high recall as the model is less likely to miss actual positives. In situations where failing to identify a positive instance can have severe consequences, such as <example 2> failing to detect a disease in its early stages, high recall could be more valuable.

When applying the metrics of precision and recall to the hypothetical example, the precision and recall values are 0.5 and 0.25, respectively. Both metrics yield lower values than the calculated accuracy of 0.6, underscoring the necessity of utilizing multiple metrics for a thorough evaluation of model performance. However, it is worth noting that both metrics are primarily focused on the classification of positive samples. This can introduce bias and can be particularly problematic when dealing with imbalanced datasets. As an example, if there is a scenario where the sample labels get reversed (i.e. the positives become negatives and vice versa) but the model parameters remain unchanged, the precision and recall values will shift to 0.625 and 0.833, respectively, as shown in Figure 2b. This outcome suggests that relying solely on precision and recall may not provide a complete understanding of model performance in certain contexts. Although incorporating metrics that focus on negative samples, such as specificity, can partially mitigate this issue, there remains a clear need for more robust and label-invariant metrics for an unbiased and comprehensive evaluation.

The trade-off between precision and recall is presented by adjusting the threshold. As the threshold increases, the model becomes more conservative and predicts fewer positive samples, yielding higher precision and lower recall. Since the precision and recall values reported from one single confusion matrix can only represent one specific threshold, the precision-recall (PR) curve is introduced to provide a more comprehensive view of the model performance. In the PR curve, the x-axis represents the recall values and the y-axis represents the precision values. The curve is derived by calculating the precision and recall values at different thresholds (Figure 2b). The area under the curve (AUC) is a common metric to summarize the performance of the PR curve. Still, the AUC is label-dependent, showing 0.76 for the original labels and 0.94 for the reversed labels.

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
