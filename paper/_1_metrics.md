# Performance Metrics

Performance metrics serve as quantitative indicators for evaluating model performance. They are essential tools for benchmarking various modeling approaches and validating hypotheses. However, it's important to note that the choice of metrics can significantly influence the evaluation results. Depending on the specific hypothesis being tested, an overly optimistic assessment may be concluded if inappropriate metrics are selected. This section aims to introduce commonly used performance metrics in the field of dairy science and discuss potential pitfalls that researchers should be cautious of.

## Regression

A regression model aims to predict a continuous variable and is commonly employed in various applications, such as estimating milk composition, yield, and feed efficiency, as well as assessing environmental impacts in livestock production. This section delves into three widely-used metrics for evaluating regression models: Pearson's Correlation Coefficient ($r$), Root Mean Squared Error (RMSE), and the Coefficient of Determination ($R^2$).

In the hypothetical example depicted in Figure 1, 100 observations are generated from two separate normal distributions. The first 50 observations are drawn from a normal distribution with a mean of -3 and a standard deviation of 1, denoted as \( \mathcal{N}(-3, 1) \). The remaining 50 observations are generated from another normal distribution, \( \mathcal{N}(3, 1) \). Utilizing two distinct distributions serves to simulate experimental block effects, preset at a magnitude of 6 units for this experiment.

Based on these simulated observations, four sets of predictions are generated:

1. **First Prediction**: Each observation is multiplied by 0.3 to establish a correlation, followed by the addition of random noise \( \mathcal{N}(0, 0.7) \) to introduce prediction errors.

2. **Second Prediction**: Values from the first prediction are multiplied by 5, simulating predictions with a larger variance while maintaining the same relative ordering with the original predictions.

3. **Third Prediction**: Each value from the first prediction is raised to the power of 5. This transformation serves to compress values within the range of -1 to 1 towards zero, without affecting their relative order. Additionally, values greater than 1 or less than -1 are pushed farther from zero, simulating outlier predictions.

4. **Fourth Prediction**: Values sampled from \( \mathcal{N}(-3, 1) \) and \( \mathcal{N}(3, 1) \) are added to the first and second blocks of observations, respectively. This amplifies the block effects, simulating a model that effectively distinguishes between different blocks but is less capable of predicting individual variations within each block.

This quartet of predictions serves to simulate potential challenges and complexities encountered in real-world modeling scenarios, thereby providing a foundation for evaluating different performance metrics used in regression problems.

### Pearson Correlation Coefficient (r)

$$
r = \frac{\sum_{i=1}^{n}(x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum_{i=1}^{n}(x_i - \bar{x})^2}\sqrt{\sum_{i=1}^{n}(y_i - \bar{y})^2}}
= \frac{cov(x, y)}{\sigma_x \sigma_y}
\tag{1}
$$

The Pearson Correlation Coefficient (\(r\)) is a widely used metric for assessing the linear relationship between two variables, as defined by Equation \(1\). In this equation, the observed values are denoted by \(x\) and the predicted values are represented by \(y\). To calculate the correlation coefficient, the equation first computes the covariance between \(x\) and \(y\), which captures how the coordinates of data points deviate from their means. This value is then normalized by the product of the standard deviations of \(x\) and \(y\). The coefficient can be either positive or negative, reflecting a positive or negative correlation between the two variables, respectively. Owing to the normalization term in the denominator, the coefficient is scale-invariant and will always fall within the range of -1 to 1. This attribute is illustrated in the first two scenarios: despite the second scenario having a variance five times larger than the first, the coefficient remains consistent.

However, the metric has its limitations. It is sensitive to the presence of outliers, as shown in the third scenario where most data points cluster near zero but a few outliers yield a high correlation coefficient. Similarly, the metric is influenced by block effects, leading to inflated correlation values, as observed in the fourth scenario. When calculated within individual blocks, the coefficients drop to 0.11 and 0.06, respectively. This emphasizes the importance of either visually inspecting regression results through scatter plots or examining them within individual blocks. If the objective is to examine the model predictability of individual variation rather than block effects, caution should be exercised when applying this metric, especially in scenarios where block effects are evident.

In practice, this coefficient is often utilized to evaluate a model capability to correctly order or rank individual observations based on a specific trait of interest. For instance, <example 1>, <example 2>, among others [ref 3-6].

### Root Mean Squared Error (RMSE)

$$
RMSE=\sqrt{\frac{\sum_{i=1}^{n}(y_i - \hat{y_i})^2}{n}}
\tag{2}
$$

The Root Mean Squared Error (RMSE) serves as a quantitative measure to gauge the average magnitude of prediction errors between observed values (\(y\)) and their predicted values (\(\hat{y}\)). It gives the error in the same units as the observation (\(y\)), and a lower RMSE value indicates a better model performance. Defined by Equation \(2\), \(n\) stands for the number of observations. Distinct from the correlation coefficient, RMSE is sensitive to scale, implying that achieving predictions with a variance akin to the observed values takes precedence over maintaining their order or trend. This is particularly pertinent when the focus is on the absolute magnitude of the error. Take for instance Scenario 2, where the predictions have been scaled by a factor of 5 compared to Scenario 1. The RMSE inflates from 2.41 to 3.63, underscoring that even if both scenarios rank the observations identically, RMSE effectively captures the discrepancies in the absolute errors. Another notable characteristic of RMSE is its sensitivity to outliers. In Prediction 3, where certain predictions deviate substantially from the majority, the squaring operation within RMSE accentuates these outliers, culminating in a RMSE value of 25.56. It's also worth mentioning that RMSE is impervious to block effects, unlike the correlation coefficient. In Prediction 4, both the complete set of predictions and the intra-block predictions yield comparable RMSE values—1.49, 1.46, and 1.52, respectively.

Due to its emphasis on absolute error, RMSE is frequently employed in contexts where the magnitude of the error is a critical consideration. For example, <example 1> and <example 2>.

### Coefficient of Determination ($R^2$)

$$
R^2 = 1 - \frac{\sum_{i=1}^{n}(y_i - \hat{y_i})^2}{\sum_{i=1}^{n}(y_i - \bar{y})^2}
= 1 - \frac{SSE}{SST}
\tag{3}
$$

The Coefficient of Determination, commonly known as \( R^2 \), provides a similar measurement of the SSE as RMSE, but it is normalized by the total sum of squares (SST) to yield a more versatile metric for comparing results across different experiments. As defined in Equation \(3\), \( n \) represents the number of observations, \( y \) signifies the actual observed values, \( \hat{y} \) indicates the predicted values, and \( \bar{y} \) is the mean of the observed values. A higher \( R^2 \) value suggests a lower prediction error. Importantly, if \( R^2 \) falls below zero, the predictions are deemed inferior to a naive approach that predicts the mean of the observed values for all samples, particularly in terms of absolute prediction errors. Similar to RMSE, \( R^2 \) is influenced by the variance of the predictions. Specifically, when predictions exhibit a high degree of variance that deviates from the actual observations, the \( R^2 \) value can be adversely impacted, dropping from 0.47 in Prediction 1 to -0.27 in Prediction 2, for example. Additionally, due to the presence of squared terms in its calculation, \( R^2 \) is sensitive to outliers, as evidenced in Prediction 3.

$$
SST = SSR + SSE
\tag{4}
$$

$$
\sum_{i=1}^{n}(y_i - \bar{y})^2 = \sum_{i=1}^{n}(\hat{y_i} - \bar{y})^2 + \sum_{i=1}^{n}(y_i - \hat{y_i})^2
\tag{5}
$$

The fourth prediction scenario offers an intriguing case study for \( R^2 \) to identify misleading model performance due to block effects. According to Equations \(4\) and \(5\), the SST can be decomposed into the Regression Sum of Squares (SSR), which represents the variation explained by the model, and the SSE, which accounts for the unexplanable errors as discussed in the RMSE section. From this standpoint, \( R^2 \) quantifies the proportion of variation that the model is able to explain. In the case of the fourth prediction, the model effectively captures the differences between various blocks, which contributes substantially to the SST. Consequently, the \( R^2 \) value rises to as high as 0.80. However, this elevated metric can be misleading. The hypothetical model in the fourth prediction does not possess the ability to capture individual variations within each block. The strength of \( R^2 \) lies in its capability to differentiate between sources of variation. Upon closer inspection of the data within each block, the \( R^2 \) values plummet to -0.71 and -1.10, respectively, highlighting that the model fails to account for intra-block variability. In summary, while both RMSE and \( R^2 \) aim to measure prediction errors, \( R^2 \) offers additional statistical insights that facilitate a more nuanced evaluation of model performance.

Current practices in dairy science usually report both R2 and RMSE, and sometimes r correlation. as exemplified in <example 1>, <example 2>.

## Classification

Classification models aim to predict categorical outcomes such as 'healthy' or 'sick', 'susceptible' or 'resistant', and 'high yield' or 'low yield'. This section presents a hypothetical example to highlight how the choice of performance metrics can lead to misleading model evaluations. Consider a binary classification model trained on ten samples—four positives (+) and six negatives (-). The model produces a list of probabilities between 0 and 1, indicating the likelihood that each sample belongs to the positive class (Table 2).

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


This example aims to simulate a scenario where the positive outcome is rare, which is commonly encountered in abnormality detection tasks. If a threshold of 0.5 is used to classify samples as positive, the following confusion matrix summarizes the model performance (Figure 2). The model has only a 25% chance of correctly identifying positive samples, which is worse than a random guess with a 50% accuracy rate. This highlights the importance of choosing appropriate performance metrics for evaluation.

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

Accuracy is a straightforward metric in classification problems, as defined in Equation 6. Here, TP, TN, FP, and FN represent true positive, true negative, false positive, and false negative, respectively. Based on this definition (Equation 6), the model performance from the example is 60%, which may appear to be better than random guesses (50%). However, this metric can be misleading when applied to imbalanced datasets, warranting the use of additional metrics for a comprehensive evaluation. However, caution must be exercised when applying this metric to imbalanced datasets. In such cases, a model could achieve a misleadingly high accuracy by simply predicting the majority class for all samples. For example, by predicting all samples as negative in an imbalanced dataset where negatives are predominant, the model could still achieve high accuracy. This demonstrates that solely relying on accuracy is insufficient for evaluating a classification model, particularly when dealing with imbalanced datasets. Therefore, it is crucial to consider additional metrics for a more comprehensive and robust evaluation.


### Precision, Recall, and Precision-Curve

$$
Precision = \frac{TP}{TP + FP} = \frac{TP}{Total Predicted Positive}
$$
$$
Recall (Sensitivity) = \frac{TP}{TP + FN} = \frac{TP}{Total Actual Positive}
$$

Precision evaluates the proportion of true positive predictions among all positive predictions. It effectively measures how reliable a positive prediction is. Recall, or sensitivity, measures the proportion of true positives among all actual positives. It gauges how effectively the model identifies positive samples. For instance, if the threshold is set as low as 0.1, the model is prone to making false positives, resulting in low precision. A high rate of false positives could be particularly costly in applications like <example 1>, where unnecessary treatments could be administered based on these incorrect results. Conversely, a low threshold can yield high recall as the model is less likely to miss actual positives. In situations where failing to identify a positive instance can have severe consequences, such as <example 2> failing to detect a disease in its early stages, high recall could be more valuable.

The trade-off between precision and recall becomes critical here. Increasing precision generally decreases recall and vice versa. This is known as the precision-recall trade-off (<ref>). Therefore, depending on the specific application and the associated costs of false positives and false negatives, one may choose to prioritize one metric over the other.

When the metrics of precision and recall are applied to the presented hypothetical example, the precision is 0.5 and the recall is 0.25. Both metrics are lower than the metric accuracy of 0.6, highlighting the importance of reporting multiple metrics to comprehensively evaluate the model performance. However, the pitfall still exists. Both the metrics focus on the positive samples, the bias towards the positive samples can be misleading, especially when the imbalanced dataset is evaluated. For example, without changing any model parameters but to swap the sample labels, that is, the positive samples are now negative and vice versa, an overoptimistic evaluation can be concluded (Figure 2b). The new precision and recall are 0.625 and 0.833, respectively. This result is clearly misleading. Although this pitfall can be avoided by reporting metrics of negative examples, such as specificity, a robust and label-invariant metric is still needed for a comparable evaluation.



### Receiver Operating Characteristic (ROC) Curve

The Receiver Operating Characteristic (ROC) curve is a common way to 


###  Matthews Correlation Coefficient (MCC)

The downside of MCC is that it cannot be intepreted directly to the 

## Object Detection and Segmentation

### Intersection over Union (IoU)

### Mean Average Precision (mAP)

### Other classification metrics



