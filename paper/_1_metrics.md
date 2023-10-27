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

Classification models are used to predict categorical outcomes, such as healthy or sick, susceptible or resistant, and high or low yield. This section provides another hypothetical example illustrating a deceived model evaluation due to the choice of performance metrics. The example is based on a binary classification problem, where there are two possible outcomes: positive (+) and negative (-) in 100 samples. The ground truth is that 20 samples are positive and 80 samples are negative. The imbalanced distribution intent to simulate a sceneraio where the positive outcome of interest is rare, which is commonly encountered in abnormality detection problem. The prediction outcome is illustrated in a pair of confusion matrices (Figure 2). The confusion matrix is a 2x2 table that summarizes the prediction results. The rows represent the ground truth, and the columns represent the prediction. TP, FP, FN, and TN stand for true positive, false positive, false negative, and true negative, respectively. Clearly, the prediction is not ideal, as the model only have 25% of chance to correctly predict the positive samples, which is worse than a random guess that has 50% correct rate. This example intends to demonstrate that the choice of performance metrics can significantly influence the evaluation results.


|  |   |Prediction| Prediction|
|--|---|-----|-----|
|  |   | (+) | (-) |
| Ground Truth | (+) | 5 | 15 |
| Ground Truth | (-) | 5 | 75 |

|  |   |Prediction| Prediction|
|--|---|-----|-----|
|  |   | (+) | (-) |
| Ground Truth | (+) | 75 | 5 |
| Ground Truth | (-) | 15 | 5 |


### Accuracy

$$
Accuracy = \frac{TP + TN}{TP + TN + FP + FN}
$$

Accuracy is the most straightforward metrics in classifcaiton problem. It counts the ratio of correct predictions over the total number of predictions. Based on the definition, the accuracy computed from the confusion matrix (figure 2a) should be 80% and might be considered as a good performance. However, caution must be exercised when applying this metric to imbalanced datasets. By simply predicting all samples as negative, the model can also achieve the same level of accuracy. Solely reporting accuracy as the metric in evaluating a classification model is not sufficient. Hence, other metrics are needed to provide a more comprehensive evaluation.

### Precision, Recall, and F1 Score

$$
Precision = \frac{TP}{TP + FP}
$$
$$
Recall = \frac{TP}{TP + FN}
$$
$$
F1 = 2 \times \frac{Precision \times Recall}{Precision + Recall}
$$

Precision and recall provide another view of classification performance. Precision 




Two matrices represent the same classification outcomes, but the second table Table 1b swap the positive and negative labels. 


### Sensitivity and Specificity

###  Matthews Correlation Coefficient (MCC)

The downside of MCC is that it cannot be intepreted directly to the 

## Object Detection and Segmentation

### Intersection over Union (IoU)

### Mean Average Precision (mAP)
