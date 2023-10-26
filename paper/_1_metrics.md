# Performance Metrics

Performance metrics serve as quantitative indicators for evaluating model performance. They are essential tools for benchmarking various modeling approaches and validating hypotheses. However, it's important to note that the choice of metrics can significantly influence the evaluation results. Depending on the specific hypothesis being tested, an overly optimistic assessment may be concluded if inappropriate metrics are selected. This section aims to introduce commonly used performance metrics in the field of dairy science and discuss potential pitfalls that researchers should be cautious of.

## Regression

A regression model aims to predict a continuous variable and is commonly employed in various applications, such as estimating milk composition, yield, and feed efficiency, as well as assessing environmental impacts in livestock production. This section delves into three widely-used metrics for evaluating regression models: Root Mean Squared Error (RMSE), Pearson's Correlation Coefficient ($r$), and the Coefficient of Determination ($R^2$).

In the hypothetical example depicted in Figure 1, 100 observations are generated from two separate normal distributions. The first 50 observations are drawn from a normal distribution with a mean of -3 and a standard deviation of 1, denoted as \( \mathcal{N}(-3, 1) \). The remaining 50 observations are generated from another normal distribution, \( \mathcal{N}(3, 1) \). Utilizing two distinct distributions serves to simulate block effects, preset at a magnitude of 6 units for this experiment.

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

It is a common metric used to describe the linear relationship between two variables. In equation (1), it first measure the covariance of x and y, which indicate how coordinate of the data points vary from their mean, and then normalize it by the product of their standard deviations. Because of the normalization step, this metric will not be affected by the scale of the data and will always lie in the range from -1 to 1. This property is revealed in the first and second predictions; Although the second prediction has five times larger variance than the first prediction, the correlation coefficient remain the same. However, this metric is sensitive to the presence of outliers, as shown in the third prediction where most predictions are compressed towards zero, but the presence of outliers still result in a high correlation coefficient. Lastly, in the fourth predcition, this metric is also sensitive to block effects, resulting an over-esimated correlation. However, when the metric is calculated in each individual block, the coefficients are dropped to 0.11 and 0.06, respectively. This highlights the importance of examining the regression prediction result in a scatter plot, or in each individual block. If the exmained hypothesis does not intend to predict the block effect but rather the individual variation, then this metric should be used with caution in the fourth prediction where the block effect is presence.


hypothetical examp 


## Classification

### Accuracy

### Precision and Recall


### Sensitivity and Specificity

### F1 Score and Matthews Correlation Coefficient (MCC)

## Object Detection and Segmentation

### Intersection over Union (IoU)

### Mean Average Precision (mAP)
