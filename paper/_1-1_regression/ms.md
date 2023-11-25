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

The Pearson Correlation Coefficient r measures the strength of the linear relationship between two variables, as defined by Equation (1). In this equation, the observations are denoted by x and the predicted values are represented by y. To calculate the correlation coefficient, the equation first computes the covariance between $x$ and $y$, which captures how coordinates of data points deviate from their means. The covariance is then normalized by the product of the standard deviations of $x$ and $y$, hence the coefficient $r$ is scale-invariant and will always fall within the range of -1 to 1.

A suitable case to use this metric is when the objective is to rank the obervations of interest rather than to predict absolute magnitude of the error. This property was illustrated in Scenario 1 and 2, where the coefficient remains consistent despite the second scenario having errors five times larger than the first. If the absolute error is of interest, this metric should be used along with other metrics such as RMSE or $R^2$. It is also worth noting that this metric can yield a value of 0.27 in Scenario 3 where 90% of the predictions were failed to capture the trend and clustered as zero. The positive performance mainly came from the fact that the predictions still ranked the remaining 10% of the observations in a fairly correct order, regardless of the large magnitude of the error. Another common pitfall of this metric is that it can be influenced by block effects, leading to inflated estimated performance if the individual variation is more of interest. This is illustrated in Scenario 4, where the overall coefficient $r$ is 0.94 but the metric within each block drops to 0.33 and 0.25, respectively. This emphasizes the importance of either visually inspecting regression results through scatter plots or examining them within individual blocks.
In practice, this metric is often used to evaluate a model that can identify high-performing individuals. De Sousa et al. used this metric to show the capability of identifying high-producing dairy cows based on the predicted nutrient digestibility [ref 1]. This metric was also applied in a model selecdtion scenario, where the multiple models were evaluated based on their ability to rank the traits of interest, such as feed intake [ref 2] and milk composition [ref 3] in dairy cows.


### Root Mean Squared Error (RMSE)

$$
RMSE=\sqrt{\frac{\sum_{i=1}^{n}(y_i - \hat{y_i})^2}{n}}
\tag{2}
$$

The Root Mean Squared Error (RMSE) serves as a quantitative measure to gauge the average magnitude of prediction errors between observed values (\(y\)) and their predicted values (\(\hat{y}\)). It gives the error in the same units as the observation (\(y\)), which is particularly useful when the absolute error is of interest. As shown in Equation 2, the RMSE is calculated by first computing the squared error, followed by the summation of all squared errors, and finally averaging the sum by the number of observations $n$.
Distinct from the correlation coefficient, RMSE is sensitive to scale, implying that achieving predictions with a variance akin to the observed values takes precedence over maintaining their order or trend. This property is evident in Scenario 2, where the RMSE inflates from 2.41 to 3.63, despite the fact that the predictions in both scenarios rank the observations identically. Another notable characteristic of RMSE is it weights more on large errors, which is convenient when making a large error is costly and should be avoided. In Scenario 3, where certain predictions deviate substantially from the majority, the squaring operation in Equation 2 accentuates these outliers, culminating in a RMSE value of 25.49. It is also worth mentioning that RMSE is impervious to block effects, which was illustrated in Scenario 4. In this scenario, both the complete set of predictions and the intra-block predictions yield similar RMSE values—1.49, 1.46, and 1.52, respectively. This phenomenon emphasizes again that RMSE is affected solely by the magnitude of the error, which neglects the ability of the model to capture relative trends in either intra-block or inter-block predictions.

In practice, RMSE is easy to interpret by real-world productive units. For example, monitoring cow body weight is a common practice to aid in the management of dairy cows. Studies by Song et al. and Xavier et al. have utilized RMSE to assess the effectiveness of three-dimensional cameras in estimating dairy cow body weight, yielding RMSE values of 41.2 kg and 12.1 kg, respectively (Song et al., 2018; Xavier et al., 2022). These figures provide a straightforward way for farmers to gauge whether the prediction error is tolerable, considering their specific operational costs and management thresholds. In essence, RMSE translates complex model accuracy into practical insights for productive agricultural units.


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
