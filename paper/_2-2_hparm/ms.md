# Model Selection

## Hyperparameter tuning and feature selection

Model selection is required when the model are not solely determined by the data. For example, a regularized linear regression model, such as ridge regression or least absolute shrinkage and selection operator (LASSO), has to define a regularization term λ before fitting the model to the given data. A larger λ will result in a more regularized model, which tends to shrink small coefficients to zero to avoid overfitting to noise in the training data. Below are the loss functions between unregularized OLS and regularized ridge regression and LASSO regression:

$$
\begin{aligned}
\mathcal{L}_{OLS}(\beta) &= \sum_{i=1}^n (y_i - x_i\beta)^2\\
\mathcal{L}_{ridge}(\beta) &= \sum_{i=1}^n (y_i - x_i\beta)^2 + \lambda \|\beta\|_2^2\\
\mathcal{L}_{LASSO}(\beta) &= \sum_{i=1}^n (y_i - x_i\beta)^2 + \lambda \|\beta\|_1
\end{aligned}
$$

Where $x_i$ and $y_i$ are the $i^{th}$ row of the design matrix $X$ and the response vector $y$, respectively. And $n$ is the sample size and $\beta$ is the coefficient vector. As all three model aims to find the optimal β that minimizes the loss function $\mathcal{L}$, the length of $\beta$ in the regularized models (i.e., ridge and LASSO regression) is penalized in the loss function.

These parameters that defined how the model is fitted and are not changed during the training process are called hyperparameters. In addition to the regularized models, hyperparameters are widely used in other prediction models for better flexibility and robustness. For example, a support vector regression (SVR)projects the regressors ($X$) onto a linear subspace to approximate the target variable $y$. However, the incorporation of a kernel function permits the SVR to further explain non-linear relationships within the data. Picking an appropriate hyperparameter, kernel function, to best fit the data can enhance the model performance in non-linear data. Another hyperparameter example is the size of the latent variables in partial least square regression (PLSR), which compresses the original regressors into a smaller set of latent variables to avoid multicollinearity problems. A lower number of latent variables will lose more information from the original regressors, while a higher number of latent variables will result in overfitting. Selecting the optimal value for these hyperparameters is known as model selection (Himeldorf and Wahba) or hyperparameter tuning.

In addition to hyperparameters, feature selection is another type of model selection in which the model is fitted to a subset of the original regressors. This procedure is commonly required in dealing with high-dimensional data, where the number of features or regressors are much larger than the number of observations and resulting poor generalization performance. For example <list spectral study and GWAS>.

When implementing the model selection, a common pitfall is to exclude the selection from the model validation process. For example, when studying production traits using hyperspectral devices where hundreds of spectral bands are available, determining the ideal subset of the bands and model hyperparameters are the essential step before start training the model. The risk of overestimating the model performance emerges when the optimal spectral bands are selected based on the performance on the test set. Even the selected model will undergo a k-fold cross validation, the model has been selected in favor of the test set therefore overestimate the model performance. This validation mistake has been discussed in many literature <list literature> and should be carefully avoided in practice. A workaround is to further split the training/test sets to training/validation/test sets, where the validation set is used to select the optimal model and leave the test set untouched throughout the training process.


## Simulation Objectives and Hypothesis

This simulation study is to investigate the impact of falsely impelemting model selection on the validation bias. The examined model selection procedures includes feature selections and hyperparameter tuning. The hypothesis is that the model performance will be significantly overestimated when the test set is incorrectly used in either of the model selection procedures.

## Simulation Design

A regression task is simulated in this study. Support Vector Regressor is used as the model, which applies a kernel function to project a subset of features $X$ to predict the label $y$. The features $X$ and label $y$ variables are sampled from a normal distribution, which provides a baseline correlation performance $r=0$ for estimating the validation bias. The sample and feature sizes are set to 100 and 1000, respectively. The feature selection process is conducted by selection the top 50 features with the highest correlation with the label $y$. In tuning the hyperparameters, the kernel functions: linear, polynomial, radial basis function (RBF), and sigmoid functions, were evaluated in the tuning process.

Notations $FS$ and $HT$ were used to denote feature selection and hyperparameter tuning, respectively. And a binary value 0 or 1 to indicate whether the model selection is implemented correctly, where 1 indicates a correct implementation. With this setting, there are a total of four different model selection combinations: 1. $FS=0;HT=0$, 2. $FS=0;HT=1$, 3. $FS=1;HT=0$, and 4. $FS=1;HT=1$. When $FS=0$, the features are selected before the data is split for the cross validation, otherwise the features are selected within each cross validation fold using only the training set. For the hyperparameter turning, the dataset will be split to training, validation, and test sets when $HT=1$. The model will be trained on the training set and evaluated the performance on the validation set for each hyperparameter. The test set will only be used once in reporting the estimated performance of the selected model. On the other hand, when $HT=0$, the data will only be split into training and test set. The model will be trained on the training set and evaluated the performance on the test set for each hyperparameter. The test set will be used multiple times, and only the best performance among all hyperparameters will be reported. The number of folds is set to 5. For example, when $HT=1$, the dataset will be first allocated 80% to training set, and 20% to the test set. Then, the training set will be further split into five folds, of which four folds of them are training set (64% of the entire dataset) and one fold is used as the validation set (16% of the entire dataset).

The validation bias is estimated by the metric difference between the esimtated performance given the model selection and the expected generalization performance (i.e., r=0). The metric used in this study is the Pearson correlation coefficient between the predicted and observed values. If one data sampling is considered as one iteration, 1000 iterations are conducted in this simulation to examine the distribution of the validation bias. A t-test is used to examine if the mean of the validation bias is significantly different from zero.


## Result

The bias were visualized in box plots, where the factor $FS$ is presented on the x-axis, and $HT$ is colored in green and yellow to indicate the result of incocrectly and correctly implement the model selection, respectively. The y-axis is the validation bias by correlatino coefficent, with a horizontal line r=0 to indicate the expected generalization performance. An obvious overestimated performance is shown when the feaeture selection is carried out for the entire dataset with or without proper hyperparameter tuning. The median bias are 0.797 and 0.761 for $FS=0;HT=0$ and $FS=0;HT=1$, respectively. In addition, even when implementing the feature selection within each cross validation fold, falsely validate the hyperpamater will result a significant bias (p-value < 0.001) with a median of 0.113 ($FS=1; HT=0$). The only unbiased esimation is to include both feature selection and hyperparameter tuning in the cross validation process ($FS=1; HT=1$), where the median bias is -0.008. The result is consistent with the hypothesis and literature that the model selection should be included in the cross validation process to avoid overestimating the model performance.


## Suggestion

Use cross validation wrap-up function to do the cross validation in the inner loop.

The accuracy of the kernel machine on test data is critically dependent on the choice of good values for the hyper-parameters, in this case λ and 
