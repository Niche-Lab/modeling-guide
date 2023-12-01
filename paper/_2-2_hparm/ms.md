# Model Selection

## Hyperparameter tuning and feature selection

Model selection becomes necessary when models are not entirely determined by the data alone. For example, in a regularized linear regression model such as ridge regression (Hoerl and Kennard, 1970) or the least absolute shrinkage and selection operator (LASSO) (Tibshirani, 1996),  it is essential to define a regularization parameter, λ, before fitting the model to the data. A larger λ value yields a more regularized model, which tends to reduce smaller coefficients to negligible values or zero. This approach helps in preventing overfitting to noise in the training data. The loss functions for unregularized ordinary least squares (OLS), ridge regression, and LASSO regression are given as follows:

$$
\begin{aligned}
\mathcal{L}_{OLS}(\beta) &= \sum_{i=1}^n (y_i - x_i\beta)^2\\
\mathcal{L}_{ridge}(\beta) &= \sum_{i=1}^n (y_i - x_i\beta)^2 + \lambda \|\beta\|_2^2\\
\mathcal{L}_{LASSO}(\beta) &= \sum_{i=1}^n (y_i - x_i\beta)^2 + \lambda \|\beta\|_1
\end{aligned}
$$

Where $x_i$ and $y_i$ represent the $i^{th}$ row of the design matrix $X$ and the response vector $y$, respectively. The term $n$ denotes the sample size, and $\beta$ is the coefficient vector. All three models aim to find the optimal $\beta$ that minimizes their respective loss function, $\mathcal{L}$. In the regularized models (i.e., ridge and LASSO regression), the length of $\beta$ is penalized in the loss function.

These pre-defined parameters, which influence model fitting and remain constant during the training process, are known as hyperparameters. Beyond regularized models, hyperparameters are crucial in other predictive models, enhancing flexibility and robustness. For example, in support vector regression (SVR) (Drucker et al., 1996), the regressors ($X$) are projected onto a linear subspace to approximate the target variable $y$. By choosing a suitable kernel function, which transforms the regressors into a non-linear space, as a hyperparameter, SVR can more effectively capture non-linear relationships, thus significantly improving model performance. Another hyperparameter example is the number of latent variables in partial least square regression (PLSR) (Abdi 2003), which condenses the original regressors into a more manageable set of latent variables, reducing multicollinearity issues. A smaller number of latent variables might lose significant information from the original regressors, while too many can lead to overfitting. Similarly, in random forests (Breiman, 2001), hyperparameters like tree depth and the number of trees dictate model complexity. The same applies to the number of hidden layers and the size of filters in convolutional neural networks (CNNs) (LeCun et al., 1998). All these examples highlight the fact that selecting the most suitable hyperparameters, a process known as model selection or hyperparameter tuning  (Himeldorf and Wahba), is crucial for optimizing model performance.

Feature selection is another crucial aspect of model selection. This process involves fitting the model to a selected subset of the original features, particularly essential in high-dimensional data scenarios where the number of features exceeds the number of observations, leading to poor model generalization. For instance, Ghaffari et al. (2019) sought to predict health traits in 38 multiparous Holstein cows using metabolite profiling strategy. Out of 170 metabolites, only 12 were identified as effective discriminators between healthy and overconditioned cows and were thus selected for the predictive model. Therefore, optimizing feature subsets is a vital model selection strategy that significantly affects model performance.

Including model selection process within the cross validation is crucial to avoid common pitfalls. The risk of inflated model performance arises when model selection is guided by results on the test dataset. Even if the chosen model is subjected to k-fold cross-validation afterward, its selection bias toward the test set can lead to an overestimation of its efficacy. This issue has been highlighted in numerous literatures  [list literature references], and practitioners must be vigilant to avoid this pitfall. A practical solution is to divide the dataset into training, validation, and test sets. The validation set is then used for model selection, ensuring the test set remains completely unused during the training phase, thereby providing a more accurate measure of model performance. For instance, the study by Rovere et al. exemplifies best practices in hyperparameter tuning and feature selection by employing an independent cross-validation step prior to assessing model performance. This approach enabled the precise selection of relevant spectral bands from the mid-infrared spectrum and the optimal number of latent dimensions in partial least squares with Bayesian regression for predicting the fatty acid profile in milk (Rovere et al. 2021). Similarly, Becker et al. (2021) demonstrated a robust evaluation by using two nested cross-validation loops; the inner loop conducted a grid search for the best hyperparameters in logistic regression, while the outer loop was designed to evaluate the performance of the resulting optimized model. Both examples underscore the importance of separating model selection from performance evaluation to ensure validity and reliability of the results.

## Simulation Objectives and Hypothesis

The objective of this simulation study is to examine the effect of improper model selection implementation on validation bias. The focus will be on the model selection procedures of feature selection and hyperparameter tuning. The study hypothesizes that utilizing the test set inappropriately during any stage of model selection will lead to a significant overestimation of model performance.

## Simulation Design

This study simulated a regression task using a Support Vector Regressor (SVR) model, 

Your simulation design description is detailed, but it can be condensed for brevity and clarity without losing critical information:

---

## Simulation Design

This study simulated a regression task using a Support Vector Regressor (SVR) model, which utilized various kernel functions to project a subset of features, $X$, to predict a target variable, $y$. Both $X$ and $y$ are drawn from a normal distribution to establish a baseline null correlation (performance $r=0$) for assessing validation bias. This study set the sample size and number of features at 100 and 1000, respectively.

Feature selection is executed by choosing the top 50 features that correlate most strongly with $y$. For hyperparameter tuning, four kernel functions were evaluated: linear, polynomial, radial basis function (RBF), and sigmoid.

This study introduces notations $FS$ for feature selection and $HT$ for hyperparameter tuning, assigning a binary indicator (0 or 1) to denote incorrect (0) or correct (1) implementation of model selection. This yields four possible combinations of model selection strategies: $FS=0; HT=0$, $FS=0; HT=1$, $FS=1; HT=0$, and $FS=1; HT=1$ (Figure 7).

When $FS=0$, feature selection precedes cross-validation splitting. If $FS=1$, feature selection occurs within each fold of the training set during cross-validation. With hyperparameter tuning ($HT$), a correct implementation ($HT=1$) involves splitting the dataset into training (64%), validation (16%), and test (20%) sets. The model is trained and tuned using the training and validation sets, respectively, while the test set is reserved for a single evaluation of model performance. Conversely, with $HT=0$, only training (80%) and test (20%) sets are used, risking validation bias as the test set informs both training and performance reporting. A 5-fold cross-validation approach was deployed for all scenarios.

Validation bias is measured as the discrepancy between the model selection-influenced performance estimate and the expected generalization performance ($r=0$), using the Pearson correlation coefficient between predicted and observed values. Over 1000 iterations, the study assess the distribution of validation bias. A t-test will determine whether the validation bias significantly deviates from zero.

## Results

The validation bias were visuzlied using box plots, with the feature selection factor $FS$ on the x-axis and hyperparameter tuning $HT$ distinguished by color — green for incorrect and yellow for correct implementation. The y-axis represents the validation bias as measured by the correlation coefficient, with the expected generalization performance marked by a horizontal line at $r=0$.

The results indicate a clear overestimation of model performance when feature selection is applied to the entire dataset, regardless of hyperparameter tuning. The median biases were 0.797 for $FS=0; HT=0$ and 0.761 for $FS=0; HT=1$. Moreover, inappropriate validation in hyperparameter tuning resulted in a significant bias (p-value < 0.001) with a median of 0.113 for $FS=1; HT=0$. The only scenario without bias significantly occurred when both feature selection and hyperparameter tuning were correctly incorporated within the cross-validation process $FS=1; HT=1$, yielding a median bias of -0.008. These findings align with the initial hypothesis and the prevailing literature, reinforcing that model selection must be integrated into the cross-validation workflow to prevent an overestimation of model performance.

## Section Conclusion

The simulation results robustly confirm the hypothesis that improper implementation of model selection inflates performance estimates. Specifically, the validation bias is markedly high when feature selection precedes data splitting, with or without correct hyperparameter tuning. Although integrating feature selection within cross-validation folds mitigates this bias, incorrect hyperparameter validation still significantly skews performance metrics. Notably, this overestimation is even more pronounced in complex models, such as neural network architectures that often entail over a million parameters. These findings underscore the necessity of meticulous cross-validation practices, particularly for feature selection and hyperparameter tuning, to ensure accurate performance estimations and generalizability in predictive modeling.
