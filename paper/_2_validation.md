# Model Validation

Model validation aims to evaluate how well a given model generalizes to an independent dataset that it has not seen during the training process. The most common methods for model validation are K-fold cross-validation (K-fold CV). To implement K-fold CV, the available dataset, denoted as \( \mathcal{D} \), is partitioned into \( K \) equally-sized folds. Mathematically, we can represent the dataset as:

\[
\mathcal{D} = \{(X, Y)\} = \{(X_1, Y_1), (X_2, Y_2), \dots, (X_K, Y_K)\}
\]

where \( X \in \mathbb{R}^{n \times p} \) represents the input features, and \( Y \in \mathbb{R}^{n \times 1} \) symbolizes the ground truth labels. \( n \) is the total number of samples, while \( p \) indicates the number of features. In each iteration of the K-fold CV, one fold is reserved as the test set, \( \mathcal{D}_{\text{test}} \) (or $\mathcal{D}_{k}$), to act as unseen data, while the remaining folds make up the training set \( \mathcal{D}_{\text{train}} \) (or \( \mathcal{D}_{-k} \)):

\[
\begin{align*}
\mathcal{D}_{\text{train}} &= \mathcal{D}_{-k} = \{(X_1, Y_1), (X_2, Y_2), \dots, (X_{k-1}, Y_{k-1}), (X_{k+1}, Y_{k+1}), \dots, (X_K, Y_K)\}\\
\mathcal{D}_{\text{test}} &= \mathcal{D}_{k} = \{(X_k, Y_k)\}
\end{align*}
\]

After splitting the dataset into $\mathcal{D}_{\text{-k}}$ and $\mathcal{D}_{\text{k}}$, the examined model $f$ is trained on the training set $\mathcal{D}_{\text{-k}}$ and denoted as $f_{\mathcal{D}_{\text{-k}}}$. The hold-out test set $\mathcal{D}_{\text{k}}$ is then used to evaluate the model performance $\hat{g}(f_{\mathcal{D}_{\text{-k}}})$, which is defined by comparing the predicted labels $\hat{Y_k} = f_{\mathcal{D}_{\text{-k}}}(X_k)$ with the true labels $Y_k$ using a performance metric $\mathcal{L}$, such as the root mean squared error (RMSE) or determination coefficient ($R^2$):

$$
\hat{g}(f_{\mathcal{D}_{\text{-k}}}) = \mathcal{L}(Y_k, \hat{Y_k}) = \mathcal{L}(Y_k, f_{\mathcal{D}_{\text{-k}}}(X_k))
$$

The K-fold CV procedure repeats the above process $K$ times until each fold has been used as the test set $\mathcal{D}_{\text{k}}$ once. Lastly, the estimated expected generalization performacne of the model $\mathbb{E}[\hat{g}(f_{\mathcal{D}})]$ leveraging the entire dataset $\mathcal{D}$ is the average of the prediction performance over all $K$ folds:

$$
\mathbb{E}[\hat{g}(f_{\mathcal{D}})] = \frac{1}{K}\sum_{k=1}^{K} \hat{g}(f_{\mathcal{D}_{\text{-k}}})
$$
