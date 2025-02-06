# Appealing letter

First, I want to show my gratitude for the invitation to submit a review paper as a guidance for the JDS audiences. Although it was rejected, I would like to share my thoughts on the current state of the machine learning (ML) papers in the JDS community. And I am writing this letter to appeal the decision.

## The concerns

Any scientific paper should be reproducible and comparable by future studies. Without meeting these criteria, the study is simply a showcase of an analytical pipeline. Although the JDS community has been witnessing a surge in the number of ML papers, of which some are missing details to fulfill these requirements. Additionally, with the emerging artificial intelligence tools (e.g., ChatGPT), anyone can generate running codes to conduct a ML procedure without any understanding of the underlying model. The JDS community is facing a new challenge in the research integrity, as verifying these AI-generated works takes additional effort for the reviewers and the editors.

In general, ML papers refer to any studies that use modeling approaches other than linear regression, such as random forest, support vector machine, and neural network. These ML models are double-edged swords. In contrast to the linear regression, they are more powerful in capturing the non-linear relationship between the explanatory and response variables. However, given the model complexity, the ML models are also prone to capture the noise in the data and lead to overfitting. It has never been an issue when most JDS papers use ANOVA table or other statistical tests to verify the modeling result. However, since the ML models are often non-linear and non-parametric, the validation approach used in linear regression is no longer applicable to these models.

## The solution

In my humble opinion, those concerns can be resolved if the future ML papers can propoerly define (1) performance metrics and (2) model validation. Performance metric provides an objective way to evaluate the present work and compare it with other studies. But a confusion arises when the metric is not well-defined. For example, solely reporting "a model performance is R^2=0.75" without defining the R^2 is not informative. "R^2" alone could have divergent interpretations between Pearson correlation and coefficient of determination. Another important aspect of a ML paper is model validation. It is a procedure that hides a portion of the data to simulate an unseen future data and evaluate the model reproducibility. Failure to validate the model may lead to an overestimation of the study result and make it uncomparable with other studies. In addition, the portion of the hidden data can affect the reported performance metrics. For example, does "R^2=0.75" mean equally good when the model is validated with 10% of the data compared to 20% of the data? These details must be defined in the paper to secure the study reproducibility.

Although these concerns can be verified by searching statistical literatures, which are scattered in different journals and books, there is no single paper that specifically focuses on avoiding these common pitfalls in the ML papers. Hences, my vision for the review paper is to provide a one-stop guideline for the JDS audiences who are interested in conducting ML studies or reviewing ML papers. This review paper provides a series of simulation experiments to demonstrate how to properly validate a ML model along with practical examples cited from the JDS papers.
To provide more concrete examples, I have conducted a survey on six ML papers from the JDS which were published in the last five years. Five of them are top results by searching on Google Scholar with the keyword "journal of dairy science machine learning". And the remaining one is chosen to represent my concern. Among these papers, two of them implemented ML models properly, while the other four showed the mentioned concerns. Below is the identified issues and how my submission can help improve these papers:

Ghaffari et al. (2019) (cited by 58)
    Concerns: Falsely conducted model validation. The study did not include feature selection and hyperparameter turning in the cross validation procedure. Besdies, the small dataset (38 cows) with many explanatory variables (170 metabolites) increases the risk of overfitting. This error has been pointed out by the book "The Elements of Statistical Learning" by Hastie et al. (2009) in p. 247, which is cited in the review paper.

Frizzarin et al. (2020) (cited by 49)
    Concerns: In p.7440, the authors falsely used the same data for hyperparamenter tuning (the number of factors in partial least square regression) from external validation, giving unfair advantage to the model. In p.7442, the term "cross-validation data" is ambiguous. It is unclear if the evaluation is independent of the training process.

Brand et al. (2020) (cited by 38)
    Concerns: The Table 1 in p.4985, the authors did not mention how the hyperpameters (i.e., retention rate and random selection rate) were tuned in cross validation. Besides, except for a brief discussion (p.4988), the paper did not describe how the external validation was conducted to obtain the final result accuracy.

Frizzarin et al. 2021 (cited by 12)
    Concerns: There is no cross validation. And the validation set is not randomly assigned nor explained how the validation set is selected. The result could only favor to the selected dataset if no randomization is conducted.

Suggestions: The risk in all these four papers can be mitigated by following the guideline in the review paper. The review paper provides a clear definition of the cross validation and how to properly conduct hyperparameter tuning in its Figure 7 and 8. Correct implementation from JDS papers were also cited for further reference. The Becker et al. 2020 and Mota et al. 2021 has set good examples for model validation, but the readers can get visualized understanding from the review paper in its Figure 9.

## The proposed action

The previous submission has received valuable feedback from the domain experts. The comments can be concluded as two main points: (1) the paper is too abstract and generic, and (2) unneccessary details of ML theories. My proposed action is to cooperate with Dr. Robin White, a nutritionist who is experienced in conducting ML studies in interdiciplinary research, to revise the paper. We will focus on providing more engaging examples in dairy science and trimming the theories part as requested. I believe that the revised paper will be more appealing to the JDS audiences and can be a valuable asset for the JDS community.

