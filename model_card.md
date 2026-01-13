# Model Card

## Model details
This project trains a machine learning model to predict whether a person makes more than $50K per year based on U.S. Census data. The model is a binary classifier built using scikit-learn and saved so it can be used later by a FastAPI application.

## Intended use
The model is intended for learning purposes only. It was built as part of a Machine Learning DevOps course to practice training a model, evaluating it, and deploying it through an API. It should not be used for real decisions about people.

## Training data
The model was trained using the `census.csv` dataset provided with the project. The target column is `salary`. Categorical features are encoded during training and the same encoder is reused during inference.

## Evaluation data
The dataset is split into training and test data using an 80/20 split. Model performance is evaluated on the test set that the model did not see during training.

## Metrics
The model is evaluated using precision, recall, and F1 score.

Results from running `python train_model.py` on the test set:
- Precision: 0.6815  
- Recall: 0.2825  
- F1 score: 0.3995  

Model performance is also evaluated on slices of the data for each categorical feature. These results are saved in `slice_output.txt`.

## Ethical considerations
This dataset includes demographic and employment information, so there is a risk that the model could reflect bias present in the data. Even if the overall metrics look reasonable, the model may perform differently for different groups.

## Caveats and recommendations
This model is relatively simple and not highly optimized. The recall score is low, meaning the model misses many people who actually earn more than $50K. If this model were improved in the future, it would be useful to tune hyperparameters, try different models, and continue monitoring performance across data slices.
