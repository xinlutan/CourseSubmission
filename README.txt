This submission contains four Jupyter notebooks, available in the Code Folder: 

(1) prepare_data.ipynb: This notebook loads raw data from the Data folder, performs some exploratory analysis, data preprocessing and cleaning.  The cleaned data ready for training, validation, and testing are stored in the CleanData folder.  Unfortunately, I had to remove the Data and CleanData folders to reduce file size, but the raw data are available for download from Kaggle website, and cleaned data can be reproduced using this notebook.

(2) explore_tune_model.ipynb: This notebooks loads the cleaned training and validation data from the CleanData folder, and trains and tunes light GBM and linear models.  Models are trained on the training data and evaluated on the validation data.  A simple ensembling scheme based on simple convex combination of light GBM and linear model predictions are considered at the end of the notebook.  The best convex combination is determined using light GBM and linear model predictions on the validation data only.

(3) train.ipynb: This notebook retrains light GBM and linear models using all the data available for training (the aggregation of both training and validation data used in explore_tune_model.ipynb).  The models are stored in the Model folder.

(4) predict.ipynb: This notebook loads the trained light GBM and linear models from the Model folder, and also the cleaned test data from thr ClearnData folder.  It then generates test predictions from both models and combines them using the alpha coefficient determined in explore_tune_model.ipynb.  The resuting predictions are then stored in the Submission folder, ready for submission.  The public and private LB scores I received are: 0.967965 and 0.967967.

In addition, some utility functions used in explore_tune_model.ipynb are included in Utility.py.