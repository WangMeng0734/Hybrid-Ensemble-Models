import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,  roc_auc_score,  confusion_matrix
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns
import HOA as HOA




# Loading data from an Excel file
file_path = r"your file_path"  # Path to your file
data = pd.read_excel(file_path, index_col=False)  # Read the Excel file into a DataFrame

# Divide the data into input features (X) and output labels (y)
X = data.iloc[:, :6]  # Select the first six columns as input features
y = data.iloc[:, -1]  # Select the last column as the output label

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
# test_size=0.20 means 20% of the data will be used for testing, and the rest will be for training
# random_state=8286 ensures that the split is reproducible (same result every time)


#==================================================================Define the fitness function===================================================================================
# HOA Optimization for LGBM Classifier
def fun_lgbm(params, iteration_results):
    # Unpack the hyperparameters
    n_estimators, learning_rate, max_depth, num_leaves = params

    # Create a 5-fold cross-validator
    kf = KFold(n_splits=5)
    accuracy_scores = []

    # Perform cross-validation
    for train_index, val_index in kf.split(X_train):
        # Split the data into training and validation sets
        X_train_kf, X_val_kf = X_train.iloc[train_index], X_train.iloc[val_index]
        y_train_kf, y_val_kf = y_train.iloc[train_index], y_train.iloc[val_index]

        # Initialize the LightGBM classifier with given hyperparameters
        model = lgb.LGBMClassifier(
            n_estimators=int(n_estimators),
            learning_rate=learning_rate,
            max_depth=int(max_depth),
            num_leaves=int(num_leaves)
        )

        # Train the model
        model.fit(X_train_kf, y_train_kf)

        # Predict on the validation set
        y_pred = model.predict(X_val_kf)

        # Evaluate accuracy
        accuracy_scores.append(accuracy_score(y_val_kf, y_pred))

    # Compute average accuracy
    avg_accuracy = np.mean(accuracy_scores)

    # Store the negative accuracy for optimization (e.g., minimizing function)
    iteration_results.append(-avg_accuracy)

    return -avg_accuracy


# HOA Optimization for XGBoost Classifier
def fun_xgb(params, iteration_results):
    # Unpack the hyperparameters
    n_estimators, learning_rate, max_depth, colsample_bytree = params

    # Create a 5-fold cross-validator
    kf = KFold(n_splits=5)
    accuracy_scores = []

    # Perform cross-validation
    for train_index, val_index in kf.split(X_train):
        # Split the data into training and validation sets
        X_train_kf, X_val_kf = X_train.iloc[train_index], X_train.iloc[val_index]
        y_train_kf, y_val_kf = y_train.iloc[train_index], y_train.iloc[val_index]

        # Initialize the XGBoost classifier with given hyperparameters
        model = xgb.XGBClassifier(
            n_estimators=int(n_estimators),
            learning_rate=learning_rate,
            max_depth=int(max_depth),
            gamma=0,
            colsample_bytree=colsample_bytree
        )

        # Train the model
        model.fit(X_train_kf, y_train_kf)

        # Predict on the validation set
        y_pred = model.predict(X_val_kf)

        # Evaluate accuracy
        accuracy_scores.append(accuracy_score(y_val_kf, y_pred))

    # Return the negative average accuracy for optimization
    return -np.mean(accuracy_scores)



#==================================================================Hyperparameter Optimization===================================================================================

# Initialize counters for unique identifiers
lgbm_counter = 0
xgb_counter = 0

# Function to generate a unique identifier for LGBM models
def generate_identifier_lgb(prefix='lgbm'):
    global lgbm_counter
    lgbm_counter += 1
    return f"{prefix}{lgbm_counter}"

# Function to generate a unique identifier for XGBoost models
def generate_identifier_xgb(prefix='xgb'):
    global xgb_counter
    xgb_counter += 1
    return f"{prefix}{xgb_counter}"

# **Hyperparameter Optimization using HOA** 

# LGBM optimization function
def optimize_lgbm():
    N, Dim = 10, 4  # Number of particles and dimensions
    T = 200  # Number of iterations
    LB_lgbm, UB_lgbm = [5, 0.01, 2, 2], [200, 1, 10, 10]  # Lower and upper bounds for the hyperparameters

    # Initialize the list to store convergence results
    iteration_results = []
    
    # Generate a unique identifier for this LGBM optimization
    identifier = generate_identifier_lgb()

    # Perform HOA (Harmony Search Optimization) for LGBM
    Best_FF_lgbm, Best_P_lgbm, conv_lgbm = HOA.HOA(N, T, LB_lgbm, UB_lgbm, Dim, 
                                                   lambda params: fun_lgbm(params, iteration_results), identifier=identifier)

    # Plot convergence curve showing fitness over iterations
    plt.plot(conv_lgbm)
    plt.xlabel("Iteration")
    plt.ylabel("Fitness")
    plt.title("Convergence Curve for LGBM Optimization")
    plt.show()

    return Best_P_lgbm, Best_FF_lgbm

# XGBoost optimization function
def optimize_xgb():
    N, Dim = 10, 4  # Number of particles and dimensions
    T = 200  # Number of iterations
    LB_xgb, UB_xgb = [5, 0.01, 2, 0.5], [200, 1, 10, 1]  # Lower and upper bounds for the hyperparameters (without gamma)

    # Initialize the list to store convergence results
    iteration_results = []

    # Generate a unique identifier for this XGBoost optimization
    identifier = generate_identifier_xgb(prefix='xgb')

    # Perform HOA (Harmony Search Optimization) for XGBoost
    Best_FF_xgb, Best_P_xgb, conv_xgb = HOA.HOA(N, T, LB_xgb, UB_xgb, Dim, 
                                                lambda params: fun_xgb(params, iteration_results), identifier=identifier)
    
    # Plot convergence curve showing fitness over iterations
    plt.plot(conv_xgb)
    plt.xlabel("Iteration")
    plt.ylabel("Fitness")
    plt.title("Convergence Curve for XGBoost Optimization")
    plt.show()

    return Best_P_xgb, Best_FF_xgb

#============================================================Create and train three hybrid LGBM base learners==============================================================

# Debugging the results of optimize_lgbm to ensure the parameters are returned as lists/tuples
best_params_lgbm_1 = optimize_lgbm()
best_params_lgbm_2 = optimize_lgbm()
best_params_lgbm_3 = optimize_lgbm()

# Check what is being returned
# Extract the hyperparameters from the first element of the tuple
best_params_lgbm_1_values = best_params_lgbm_1[0]  # This is the array containing the hyperparameters
best_params_lgbm_2_values = best_params_lgbm_2[0]
best_params_lgbm_3_values = best_params_lgbm_3[0]

# Initialize the models using the extracted values
model_lgbm_1 = lgb.LGBMClassifier(
    n_estimators=int(best_params_lgbm_1_values[0]),  # Accessing n_estimators
    learning_rate=best_params_lgbm_1_values[1],      # Accessing learning_rate
    max_depth=int(best_params_lgbm_1_values[2]),     # Accessing max_depth
    num_leaves=int(best_params_lgbm_1_values[3])     # Accessing num_leaves
)

model_lgbm_2 = lgb.LGBMClassifier(
    n_estimators=int(best_params_lgbm_2_values[0]),  # Accessing n_estimators
    learning_rate=best_params_lgbm_2_values[1],      # Accessing learning_rate
    max_depth=int(best_params_lgbm_2_values[2]),     # Accessing max_depth
    num_leaves=int(best_params_lgbm_2_values[3])     # Accessing num_leaves
)

model_lgbm_3 = lgb.LGBMClassifier(
    n_estimators=int(best_params_lgbm_3_values[0]),  # Accessing n_estimators
    learning_rate=best_params_lgbm_3_values[1],      # Accessing learning_rate
    max_depth=int(best_params_lgbm_3_values[2]),     # Accessing max_depth
    num_leaves=int(best_params_lgbm_3_values[3])     # Accessing num_leaves
)

 
#============================================================================Create and train three hybrid XGB base learners==============================================================

best_params_xgb_1 = optimize_xgb()
best_params_xgb_2 = optimize_xgb()
best_params_xgb_3 = optimize_xgb()
# Extract the hyperparameters from the first element of the tuple
best_params_xgb_1_values = best_params_xgb_1[0]  # This is the array containing the hyperparameters
best_params_xgb_2_values = best_params_xgb_2[0]
best_params_xgb_3_values = best_params_xgb_3[0]

# Initialize the XGBoost models using the extracted values
model_xgb_1 = xgb.XGBClassifier(
    n_estimators=int(best_params_xgb_1_values[0]),  # Accessing n_estimators
    learning_rate=best_params_xgb_1_values[1],      # Accessing learning_rate
    max_depth=int(best_params_xgb_1_values[2]),     # Accessing max_depth
    colsample_bytree=best_params_xgb_1_values[3]    # Accessing colsample_bytree
)

model_xgb_2 = xgb.XGBClassifier(
    n_estimators=int(best_params_xgb_2_values[0]),  # Accessing n_estimators
    learning_rate=best_params_xgb_2_values[1],      # Accessing learning_rate
    max_depth=int(best_params_xgb_2_values[2]),     # Accessing max_depth
    colsample_bytree=best_params_xgb_2_values[3]    # Accessing colsample_bytree
)

model_xgb_3 = xgb.XGBClassifier(
    n_estimators=int(best_params_xgb_3_values[0]),  # Accessing n_estimators
    learning_rate=best_params_xgb_3_values[1],      # Accessing learning_rate
    max_depth=int(best_params_xgb_3_values[2]),     # Accessing max_depth
    colsample_bytree=best_params_xgb_3_values[3]    # Accessing colsample_bytree
)


#==========================================================================Evaluate each base learner and get prediction probabilities==========================================

# Evaluate each base learner, get prediction probabilities, and confusion matrices
def evaluate_model(model, model_name):
    # Train the model
    model.fit(X_train, y_train)
    
    # Predictions for the training set
    y_train_pred = model.predict(X_train)
    y_train_prob = model.predict_proba(X_train)[:, 1]
    
    # Predictions for the test set
    y_test_pred = model.predict(X_test)
    y_test_prob = model.predict_proba(X_test)[:, 1]
    
    # Calculate evaluation metrics for the training set
    train_accuracy = accuracy_score(y_train, y_train_pred)
    train_precision = precision_score(y_train, y_train_pred)
    train_recall = recall_score(y_train, y_train_pred)
    train_f1 = f1_score(y_train, y_train_pred)
    train_auc = roc_auc_score(y_train, y_train_prob)  # Calculate AUC for the training set
    
    # Calculate evaluation metrics for the test set
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_precision = precision_score(y_test, y_test_pred)
    test_recall = recall_score(y_test, y_test_pred)
    test_f1 = f1_score(y_test, y_test_pred)
    test_auc = roc_auc_score(y_test, y_test_prob)  # Calculate AUC for the test set
    
    # Calculate confusion matrix for the training set
    train_cm = confusion_matrix(y_train, y_train_pred)
    test_cm = confusion_matrix(y_test, y_test_pred)
    
    # Print the confusion matrix for the training set
    print(f"\n{model_name} - Training Set Confusion Matrix:")
    print(train_cm)
    sns.heatmap(train_cm, annot=True, fmt="d", cmap="Blues", xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
    plt.title(f"{model_name} - Training Set Confusion Matrix")
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()
    
    # Print the confusion matrix for the test set
    print(f"\n{model_name} - Test Set Confusion Matrix:")
    print(test_cm)
    sns.heatmap(test_cm, annot=True, fmt="d", cmap="Blues", xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
    plt.title(f"{model_name} - Test Set Confusion Matrix")
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()
    
    # Print evaluation metrics for the training set
    print(f"\n{model_name} - Training Set Evaluation Metrics:")
    print(f"Accuracy: {train_accuracy:.6f}, Precision: {train_precision:.6f}, Recall: {train_recall:.6f}, F1 Score: {train_f1:.6f}, AUC: {train_auc:.6f}")
    
    # Print evaluation metrics for the test set
    print(f"\n{model_name} - Test Set Evaluation Metrics:")
    print(f"Accuracy: {test_accuracy:.6f}, Precision: {test_precision:.6f}, Recall: {test_recall:.6f}, F1 Score: {test_f1:.6f},  AUC: {test_auc:.6f}")
    
    return y_test_prob

# Evaluate each base learner and get prediction probabilities
prob_lgbm_1 = evaluate_model(model_lgbm_1, 'LightGBM-1')
prob_lgbm_2 = evaluate_model(model_lgbm_2, 'LightGBM-2')
prob_lgbm_3 = evaluate_model(model_lgbm_3, 'LightGBM-3')
prob_xgb_1 = evaluate_model(model_xgb_1, 'XGBoost-1')
prob_xgb_2 = evaluate_model(model_xgb_2, 'XGBoost-2')
prob_xgb_3 = evaluate_model(model_xgb_3, 'XGBoost-3')


# =======================================================================Hybrid Voting Model===========================================================================================

# Create VOTE Model (Soft Voting)
vote_model = VotingClassifier(
    estimators=[('lgbm_1', model_lgbm_1),
                ('lgbm_2', model_lgbm_2),
                ('lgbm_3', model_lgbm_3),
                ('xgb_1', model_xgb_1),
                ('xgb_2', model_xgb_2),
                ('xgb_3', model_xgb_3)],
    voting='soft'  # Soft voting (based on probabilities)
)

# Evaluate VOTE Model and Get Confusion Matrix
def print_evaluation_metrics(y_true, y_pred, prob, model_name):
    # Calculate accuracy, precision, recall, F1 score, AUC
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, prob)
    

    print(f"\n{model_name} - Training/Test Set Evaluation Metrics:")
    print(f"Accuracy: {accuracy:.6f}")
    print(f"Precision: {precision:.6f}")
    print(f"Recall: {recall:.6f}")
    print(f"F1 Score: {f1:.6f}")
    print(f"AUC: {auc:.6f}")
    

def plot_confusion_matrix(cm, model_name, dataset_type):
    # Plot the confusion matrix using a heatmap
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
    plt.title(f"{model_name} - {dataset_type} Confusion Matrix")
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

# Train VOTE Model
vote_model.fit(X_train, y_train)

# Predict and Evaluate VOTE Model
vote_train_prob = vote_model.predict_proba(X_train)[:, 1]  # Get probability for the positive class
vote_test_prob = vote_model.predict_proba(X_test)[:, 1]  # Get probability for the positive class

# Use the Default Threshold (0.5) for Final Prediction
vote_best_test_pred = (vote_test_prob >= 0.5).astype(int)
vote_best_train_pred = (vote_train_prob >= 0.5).astype(int)

# Print VOTE Model Evaluation Metrics
print("\nVOTE Model with Default Threshold - Training Set Evaluation Metrics:")
print_evaluation_metrics(y_train, vote_best_train_pred, vote_train_prob, "VOTE - Training Set")

print("\nVOTE Model with Default Threshold - Test Set Evaluation Metrics:")
print_evaluation_metrics(y_test, vote_best_test_pred, vote_test_prob, "VOTE - Test Set")

# Calculate Confusion Matrix for Training and Test Set
train_cm = confusion_matrix(y_train, vote_best_train_pred)
test_cm = confusion_matrix(y_test, vote_best_test_pred)

# Plot Confusion Matrix for Training and Test Set
plot_confusion_matrix(train_cm, "VOTE", "Training Set")
plot_confusion_matrix(test_cm, "VOTE", "Test Set")

# =======================================================================Hybrid Stacking Model===========================================================================================

# Stacking Model Creation
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression

# Evaluate Stacking Model and Get Confusion Matrix
def print_evaluation_metrics(y_true, y_pred, prob, model_name):
    # Calculate accuracy, precision, recall, F1 score, AUC
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, prob)
    

    print(f"\n{model_name} - Training/Test Set Evaluation Metrics:")
    print(f"Accuracy: {accuracy:.6f}")
    print(f"Precision: {precision:.6f}")
    print(f"Recall: {recall:.6f}")
    print(f"F1 Score: {f1:.6f}")
    print(f"AUC: {auc:.6f}")
    

def plot_confusion_matrix(cm, model_name, dataset_type):
    # Plot the confusion matrix using a heatmap
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
    plt.title(f"{model_name} - {dataset_type} Confusion Matrix")
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

# Create Stacking Model
stacking_model = StackingClassifier(
    estimators=[('lgbm_1', model_lgbm_1),
                ('lgbm_2', model_lgbm_2),
                ('lgbm_3', model_lgbm_3),
                ('xgb_1', model_xgb_1),
                ('xgb_2', model_xgb_2),
                ('xgb_3', model_xgb_3)],
    final_estimator=LogisticRegression()  # Logistic regression as the final estimator
)

# Train Stacking Model
stacking_model.fit(X_train, y_train)

# Predict with Stacking Model
stacking_train_prob = stacking_model.predict_proba(X_train)[:, 1]  # Get probability for the positive class
stacking_test_prob = stacking_model.predict_proba(X_test)[:, 1]  # Get probability for the positive class

# Use Default Threshold (0.5) for Final Prediction
stacking_best_test_pred = (stacking_test_prob >= 0.5).astype(int)
stacking_best_train_pred = (stacking_train_prob >= 0.5).astype(int)

# Print Stacking Model Evaluation Metrics
print("\nStacking Model with Default Threshold - Training Set Evaluation Metrics:")
print_evaluation_metrics(y_train, stacking_best_train_pred, stacking_train_prob, "Stacking - Training Set")

print("\nStacking Model with Default Threshold - Test Set Evaluation Metrics:")
print_evaluation_metrics(y_test, stacking_best_test_pred, stacking_test_prob, "Stacking - Test Set")

# Calculate Confusion Matrix for Training and Test Set
train_cm = confusion_matrix(y_train, stacking_best_train_pred)
test_cm = confusion_matrix(y_test, stacking_best_test_pred)

# Plot Confusion Matrix for Training and Test Set
plot_confusion_matrix(train_cm, "Stacking", "Training Set")
plot_confusion_matrix(test_cm, "Stacking", "Test Set")












