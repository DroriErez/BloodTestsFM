import numpy as np
import pandas as pd
import csv
import lightgbm as lgb
import seaborn as sns
from scipy.stats import pearsonr

import bloodtest_data_utils

import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score



def fill_matrix(demographic_matrix,matrix,mode):

    filled_matrix = matrix.copy()
    # print(filled_matrix)
    if mode == "median": # median
        num_columns = filled_matrix.shape[1]
        for col_index in range(num_columns):
            col = filled_matrix[:, col_index]
            median_value = np.median((col != 0) & (~np.isnan(col)))
            zero_mask = (col == 0) | np.isnan(col)
            filled_matrix[:, col_index][zero_mask] = median_value
    elif mode == "knn": # k-nearest neighbor
        k = 3
        num_columns = filled_matrix.shape[1]
        for col_index in range(num_columns):
            col = filled_matrix[:, col_index]
            for gender in [bloodtest_data_utils.MALE,bloodtest_data_utils.FEMALE]:
                gender_mask = demographic_matrix[:,bloodtest_data_utils.GENDER_COL_INDEX] == gender
                zero_mask = (col == 0) | np.isnan(col) 
                zero_mask = zero_mask & gender_mask
                if (np.sum(zero_mask) > 0):
                    non_zero_mask = (col != 0) & ~np.isnan(col)
                    non_zero_mask = non_zero_mask & gender_mask
                    x_values = demographic_matrix[:,bloodtest_data_utils.AGE_COL_INDEX][non_zero_mask]
                    y_values = col[non_zero_mask]
                    knn = KNeighborsRegressor(n_neighbors=k, weights='distance')
                    knn.fit(x_values.reshape(-1, 1),y_values)

                    y_pred = knn.predict(demographic_matrix[:,bloodtest_data_utils.AGE_COL_INDEX][zero_mask].reshape(-1, 1))

                    filled_matrix[:, col_index][zero_mask] = y_pred
                 

    # print(demographic_matrix[:,AGE_COL_INDEX])
    # print(filled_matrix)

    return filled_matrix

def prepare_matrix_to_linear_regression(orig_matrix,filled_matrix,col):

    # seperate rows with values on the orig matrix 
    

    row_mask = (orig_matrix[:,col] != 0) & ~np.isnan(orig_matrix[:,col])
    Xs_matrix = filled_matrix[row_mask]

    #take the column as Y
    Ys_matrix = Xs_matrix[:,col]

    #take all other values as Xs
    cols_to_remove = []
    cols_to_remove.append(col) 

    for blood_test_couple in bloodtest_data_utils.COUPLED_BLOOD_TESTS:
        if FILTERED_BT_LABELS[col] in blood_test_couple:
            for blood_test in blood_test_couple:
                if (FILTERED_BT_LABELS[col] != blood_test):
                    cols_to_remove.append(FILTERED_BT_LABELS.index(blood_test)) 

    Xs_matrix = np.delete(Xs_matrix, cols_to_remove, axis=1)


    return Xs_matrix, Ys_matrix

def plot_linear_regression_results(X_test,Y_test,Y_pred,mse):

    plt.title("Linear Regression - MSE:{}".format(mse))

    plt.scatter(range(Y_test.shape[0]), Y_test, label="Data")
    plt.scatter(range(Y_pred.shape[0]), Y_pred, color='red', label="Linear Regression prediction".format(mse))
    plt.xlabel("Feature")
    plt.ylabel("Target")
    plt.legend()
    plt.show()

def linear_regression_model(X,Y):


    # Split the data into training and testing sets
    X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, random_state=42)

    # Create a LinearRegression model
    model = LinearRegression()

    # Train the model on the training data
    model.fit(X_train, Y_train)

    # Make predictions on the test data
    Y_pred = model.predict(X_test)

    mse = mean_squared_error(Y_test, Y_pred)
    rmse = np.sqrt(mse)

    pearson_correlation = np.corrcoef(Y_test, Y_pred)[0, 1]

    print(f"Mean Squared Error (MSE): {mse:.2f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    print(f"Pearson correlation : {pearson_correlation:.2f}")

    # plot_linear_regression_results(X_test,Y_test,Y_pred,mse)
    # print(pearson_correlation,model.coef_)
    return Y_pred, mse, pearson_correlation

def light_gbm_model(X,Y):


    # Split the data into training and testing sets
    X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, random_state=42)

    train_data = lgb.Dataset(X_train, label=Y_train)
    test_data = lgb.Dataset(X_test, label=Y_test, reference=train_data)

    params = {
        'objective': 'regression',
        'metric': 'mse',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9
}

    num_round = 100  # Number of boosting rounds
    model = lgb.train(params, train_data, num_round)

    # Train the model on the training data

    Y_pred = model.predict(X_test, num_iteration=model.best_iteration)

    mse = mean_squared_error(Y_test, Y_pred)
    rmse = np.sqrt(mse)
    pearson_correlation = np.corrcoef(Y_test, Y_pred)[0, 1]

    print(f"Mean Squared Error (MSE): {mse:.2f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    print(f"Pearson correlation : {pearson_correlation:.2f}")

    plot_linear_regression_results(X_test,Y_test,Y_pred,mse)
    return Y_pred, mse, pearson_correlation


bt_df = bloodtest_data_utils.load_data_as_df("BT.csv")

BT_LABELS = bt_df.columns.tolist()
BT_LABELS.remove('age')
BT_LABELS.remove('gender')
BT_LABELS.remove('RegistrationCode')
BT_LABELS.remove('Date')

print("Number of records:{} number of blood tests:{}".format(bt_df.shape[0], len(BT_LABELS)))


data_matrix,demographic_matrix = bloodtest_data_utils.generate_np_matrixes_from_10df(bt_df)

print("Number of participants: {}".format(data_matrix.shape[0]))


blood_tests_df = bloodtest_data_utils.load_data_as_df(bloodtest_data_utils.MODEL_BLOODTEST_FILE)
FILTERED_BT_LABELS =  blood_tests_df['BloodTest'].tolist()

print("Number of model blood tests: {}".format(len(FILTERED_BT_LABELS)))

filtered_indexes = [BT_LABELS.index(value) for index, value in enumerate(FILTERED_BT_LABELS) if value in BT_LABELS]

filtered_data_matrix = data_matrix[:,filtered_indexes]

models_definition = [
    {"model":"linear","imputation":True,"imputation_method":"median","enrichment_strategy":"Only_first"},
    {"model":"linear","imputation":True,"imputation_method":"knn","enrichment_strategy":"Only_first"}
]


distances = []

for model_definition in models_definition:

    model_performance_df = pd.DataFrame(columns=["Test","Pearson","MSE"])


    if model_definition["imputation"]:
        filled_matrix = fill_matrix(demographic_matrix,filtered_data_matrix, model_definition["imputation_method"])
        # filled_matrix = fill_matrix(demographic_matrix,filtered_data_matrix, mode=1)
    else:
        filled_matrix = np.copy(filtered_data_matrix)

    correlation_v = np.empty(filtered_data_matrix.shape[1])

    for col in range(filtered_data_matrix.shape[1]):
        X,Y = prepare_matrix_to_linear_regression(filtered_data_matrix,filled_matrix,col)

        if model_definition['model'] == 'linear':
            y_pred,mse,pearson = linear_regression_model(X,Y)

        correlation_v[col] = pearson
        model_performance_df.loc[col] = [FILTERED_BT_LABELS[col],pearson,mse]

    distance = np.linalg.norm(np.ones(filtered_data_matrix.shape[1]) - correlation_v)
    distances.append(distance)

    filename = "ModelPerformance_"+model_definition['model']
    if (model_definition["imputation"]):
        filename+="_"+model_definition['imputation_method']
    filename+= "_"+ model_definition['enrichment_strategy']


    model_performance_df.to_csv("model_results\\"+filename+".csv")

# print(data_matrix)
# for col in range(filtered_data_matrix.shape[1]):
#     X,Y = prepare_matrix_to_linear_regression(filtered_data_matrix,filled_matrix,col)
#     # print("X matrix for col:",col)
#     # print(X)
#     # print("Y matrix for col", col)
#     # print(Y)

#     y_pred,mse,pearson =light_gbm_model(X,Y)

sorted_indices = np.argsort(distances)

for i in sorted_indices:
    print (models_definition[i],distances[i])



