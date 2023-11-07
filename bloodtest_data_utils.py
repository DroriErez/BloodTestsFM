import numpy as np
import pandas as pd
import csv
import lightgbm as lgb
import seaborn as sns


import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Generate a 3x3 random matrix with values sampled from a normal distribution (mean=0, std=1)

m = 100000

BT_LABELS = ["A","B","C","D","E","F","G","H","I","J"]

COUPLED_BLOOD_TESTS = [["A","B","G"],["C","D"]]

n = len(BT_LABELS) 
GENDER_COL_INDEX = 0
AGE_COL_INDEX = 1
MALE = 0
FEMALE = 1

def generate_data():
    bt_matrix = np.random.rand(m, n)

    demographic_matrix = np.random.rand(m, 2)
    demographic_matrix[:,0] = demographic_matrix[:,0] > 0.5
    demographic_matrix[:,1] = demographic_matrix[:,1] *60 + 18


    threshold = 0.3
    bt_matrix[bt_matrix < threshold] = 0

    bt_matrix = np.round(bt_matrix * 100,2)


    # Create a DataFrame with row and column labels
    df = pd.DataFrame(bt_matrix,  columns=BT_LABELS)

    # Specify the file path
    file_path = "demo_data.csv"

    # Save the DataFrame (with indexes) to a CSV file
    df.to_csv(file_path,index=False)

    return bt_matrix,demographic_matrix

def fill_matrix(demographic_matrix,matrix,mode):

    filled_matrix = matrix.copy()
    # print(filled_matrix)
    if mode == 0: # median
        num_columns = filled_matrix.shape[1]
        for col_index in range(num_columns):
            col = filled_matrix[:, col_index]
            median_value = np.median(col[col != 0])
            zero_mask = (col == 0)
            col[zero_mask] = median_value
    elif mode == 1: # k-nearest neighbor
        k = 3
        num_columns = filled_matrix.shape[1]
        for col_index in range(num_columns):
            col = filled_matrix[:, col_index]
            for gender in [MALE,FEMALE]:
                gender_mask = demographic_matrix[:,GENDER_COL_INDEX] == gender
                zero_mask = (col == 0 & gender_mask)
                if (np.sum(zero_mask) > 0):
                    non_zero_mask = (col != 0 & gender_mask)
                    x_values = demographic_matrix[:,AGE_COL_INDEX][non_zero_mask]
                    y_values = col[non_zero_mask]
                    knn = KNeighborsRegressor(n_neighbors=k, weights='distance')
                    knn.fit(x_values.reshape(-1, 1),y_values)

                    col[zero_mask] = knn.predict(demographic_matrix[:,AGE_COL_INDEX][zero_mask].reshape(-1, 1))
                 

    # print(demographic_matrix[:,AGE_COL_INDEX])
    # print(filled_matrix)

    return filled_matrix

def save_fill_rates_to_csv(fill_rate, male_fill_rate,female_fill_rate,min_fill_rate):


        # Specify the CSV file name
        csv_file = "fill_rate.csv"

        # Combine the arrays into a list of rows
        rows = zip(fill_rate, male_fill_rate, female_fill_rate,min_fill_rate)

        # Write the rows to the CSV file
        with open(csv_file, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["fill rate", "male fill rate", "female fill rate","min fill rate"])  # Header
            writer.writerows(rows)




def plot_tests_fill_ratio(ratio,subtitle = ""):
    indexed_arr = list(enumerate(ratio))

    # Sort the indexed array based on values
    sorted_arr = sorted(indexed_arr,reverse=True, key=lambda x: x[1])

    # Print the sorted array with original indices

    plt.bar([], [])

    for index, value in sorted_arr:
        plt.bar(BT_LABELS[index], value*100)


    # Add labels and a title
    plt.xlabel('Blood tests')
    plt.ylabel('Fill rate')
    plt.title('Blood tests fill rate % - {}'.format(subtitle))

    # Show the plot
    plt.show()


def data_columns_fill_ratio(data_matrix):

    column_sums = np.count_nonzero(data_matrix, axis=0)
    ratios = column_sums / data_matrix.shape[0]
    return ratios

def prepare_matrix_to_linear_regression(orig_matrix,filled_matrix,col):

    # seperate rows with values on the orig matrix 

    row_mask = orig_matrix[:,col] != 0
    Xs_matrix = filled_matrix[row_mask]

    #take the column as Y
    Ys_matrix = Xs_matrix[:,col]

    #take all other values as Xs
    cols_to_remove = []
    cols_to_remove.append(col) 

    for blood_test_couple in COUPLED_BLOOD_TESTS:
        if BT_LABELS[col] in blood_test_couple:
            for blood_test in blood_test_couple:
                if (BT_LABELS[col] != blood_test):
                    cols_to_remove.append(BT_LABELS.index(blood_test)) 

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

    plot_linear_regression_results(X_test,Y_test,Y_pred,mse)
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

def plot_correlation_matrix(correlation_matrix):


    # Set the style of the heatmap (optional)
    sns.set(style="white")

    # Create a heatmap of the correlation matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=.5)
    plt.title("Correlation Matrix")

    # Show the plot
    plt.show()


data_matrix,demographic_matrix = generate_data()

fill_rate = data_columns_fill_ratio(data_matrix)
plot_tests_fill_ratio(fill_rate,"All population")

male_mask = (demographic_matrix[:,GENDER_COL_INDEX] == MALE)
male_ratios = data_columns_fill_ratio(data_matrix[male_mask])
plot_tests_fill_ratio(male_ratios,"Male")

female_mask = (demographic_matrix[:,GENDER_COL_INDEX] == FEMALE)
female_ratios = data_columns_fill_ratio(data_matrix[female_mask])
plot_tests_fill_ratio(female_ratios,"Female")

plot_tests_fill_ratio(np.minimum(male_ratios,female_ratios),"Min across genders")


filled_matrix = fill_matrix(demographic_matrix,data_matrix, mode=0)
filled_matrix = fill_matrix(demographic_matrix,data_matrix, mode=1)

correlation_matrix = np.corrcoef(filled_matrix, rowvar=False)

plot_correlation_matrix(correlation_matrix)

save_fill_rates_to_csv(fill_rate,male_ratios,female_ratios,np.minimum(male_ratios,female_ratios))

# print(data_matrix)
for col in range(data_matrix.shape[1]):
    X,Y = prepare_matrix_to_linear_regression(data_matrix,filled_matrix,col)

    y_pred,mse,pearson = linear_regression_model(X,Y)

# print(data_matrix)
for col in range(data_matrix.shape[1]):
    X,Y = prepare_matrix_to_linear_regression(data_matrix,filled_matrix,col)
    # print("X matrix for col:",col)
    # print(X)
    # print("Y matrix for col", col)
    # print(Y)

    y_pred,mse,pearson =light_gbm_model(X,Y)




