import numpy as np
import pandas as pd
import csv
import lightgbm as lgb
from scipy.stats import pearsonr

import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Generate a 3x3 random matrix with values sampled from a normal distribution (mean=0, std=1)

# BT_LABELS = []
# FILTERED_BT_LABELS = []

COUPLED_BLOOD_TESTS = [["bt__non_hdl_cholesterol","bt__hdl_cholesterol","bt__ldl_cholesterol_calc","bt__total_cholesterol","bt__ldl_cholesterol"],["bt__rbc_micro%","bt__rbc_hypo_%","bt__rbc","bt__hct","bt__hemoglobin"],["bt__luc_abs","bt__luc_%"],
                       ["bt__lymphocytes_abs","bt__lymphocytes_%"],["bt__neutrophils_%","bt__neutrophils_abs","bt__basophils_%","bt__basophils_abs","bt__eosinophils_abs","bt__eosinophils_%"],["bt__ast_got","bt__alt_gpt"]]
FILL_RATE_THRESHOLD = 0.2

MODEL_BLOODTEST_FILE = "model_blood_tests_list.csv"
# n = len(BT_LABELS) 
GENDER_COL_INDEX = 0
AGE_COL_INDEX = 1
MALE = 1
FEMALE = 0

def load_data_as_df(csvfile=None):

    if csvfile != None:
        df = pd.read_csv(csvfile)
        return df
    else:
        # Here we should the DF with 10K data loaders
        return None
    

def generate_np_matrixes_from_10df(df):

    num_rows, num_columns = df.shape

    data_matrix = np.empty((num_rows, num_columns - 4)) #removing id, date, age, gender
    demographic_matrix = np.empty((num_rows, 2))

    last_id = 0
    np_index = 0
    for index, row in df.iterrows():
        if last_id != row['RegistrationCode']:
            last_id = row['RegistrationCode']
            if ((not pd.isnull(row['gender']) and (not pd.isnull(row['gender'])))):
                demographic_matrix[np_index,GENDER_COL_INDEX] = row['gender']
                demographic_matrix[np_index,AGE_COL_INDEX] = row['age']
                del row['RegistrationCode']
                del row['gender']
                del row['age']
                del row['Date']
                data_matrix[np_index] = row
                np_index += 1

    return data_matrix[:np_index],demographic_matrix[:np_index]

def generate_data():
    BT_LABELS = ["A","B","C","D","E","F","G","H","I"]
    m = 100000
    bt_matrix = np.random.rand(m, len(BT_LABELS))

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
            for gender in [MALE,FEMALE]:
                gender_mask = demographic_matrix[:,GENDER_COL_INDEX] == gender
                zero_mask = (col == 0) | np.isnan(col) 
                zero_mask = zero_mask & gender_mask
                if (np.sum(zero_mask) > 0):
                    non_zero_mask = (col != 0) & ~np.isnan(col)
                    non_zero_mask = non_zero_mask & gender_mask
                    x_values = demographic_matrix[:,AGE_COL_INDEX][non_zero_mask]
                    y_values = col[non_zero_mask]
                    knn = KNeighborsRegressor(n_neighbors=k, weights='distance')
                    knn.fit(x_values.reshape(-1, 1),y_values)

                    y_pred = knn.predict(demographic_matrix[:,AGE_COL_INDEX][zero_mask].reshape(-1, 1))

                    filled_matrix[:, col_index][zero_mask] = y_pred
                 

    # print(demographic_matrix[:,AGE_COL_INDEX])
    # print(filled_matrix)

    return filled_matrix

def genereate_bloodtests_based_on_fill_rate(labels,data_matrix,demographic_matrix,fill_rate_threshold=FILL_RATE_THRESHOLD):

    fill_rate = data_columns_fill_ratio(data_matrix)

    male_mask = (demographic_matrix[:,GENDER_COL_INDEX] == MALE)
    male_ratios = data_columns_fill_ratio(data_matrix[male_mask])

    female_mask = (demographic_matrix[:,GENDER_COL_INDEX] == FEMALE)
    female_ratios = data_columns_fill_ratio(data_matrix[female_mask])


    min_fill_ratio_by_gender = np.minimum(male_ratios,female_ratios)

    filtered_bt_labels = [label for label, fill_rate in zip(labels, min_fill_ratio_by_gender) if fill_rate > fill_rate_threshold]

    return filtered_bt_labels, fill_rate, male_ratios, female_ratios, min_fill_ratio_by_gender



def data_columns_fill_ratio(data_matrix):

    num_of_rows = data_matrix.shape[0]
    column_sums = np.count_nonzero(~np.isnan(data_matrix), axis=0)
    ratios = column_sums / num_of_rows
    return ratios

def prepare_matrix_to_linear_regression(orig_matrix,filled_matrix,col):

    # seperate rows with values on the orig matrix 
    

    row_mask = (orig_matrix[:,col] != 0) & ~np.isnan(orig_matrix[:,col])
    Xs_matrix = filled_matrix[row_mask]

    #take the column as Y
    Ys_matrix = Xs_matrix[:,col]

    #take all other values as Xs
    cols_to_remove = []
    cols_to_remove.append(col) 

    for blood_test_couple in COUPLED_BLOOD_TESTS:
        if FILTERED_BT_LABELS[col] in blood_test_couple:
            for blood_test in blood_test_couple:
                if (FILTERED_BT_LABELS[col] != blood_test):
                    cols_to_remove.append(FILTERED_BT_LABELS.index(blood_test)) 

    Xs_matrix = np.delete(Xs_matrix, cols_to_remove, axis=1)


    return Xs_matrix, Ys_matrix







