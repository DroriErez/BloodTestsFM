import bloodtest_data_utils
import numpy as np
import pandas as pd
import csv
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns



def save_fill_rates_to_csv(labels,fill_rate, male_fill_rate,female_fill_rate,min_fill_rate):


        # Specify the CSV file name
        csv_file = "fill_rate.csv"

        # Combine the arrays into a list of rows
        rows = zip(labels,fill_rate, male_fill_rate, female_fill_rate,min_fill_rate)

        # Write the rows to the CSV file
        with open(csv_file, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["Test","fill rate", "male fill rate", "female fill rate","min fill rate"])  # Header
            writer.writerows(rows)




def plot_tests_fill_ratio(labels,ratio,subtitle = ""):

    indexed_arr = list(enumerate(ratio))

    # Sort the indexed array based on values
    sorted_arr = sorted(indexed_arr,reverse=True, key=lambda x: x[1])

    # Print the sorted array with original indices


    bar_count = 0
    for index, value in sorted_arr:
        if value >= bloodtest_data_utils.FILL_RATE_THRESHOLD:
            plt.barh(labels[index], value*100)
            bar_count += 1
    
    print("For {} - Number of blood tests over threshold is {}".format(subtitle,bar_count))


    # Add labels and a title
    plt.xlabel('Blood tests')
    plt.ylabel('Fill rate')
    plt.title('Blood tests fill rate % - {}'.format(subtitle))

    # Show the plot
    plt.show(block=True)


def calc_correlated_matrix(bt_sparse_matrix):

    PEARSON_THRESHOLD = 0.7

    num_columns = bt_sparse_matrix.shape[1]
    correlation_matrix = np.zeros((num_columns,num_columns))
    for i in range(num_columns):
        for j in range(i, num_columns):

            mask_i = ~np.isnan(bt_sparse_matrix[:, i]) & (bt_sparse_matrix[:, i] != 0)
            mask_j = ~np.isnan(bt_sparse_matrix[:, j]) & (bt_sparse_matrix[:, j] != 0)

            values_i = bt_sparse_matrix[:, i][mask_i & mask_j ]
            values_j = bt_sparse_matrix[:, j][mask_i & mask_j ]

            # Calculate correlation coefficient
            if len(values_i) > 1:
                correlation_coef, _ = pearsonr(values_i, values_j)
            else:
                correlation_coef = 0

            # Store the result in the correlation matrix
            correlation_matrix[i, j] = correlation_coef
            correlation_matrix[j, i] = correlation_coef

            if (np.abs(correlation_matrix[i, j]) > PEARSON_THRESHOLD and i != j):

                bFound = False

                for bundled_list in bloodtest_data_utils.COUPLED_BLOOD_TESTS:
                    if FILTERED_BT_LABELS[i] in bundled_list and FILTERED_BT_LABELS[j] in bundled_list:
                        bFound = True
                # if (not bFound):
                #     print("For {}({}) and {}({}) Pearson value is {} with {} samples".format(FILTERED_BT_LABELS[i],i,FILTERED_BT_LABELS[j],j,correlation_coef,len(values_i)))

    return correlation_matrix

def plot_correlation_matrix(correlation_matrix):


    # Set the style of the heatmap (optional)
    sns.set(style="white")

    # Create a heatmap of the correlation matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=.5)
    plt.title("Correlation Matrix")

    # Show the plot
    plt.show(block=True)



bt_df = bloodtest_data_utils.load_data_as_df("BT.csv")

BT_LABELS = bt_df.columns.tolist()
BT_LABELS.remove('age')
BT_LABELS.remove('gender')
BT_LABELS.remove('RegistrationCode')
BT_LABELS.remove('Date')

print("Number of records:{} number of blood tests:{}".format(bt_df.shape[0], len(BT_LABELS)))


blood_tests_list_df = pd.DataFrame({'Blood tests': BT_LABELS})

# Save the DataFrame to a CSV file
blood_tests_list_df.to_csv("blood_tests_full_list.csv", index=True)

data_matrix,demographic_matrix = bloodtest_data_utils.generate_np_matrixes_from_10df(bt_df)

print("Number of participants: {}".format(data_matrix.shape[0]))


# data_matrix,demographic_matrix = generate_data()


FILTERED_BT_LABELS, fill_rate, male_ratios,female_ratios,min_fill_ratio_by_gender = bloodtest_data_utils.genereate_bloodtests_based_on_fill_rate(BT_LABELS,data_matrix,demographic_matrix)

print("Number of tests with fill rate on both genders higher than {}: {}".format(bloodtest_data_utils.FILL_RATE_THRESHOLD,len(FILTERED_BT_LABELS)))

save_fill_rates_to_csv(BT_LABELS,fill_rate,male_ratios,female_ratios,min_fill_ratio_by_gender)

bloodtest_list_df = pd.DataFrame(FILTERED_BT_LABELS, columns=['BloodTest'])  # Replace 'Column_Name' with a meaningful name
bloodtest_list_df.to_csv(bloodtest_data_utils.MODEL_BLOODTEST_FILE,index=True)


plot_tests_fill_ratio(BT_LABELS,fill_rate,"All population")
plot_tests_fill_ratio(BT_LABELS,male_ratios,"Male fill rate")
plot_tests_fill_ratio(BT_LABELS,female_ratios,"Female fill ratio")
plot_tests_fill_ratio(BT_LABELS,min_fill_ratio_by_gender,"Min across genders")

filtered_indexes = [BT_LABELS.index(value) for index, value in enumerate(FILTERED_BT_LABELS) if value in BT_LABELS]
filtered_data_matrix = data_matrix[:,filtered_indexes]


correlation_matrix = calc_correlated_matrix(filtered_data_matrix)

correlation_df = pd.DataFrame(correlation_matrix, index=FILTERED_BT_LABELS, columns=FILTERED_BT_LABELS)

# Save the DataFrame to CSV
correlation_df.to_csv("pearson_correlation2f.csv",float_format='%.2f')

plot_correlation_matrix(correlation_matrix)

