import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Generate a 3x3 random matrix with values sampled from a normal distribution (mean=0, std=1)

m = 1000
n = 10

indexes = ["A","B","C","D","E","F","G","H","I","J"]


def generate_data():
    random_matrix = np.random.rand(m, n)


    threshold = 0.3
    random_matrix[random_matrix < threshold] = 0

    random_matrix = np.round(random_matrix * 100,2)


    # Create a DataFrame with row and column labels
    df = pd.DataFrame(random_matrix,  columns=indexes)

    # Specify the file path
    file_path = "demo_data.csv"

    # Save the DataFrame (with indexes) to a CSV file
    df.to_csv(file_path,index=False)

    return random_matrix

def plot_tests_fill_ratio(ratio):
    indexed_arr = list(enumerate(ratio))

    # Sort the indexed array based on values
    sorted_arr = sorted(indexed_arr,reverse=True, key=lambda x: x[1])

    # Print the sorted array with original indices

    plt.bar([], [])

    for index, value in sorted_arr:
        print(f"Index: {index}, Value: {value}, Test:{indexes[index]}") 
        plt.bar(indexes[index], value*100)


    # Add labels and a title
    plt.xlabel('Blood tests')
    plt.ylabel('Fill rate')
    plt.title('Blood tests fill rate %')

    # Show the plot
    plt.show()


def data_columns_fill_ratio(data_matrix):

    column_sums = np.count_nonzero(data_matrix, axis=0)
    print(column_sums,data_matrix.shape)
    ratios = column_sums / data_matrix.shape[0]
    return ratios

data_matrix = generate_data()
ratios = data_columns_fill_ratio(data_matrix)
plot_tests_fill_ratio(ratios)

