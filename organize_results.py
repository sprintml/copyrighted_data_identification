import pandas as pd
from tabulate import tabulate
import argparse


def main(csv_file_path):
    # Read the CSV file into a DataFrame
    try:
        df = pd.read_csv(csv_file_path)
    except FileNotFoundError:
        print("Error: File not found.")
        return

    # Create pivot tables
    table_tpr_1 = pd.pivot_table(df, values="tpr_1%", index="model", columns="attack")
    table_tpr_01 = pd.pivot_table(
        df, values="tpr_0.1%", index="model", columns="attack"
    )
    table_p_value = pd.pivot_table(
        df, values="p_value", index="model", columns="attack"
    )

    # Function to print a DataFrame as a formatted table with grid lines
    def print_table(df, title):
        print(title)
        # Convert values to percentages and display with % for tpr_1% and tpr_0.1%
        if "tpr" in title:
            df_percent = df.applymap(lambda x: f"{x*100:.2f}%")
        else:
            df_percent = df
        print(tabulate(df_percent, headers="keys", tablefmt="grid"))

    # Print the tables
    print_table(table_tpr_1, "\nTable: tpr_1% values")
    print_table(table_tpr_01, "\nTable: tpr_0.1% values")
    print_table(table_p_value, "\nTable: p_value values")


if __name__ == "__main__":
    # Create an ArgumentParser object
    parser = argparse.ArgumentParser(description="Process CSV file for analysis.")

    # Add an argument for the CSV file path
    parser.add_argument("csv_file_path", type=str, help="Path to the CSV file")

    # Parse the command-line arguments
    args = parser.parse_args()

    # Call the main function with the provided CSV file path
    main(args.csv_file_path)
