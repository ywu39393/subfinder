## In this script, we will look at the data preparation steps with the unsupervised sequences


## make all the library imports here
import pandas as pd
import os

## function to take the many text files that are there in the 
## unsupervised folders, and create a nice csv after reading them in
def read_the_data(input_folder, output_file):
    # List all CSV files in the input folder
    csv_files = [f for f in os.listdir(input_folder) if f.endswith('.csv')]
    
    # Read each CSV file into a DataFrame and store it in a list
    dataframes = [pd.read_csv(os.path.join(input_folder, csv_file)) for csv_file in csv_files]
    
    # Concatenate all DataFrames into a single DataFrame
    combined_df = pd.concat(dataframes, ignore_index=True)
    
    # Save the combined DataFrame to a new CSV file
    combined_df.to_csv(output_file, index=False)