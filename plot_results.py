import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import glob, os


def main():
    # todo: load the "results.csv" file from the mia-results directory
    # todo: read the data into a list
    # todo: plot the Dice coefficients per label (i.e. white matter, gray matter, hippocampus, amygdala, thalamus)
    #  in a boxplot

    # alternative: instead of manually loading/reading the csv file you could also use the pandas package
    # but you will need to install it first ('pip install pandas') and import it to this file ('import pandas as pd')
    df = pd.read_csv('mia-result/2025-12-05-13-39-10/results.csv', sep=";")
    
    amy = df[df['LABEL'] == 'WhiteMatter']
 
    csv_files = glob.glob('mia-result/*/results_summary.csv', recursive=True)

    all_data = []

    for file_path in csv_files:
        # Extract directory name (date) - parent directory of the file
        date = os.path.basename(os.path.dirname(file_path))
        
        # Read CSV
        df = pd.read_csv(file_path, sep=";")
        
        # Add date column
        df['date'] = date
        
        all_data.append(df)

    # Combine all DataFrames
    combined_df = pd.concat(all_data, ignore_index=True)
    print(combined_df)


if __name__ == '__main__':
    main()
