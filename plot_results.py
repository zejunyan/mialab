import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import glob, os
import seaborn as sns


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
        date = os.path.basename(os.path.dirname(file_path))
        
        df = pd.read_csv(file_path, sep=";")
        
        df['date'] = date
        
        all_data.append(df)

    df = pd.concat(all_data, ignore_index=True)

    df.columns = ['Label', 'Metric', 'Statistic', 'Value', 'Date']

    df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d-%H-%M-%S')

    df_mean = df[df['Statistic'] == 'MEAN'].copy()
    df_std = df[df['Statistic'] == 'STD'].copy()

    df_mean_pivot = df_mean.pivot_table(
        index=['Date', 'Label'], 
        columns='Metric', 
        values='Value'
    ).reset_index()

    df_std_pivot = df_std.pivot_table(
        index=['Date', 'Label'], 
        columns='Metric', 
        values='Value'
    ).reset_index()

    df_mean_pivot.columns = ['Date', 'Label', 'DICE', 'HDRFDST']
    df_std_pivot.columns = ['Date', 'Label', 'DICE_std', 'HDRFDST_std']

    df= pd.merge(df_mean_pivot, df_std_pivot, on=['Date', 'Label'])

    df.to_csv('output.csv', index=False)

if __name__ == '__main__':
    main()
