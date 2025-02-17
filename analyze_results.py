import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd 
from pathlib import Path 


def plot_results():
    """
    This function read from the results_sheets folder and plots the results (lineplot)
    into the results_plots folder
    """

    # Read the results 
    sheet_path = Path().cwd() / 'results_sheets'
    saving_path = Path().cwd() / 'results_plots'
    metrics = ['perplexity', 'accuracy']
    for sheet_file in sheet_path.iterdir():
        sheets = pd.read_excel(sheet_file, sheet_name=None)
        current_saving_path = saving_path / sheet_file.name.split('.')[0]
        current_saving_path.mkdir(parents=True, exist_ok=True)
        for sheet_name, df in sheets.items():
            print(f"Sheet: {sheet_name}")
            df = df.set_index(df.columns[0])
            print(df.head())
            variable_param = df.columns.tolist()
            for metric in metrics:
                if metric in df.index.tolist():
                    metric_values = pd.to_numeric(df.loc[metric]).values.tolist()
                    plt.figure(figsize=(8, 5))
                    sns.lineplot(x=variable_param, y=metric_values, marker='o', label=sheet_file.name.split('.')[0])
                    if f"FULL_GPN_{metric}" in df.index.tolist():
                        stoa_metric = pd.to_numeric(df.loc[f"FULL_GPN_{metric}"]).values.tolist()
                        sns.lineplot(x=variable_param, y=stoa_metric, marker='o', label='GPN')

                    # Customize the plot
                    plt.title(f'{metric} vs {sheet_name}')
                    plt.xlabel(f'{sheet_name}')
                    plt.ylabel(f'{metric}')
                    plt.grid(True)  

                    plt.savefig(current_saving_path / f"{sheet_name}_{metric}.png")

                    # Show the plot
                    plt.show()

if __name__ == '__main__':
    plot_results()
    