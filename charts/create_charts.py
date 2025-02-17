import pandas as pd
import matplotlib.pyplot as plt
import os
import openpyxl
print(openpyxl.__version__)


# Carica il file Excel
file_path = '../../../../work/ai4bio2023/results_tables/Results_experiments.xlsx'
# Carica il file Excel
df = pd.read_excel(file_path, header=None, engine='openpyxl')
print(df.head())  # Mostra le prime righe del file

# Parsing dei dati
models = []
current_model = {}

for _, row in df.iterrows():
    if isinstance(row[0], str) and 'ConvNet' in row[0]:
        if current_model:
            models.append(current_model)
        current_model = {'model': row[0]}
    elif isinstance(row[0], str):
        key, value = row[0].split()
        current_model[key] = value

if current_model:
    models.append(current_model)

# Converti i dati in un DataFrame
results_df = pd.DataFrame(models)

# Convertire colonne numeriche
results_df['test_loss'] = pd.to_numeric(results_df['test_loss'], errors='coerce')
results_df['perplexity'] = pd.to_numeric(results_df['perplexity'], errors='coerce')

# Convertire tempo totale in secondi per confronto
def convert_runtime_to_seconds(runtime_str):
    h, m, s = map(int, runtime_str.split(':'))
    return h * 3600 + m * 60 + s

results_df['total_runtime'] = results_df['total_runtime'].apply(convert_runtime_to_seconds)

# Estrai il percorso della cartella del file
output_dir = os.path.dirname(file_path)

# Grafico della perdita di test (test_loss)
plt.figure(figsize=(8, 5))
plt.plot(results_df['model'], results_df['test_loss'], marker='o', label='Test Loss', color='blue')
plt.xlabel('Model')
plt.ylabel('Test Loss')
plt.title('Test Loss Comparison')
plt.xticks(rotation=15)
plt.grid()
plt.legend()
plt.savefig(os.path.join(output_dir, 'test_loss_comparison.png'))
plt.close()

# Grafico della Perplexity
plt.figure(figsize=(8, 5))
plt.plot(results_df['model'], results_df['perplexity'], marker='o', label='Perplexity', color='orange')
plt.xlabel('Model')
plt.ylabel('Perplexity')
plt.title('Perplexity Comparison')
plt.xticks(rotation=15)
plt.grid()
plt.legend()
plt.savefig(os.path.join(output_dir, 'perplexity_comparison.png'))
plt.close()

# Grafico del tempo totale di esecuzione
plt.figure(figsize=(8, 5))
plt.bar(results_df['model'], results_df['total_runtime'], color='green', label='Total Runtime')
plt.xlabel('Model')
plt.ylabel('Total Runtime (seconds)')
plt.title('Total Runtime Comparison')
plt.xticks(rotation=15)
plt.grid()
plt.legend()
plt.savefig(os.path.join(output_dir, 'total_runtime_comparison.png'))
plt.close()

print(f"Grafici salvati nella cartella: {output_dir}")