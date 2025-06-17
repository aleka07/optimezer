import os
import subprocess
import time
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import ScalarFormatter

# --- 1. Configuration ---
BASE_DIR = 'jan'
DATES = ['3', '4', '6', '7', '8', '10']
RESULTS_FILENAME = 'execution_times.csv'  # File to save/load benchmark results

# Dictionary of scripts to run
SCRIPTS_TO_RUN = {
    'CP-SAT': 'time_min.py',
    'GA': 'ga copy.py',
    'DQN': 'dqn copy.py'
}

# Define a fixed order for consistency in data collection and plotting
ALGORITHM_ORDER = ['CP-SAT', 'GA', 'DQN']

python_executable = sys.executable
child_env = os.environ.copy()
child_env["PYTHONIOENCODING"] = "utf-8"

# --- 2. Data Collection or Loading ---
if os.path.exists(RESULTS_FILENAME):
    print(f"--- Found results file '{RESULTS_FILENAME}'. Loading data... ---")
    print("--- To re-run the benchmarks, please delete this file. ---")
    results_df = pd.read_csv(RESULTS_FILENAME)
else:
    print(f"--- File '{RESULTS_FILENAME}' not found. Starting benchmarks... ---")
    execution_results = []
    print(f"--- Using Python interpreter: {python_executable} ---")
    for date in DATES:
        date_path = os.path.join(BASE_DIR, date)
        if not os.path.isdir(date_path):
            print(f"Warning: Directory for date '{date}' not found. Skipping.")
            continue
        print(f"\nProcessing data for Jan {date}...")
        for alg_name in ALGORITHM_ORDER:
            script_name = SCRIPTS_TO_RUN[alg_name]
            full_script_path = os.path.join(date_path, script_name)
            if not os.path.isfile(full_script_path):
                print(f"  - Warning: Script '{script_name}' not found in '{date_path}'. Skipping.")
                continue
            print(f"  - Running: {alg_name}...", end='', flush=True)
            start_time = time.time()
            try:
                subprocess.run(
                    [python_executable, script_name], cwd=date_path, check=True,
                    capture_output=True, text=True, encoding='utf-8', env=child_env
                )
            except subprocess.CalledProcessError as e:
                print(f"\n    !!! ERROR running '{script_name}' for date {date}:")
                print(f"    STDOUT: {e.stdout}")
                print(f"    STDERR: {e.stderr}")
                continue
            end_time = time.time()
            duration = end_time - start_time
            print(f" Finished in {duration:.2f} sec.")
            execution_results.append({
                'Date': f'Jan {date}', 'Algorithm': alg_name, 'ExecutionTime_sec': duration
            })
    print("\n--- Benchmarking finished. ---")
    if not execution_results:
        print("Could not collect any data. Results file will not be created.")
        sys.exit()
    results_df = pd.DataFrame(execution_results)
    results_df.to_csv(RESULTS_FILENAME, index=False)
    print(f"--- Results saved to file: {RESULTS_FILENAME} ---")

# --- 3. Results Visualization ---
print("\n--- Preparing the plot. ---")
plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(8, 6))

sns.barplot(data=results_df, x='Date', y='ExecutionTime_sec', hue='Algorithm',
            palette='viridis', ax=ax, edgecolor='black', hue_order=ALGORITHM_ORDER)

ax.set_yscale('log')
ax.yaxis.set_major_formatter(ScalarFormatter())

ax.set_title('Algorithm Execution Time Comparison (Log Scale)', fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel('Production Task Date', fontsize=11, labelpad=10)
ax.set_ylabel('Execution Time, seconds (log scale)', fontsize=11, labelpad=10)

# --- ИЗМЕНЕНИЕ: Легенда вынесена за пределы графика ---
# Move the legend outside the plot area to prevent any overlap
ax.legend(title='Algorithm', title_fontsize='11', fontsize='9',
          bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)

ax.tick_params(axis='x', labelsize=10, rotation=45)
ax.tick_params(axis='y', labelsize=10)

for container in ax.containers:
    ax.bar_label(container, fmt='%.1f', label_type='edge', fontsize=8, padding=3)

min_val = results_df['ExecutionTime_sec'].min()
if min_val > 0:
    ax.set_ylim(bottom=min_val * 0.5)

# `tight_layout` will try to make space for the legend
plt.tight_layout()

output_filename = 'algorithms_execution_time_comparison_log.png'
# Using bbox_inches='tight' is a robust way to ensure the saved file includes the external legend
plt.savefig(output_filename, dpi=300, bbox_inches='tight')
plt.show()

print(f"\nLog scale plot saved to file: {output_filename}")