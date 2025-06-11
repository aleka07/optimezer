import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns

# --- 1. CONFIGURATION ---

# Map the friendly algorithm name to its schedule file
# Assuming production_schedule_v2.csv is from the CP-SAT solver
FILES_TO_PROCESS = {
    'DQN Algorithm': 'dqn_production_schedule.csv',
    'Genetic Algorithm': 'ga_production_schedule1.csv',
    'CP-SAT Solver': 'production_schedule_v2.csv'
}

# Directory to save the output charts
OUTPUT_DIR = 'gantt_charts_comparison'

# --- 2. SCRIPT LOGIC ---

def create_gantt_charts():
    """
    Reads schedule data from CSVs, calculates makespans, and generates
    both a combined and separate Gantt charts for comparison.
    """
    # Create the output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Output directory '{OUTPUT_DIR}' is ready.")

    schedules_data = {}
    makespans = {}

    # Load and process data for each algorithm
    print("\n--- Loading and Processing Schedule Data ---")
    for alg_name, filename in FILES_TO_PROCESS.items():
        try:
            df = pd.read_csv(filename)
            
            # --- Data Validation ---
            required_cols = {'Batch_ID', 'Stage', 'Start_Time_Min', 'End_Time_Min'}
            if not required_cols.issubset(df.columns):
                print(f"!!! ERROR: File '{filename}' is missing required columns. Skipping.")
                print(f"    Required: {required_cols}")
                print(f"    Found: {set(df.columns)}")
                continue

            # Calculate duration if not present
            if 'Duration_Min' not in df.columns:
                df['Duration_Min'] = df['End_Time_Min'] - df['Start_Time_Min']

            # Calculate makespan
            makespan = df['End_Time_Min'].max()
            makespans[alg_name] = makespan
            
            # Store the dataframe
            schedules_data[alg_name] = df
            
            print(f"  - Successfully loaded '{filename}' for '{alg_name}'. Makespan: {makespan:.2f} min.")
            
        except FileNotFoundError:
            print(f"!!! WARNING: File '{filename}' not found. Skipping.")
            continue

    if not schedules_data:
        print("\nNo data was loaded. Exiting script.")
        return

    # --- 3. QUANTITATIVE COMPARISON ---
    print("\n--- Quantitative Comparison (Makespan) ---")
    sorted_makespans = sorted(makespans.items(), key=lambda item: item[1])
    for i, (alg, time) in enumerate(sorted_makespans):
        print(f"{i+1}. {alg}: {time:.2f} minutes")
    
    # --- 4. VISUALIZATION ---

    # Get a list of all unique stages to create a consistent color palette
    all_stages = pd.concat(schedules_data.values())['Stage'].unique()
    # Using a vibrant and distinct color palette
    colors = sns.color_palette('tab20', n_colors=len(all_stages))
    color_map = {stage: colors[i] for i, stage in enumerate(all_stages)}

    # A) Generate the COMBINED (FACETED) chart - The best for comparison
    create_combined_gantt_chart(schedules_data, makespans, color_map)

    # B) Generate SEPARATE charts for each algorithm
    create_separate_gantt_charts(schedules_data, color_map)

    print(f"\nAll charts have been saved to the '{OUTPUT_DIR}' folder.")


def create_combined_gantt_chart(schedules_data, makespans, color_map):
    """Generates a single figure with a subplot for each algorithm's Gantt chart."""
    
    num_algs = len(schedules_data)
    # Create subplots that share the X-axis for direct comparison
    fig, axes = plt.subplots(num_algs, 1, figsize=(18, 6 * num_algs), sharex=True)
    fig.suptitle('Gantt Chart Comparison of Scheduling Algorithms', fontsize=20, weight='bold')

    # Get a consistent order for batches across all plots, based on the first algorithm's schedule
    first_alg_df = list(schedules_data.values())[0]
    batch_order = first_alg_df.groupby('Batch_ID')['Start_Time_Min'].min().sort_values(ascending=False).index

    for i, (alg_name, df) in enumerate(schedules_data.items()):
        ax = axes[i]
        
        # Sort dataframe according to the defined batch order for consistent y-axis
        df['Batch_ID_cat'] = pd.Categorical(df['Batch_ID'], categories=batch_order, ordered=True)
        df = df.sort_values('Batch_ID_cat')
        
        for _, task in df.iterrows():
            ax.barh(
                y=task['Batch_ID'],
                width=task['Duration_Min'],
                left=task['Start_Time_Min'],
                edgecolor='black',
                color=color_map.get(task['Stage'], 'gray'),
                label=task['Stage']
            )

        # Add a vertical line to mark the makespan
        makespan = makespans[alg_name]
        ax.axvline(x=makespan, color='red', linestyle='--', linewidth=2, label=f'Makespan: {makespan:.2f} min')
        
        ax.set_title(alg_name, fontsize=16, loc='left')
        ax.set_ylabel('Batch ID', fontsize=12)
        ax.grid(axis='x', linestyle=':', color='gray')

    # Create a single, shared legend for the entire figure
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles)) # remove duplicate labels
    fig.legend(by_label.values(), by_label.keys(), loc='upper right', fontsize=12, title='Stage/Event', title_fontsize=13)

    plt.xlabel('Time (Minutes)', fontsize=14)
    plt.tight_layout(rect=[0, 0, 0.9, 0.96]) # Adjust layout to make space for suptitle and legend
    
    output_path = os.path.join(OUTPUT_DIR, 'gantt_chart_comparison_COMBINED.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"\n- Combined comparison chart saved to: {output_path}")


def create_separate_gantt_charts(schedules_data, color_map):
    """Generates and saves a separate Gantt chart for each algorithm."""
    print("- Generating separate charts...")
    for alg_name, df in schedules_data.items():
        fig, ax = plt.subplots(figsize=(18, 10))

        # Sort batches by their first start time for a logical layout
        batch_order = df.groupby('Batch_ID')['Start_Time_Min'].min().sort_values(ascending=False).index
        df['Batch_ID_cat'] = pd.Categorical(df['Batch_ID'], categories=batch_order, ordered=True)
        df = df.sort_values('Batch_ID_cat')
        
        for _, task in df.iterrows():
            ax.barh(
                y=task['Batch_ID'],
                width=task['Duration_Min'],
                left=task['Start_Time_Min'],
                edgecolor='black',
                color=color_map.get(task['Stage'], 'gray'),
                label=task['Stage']
            )
        
        makespan = df['End_Time_Min'].max()
        ax.axvline(x=makespan, color='red', linestyle='--', linewidth=2, label=f'Makespan: {makespan:.2f} min')

        ax.set_title(f'Production Schedule: {alg_name}', fontsize=18, weight='bold')
        ax.set_xlabel('Time (Minutes)', fontsize=14)
        ax.set_ylabel('Batch ID', fontsize=14)
        ax.grid(axis='x', linestyle=':', color='gray')

        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), title='Stage/Event', bbox_to_anchor=(1.01, 1), loc='upper left')
        
        filename = f"gantt_chart_{alg_name.replace(' ', '_')}.png"
        output_path = os.path.join(OUTPUT_DIR, filename)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"  - Saved: {output_path}")


if __name__ == '__main__':
    create_gantt_charts()