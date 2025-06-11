import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns

# --- 1. CONFIGURATION ---
FILES_TO_PROCESS = {
    'DQN Algorithm': 'dqn_production_schedule.csv',
    'Genetic Algorithm': 'ga_production_schedule1.csv',
    'CP-SAT Solver': 'production_schedule_v2.csv'
}
OUTPUT_DIR = 'gantt_charts_comparison'

# --- 2. SCRIPT LOGIC ---

def create_gantt_charts():
    """
    Reads schedule data, groups tasks by product type, calculates makespans,
    and generates readable, aligned Gantt charts for comparison.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Output directory '{OUTPUT_DIR}' is ready.")

    schedules_data = {}
    makespans = {}

    print("\n--- Loading and Processing Schedule Data ---")
    for alg_name, filename in FILES_TO_PROCESS.items():
        try:
            df = pd.read_csv(filename)
            required_cols = {'Batch_ID', 'Stage', 'Start_Time_Min', 'End_Time_Min'}
            if not required_cols.issubset(df.columns):
                print(f"!!! ERROR: File '{filename}' is missing required columns. Skipping.")
                continue

            # --- KEY CHANGE 1: Extract Product Type from Batch ID ---
            # This is the core fix for the unreadable Y-axis.
            # We split "Product_Name_batch_1" and take the "Product_Name" part.
            df['Product_Type'] = df['Batch_ID'].str.split('_batch_').str[0]
            
            df['Duration_Min'] = df['End_Time_Min'] - df['Start_Time_Min']
            makespan = df['End_Time_Min'].max()
            makespans[alg_name] = makespan
            schedules_data[alg_name] = df
            print(f"  - Successfully loaded '{filename}'. Makespan: {makespan:.2f} min.")
            
        except FileNotFoundError:
            print(f"!!! WARNING: File '{filename}' not found. Skipping.")
            continue

    if not schedules_data:
        print("\nNo data was loaded. Exiting script.")
        return

    print("\n--- Quantitative Comparison (Makespan) ---")
    sorted_makespans = sorted(makespans.items(), key=lambda item: item[1])
    for i, (alg, time) in enumerate(sorted_makespans):
        print(f"{i+1}. {alg}: {time:.2f} minutes")

    # --- 4. VISUALIZATION ---
    
    # --- KEY CHANGE 2: Create a single, master list of all product types ---
    # This will be used to create a consistent Y-axis for all charts.
    all_product_types = pd.concat(schedules_data.values())['Product_Type'].unique()
    # Sort it so the order is logical (alphabetical) and consistent
    product_order = sorted(list(all_product_types), reverse=True) # reverse=True for top-down alphabetical

    all_stages = pd.concat(schedules_data.values())['Stage'].unique()
    colors = sns.color_palette('tab10', n_colors=len(all_stages))
    color_map = {stage: colors[i] for i, stage in enumerate(all_stages)}

    create_combined_gantt_chart(schedules_data, makespans, color_map, product_order)
    create_separate_gantt_charts(schedules_data, color_map, product_order)

    print(f"\nAll charts have been saved to the '{OUTPUT_DIR}' folder.")

def create_combined_gantt_chart(schedules_data, makespans, color_map, product_order):
    """Generates a single, aligned figure with a subplot for each algorithm."""
    num_algs = len(schedules_data)
    fig, axes = plt.subplots(num_algs, 1, figsize=(20, 12), sharex=True, sharey=True) # sharey=True is crucial
    fig.suptitle('Gantt Chart Comparison of Scheduling Algorithms', fontsize=20, weight='bold')

    for i, (alg_name, df) in enumerate(schedules_data.items()):
        ax = axes[i]
        
        # --- KEY CHANGE 3: Use Product_Type and the master product_order for the Y-axis ---
        df['Product_Type_cat'] = pd.Categorical(df['Product_Type'], categories=product_order, ordered=True)
        df = df.sort_values('Product_Type_cat')
        
        for _, task in df.iterrows():
            ax.barh(
                y=task['Product_Type'], # Plot on the Product_Type line
                width=task['Duration_Min'],
                left=task['Start_Time_Min'],
                edgecolor='black',
                height=0.6, # Make bars thinner to reduce overlap clutter
                color=color_map.get(task['Stage'], 'gray'),
                label=task['Stage']
            )

        makespan = makespans[alg_name]
        ax.axvline(x=makespan, color='red', linestyle='--', linewidth=2, label=f'Makespan: {makespan:.2f} min')
        ax.set_title(alg_name, fontsize=16, loc='left')
        ax.set_ylabel('Product Type', fontsize=12)
        ax.grid(axis='x', linestyle=':', color='gray')
        ax.tick_params(axis='y', labelsize=10)

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig.legend(by_label.values(), by_label.keys(), loc='upper right', fontsize=12, title='Stage/Event', title_fontsize=13)

    plt.xlabel('Time (Minutes)', fontsize=14)
    plt.tight_layout(rect=[0, 0, 0.9, 0.96])
    output_path = os.path.join(OUTPUT_DIR, 'gantt_chart_comparison_COMBINED_v2.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"\n- Combined comparison chart saved to: {output_path}")

def create_separate_gantt_charts(schedules_data, color_map, product_order):
    """Generates a separate, readable Gantt chart for each algorithm."""
    print("- Generating separate charts...")
    for alg_name, df in schedules_data.items():
        fig, ax = plt.subplots(figsize=(20, 10))

        df['Product_Type_cat'] = pd.Categorical(df['Product_Type'], categories=product_order, ordered=True)
        df = df.sort_values('Product_Type_cat')
        
        for _, task in df.iterrows():
            ax.barh(
                y=task['Product_Type'],
                width=task['Duration_Min'],
                left=task['Start_Time_Min'],
                edgecolor='black',
                height=0.6,
                color=color_map.get(task['Stage'], 'gray'),
                label=task['Stage']
            )
        
        makespan = df['End_Time_Min'].max()
        ax.axvline(x=makespan, color='red', linestyle='--', linewidth=2, label=f'Makespan: {makespan:.2f} min')

        ax.set_title(f'Production Schedule: {alg_name}', fontsize=18, weight='bold')
        ax.set_xlabel('Time (Minutes)', fontsize=14)
        ax.set_ylabel('Product Type', fontsize=14)
        ax.grid(axis='x', linestyle=':', color='gray')

        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), title='Stage/Event', bbox_to_anchor=(1.01, 1), loc='upper left')
        
        filename = f"gantt_chart_{alg_name.replace(' ', '_')}_v2.png"
        output_path = os.path.join(OUTPUT_DIR, filename)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"  - Saved: {output_path}")

if __name__ == '__main__':
    create_gantt_charts()