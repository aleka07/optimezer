import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns
import numpy as np

# --- 1. CONFIGURATION ---
FILES_TO_PROCESS = {
    'DQN Algorithm': 'dqn_production_schedule.csv',
    'Genetic Algorithm': 'ga_production_schedule1.csv',
    'CP-SAT Solver': 'production_schedule_v2.csv'
}
OUTPUT_DIR = 'gantt_charts_comparison'

# --- 2. SCRIPT LOGIC ---

def plot_gantt_on_ax(ax, df, color_map, product_order):
    """
    Helper function to plot a single, well-formatted Gantt chart on a given Matplotlib axis.
    This function contains the core logic for creating the "mini-lanes" for each batch.
    """
    # Create a mapping from product type string to a numerical Y-axis index
    product_to_y_index = {product: i for i, product in enumerate(product_order)}
    
    # --- KEY LOGIC: CALCULATE LANES FOR BATCHES ---
    # Find the maximum batch number for each product to determine lane spacing
    max_batches_per_product = df.groupby('Product_Type')['Batch_Number'].max().fillna(1)
    
    # Total height allocated for each product row on the Y-axis
    total_row_height = 0.8 

    for _, task in df.iterrows():
        product_type = task['Product_Type']
        base_y = product_to_y_index[product_type]
        batch_num = task['Batch_Number']
        
        # Determine how many lanes are needed for this product
        num_lanes = max_batches_per_product.get(product_type, 1)
        
        # Calculate the height of a single "mini-lane" for one batch
        lane_height = total_row_height / num_lanes
        
        # Calculate the Y-position for this specific batch's lane
        # This centers the group of lanes around the product's base Y-position
        y_pos = (base_y - total_row_height / 2) + (batch_num - 0.5) * lane_height

        ax.barh(
            y=y_pos,
            width=task['Duration_Min'],
            left=task['Start_Time_Min'],
            edgecolor='black',
            linewidth=0.5,
            height=lane_height * 0.9,  # 90% of lane height to create a small gap
            color=color_map.get(task['Stage'], 'gray'),
            label=task['Stage']
        )
    
    # Configure the Y-axis to show product names instead of numbers
    ax.set_yticks(np.arange(len(product_order)))
    ax.set_yticklabels(product_order)

def create_gantt_charts():
    """Main function to generate and save all Gantt charts."""
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
                print(f"!!! ERROR in '{filename}': Missing required columns. Skipping.")
                continue

            # --- KEY FIX 1: Extract Product Type AND Batch Number ---
            df['Product_Type'] = df['Batch_ID'].str.split('_batch_').str[0]
            df['Batch_Number'] = df['Batch_ID'].str.extract(r'_batch_(\d+)').astype(int).fillna(1)
            
            df['Duration_Min'] = df['End_Time_Min'] - df['Start_Time_Min']
            makespan = df['End_Time_Min'].max()
            makespans[alg_name] = makespan
            schedules_data[alg_name] = df
            print(f"  - Loaded '{filename}'. Makespan: {makespan:.2f} min.")
            
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

    # --- VISUALIZATION SETUP ---
    all_product_types = pd.concat(schedules_data.values())['Product_Type'].unique()
    product_order = sorted(list(all_product_types), reverse=True)

    all_stages = pd.concat(schedules_data.values())['Stage'].unique()
    # Use a color palette that is friendly to colorblindness
    colors = sns.color_palette('tab10', n_colors=len(all_stages))
    color_map = {stage: colors[i] for i, stage in enumerate(all_stages)}

    # --- GENERATE CHARTS ---
    # 1. Combined Chart
    num_algs = len(schedules_data)
    fig, axes = plt.subplots(num_algs, 1, figsize=(20, 15), sharex=True, sharey=True)
    fig.suptitle('Gantt Chart Comparison of Scheduling Algorithms', fontsize=20, weight='bold')

    for i, (alg_name, df) in enumerate(schedules_data.items()):
        ax = axes[i]
        plot_gantt_on_ax(ax, df, color_map, product_order)
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
    output_path = os.path.join(OUTPUT_DIR, 'gantt_chart_comparison_COMBINED_v3.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"\n- Combined comparison chart saved to: {output_path}")

    # 2. Separate Charts
    print("- Generating separate charts...")
    for alg_name, df in schedules_data.items():
        fig, ax = plt.subplots(figsize=(20, 10))
        plot_gantt_on_ax(ax, df, color_map, product_order)
        makespan = df['End_Time_Min'].max()
        ax.axvline(x=makespan, color='red', linestyle='--', linewidth=2, label=f'Makespan: {makespan:.2f} min')
        ax.set_title(f'Production Schedule: {alg_name}', fontsize=18, weight='bold')
        ax.set_xlabel('Time (Minutes)', fontsize=14)
        ax.set_ylabel('Product Type', fontsize=14)
        ax.grid(axis='x', linestyle=':', color='gray')
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), title='Stage/Event', bbox_to_anchor=(1.01, 1), loc='upper left')
        filename = f"gantt_chart_{alg_name.replace(' ', '_')}_v3.png"
        output_path = os.path.join(OUTPUT_DIR, filename)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"  - Saved: {output_path}")

if __name__ == '__main__':
    create_gantt_charts()