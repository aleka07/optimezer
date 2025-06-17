import csv
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import datetime
import math
import collections
import os

# --- 1. CONFIGURATION ---

# IMPORTANT: Set the folder where your source CSV files are located.
# The script will look for the CSV files inside this directory.
DATA_SOURCE_FOLDER = 'jan/7' 

# The name of the directory where the generated charts will be saved.
# IMPORTANT: This folder will be created INSIDE the DATA_SOURCE_FOLDER.
CHART_OUTPUT_DIR_NAME = 'gantt_chartss'

# List of CSV files to process from the DATA_SOURCE_FOLDER
INPUT_FILES = [
    'production_schedule_v2.csv',
    'ga_production_schedule1.csv',
    'dqn_production_schedule.csv'
]

# --- Color and Legend Configuration ---
# The stage names here MUST EXACTLY MATCH the names in your CSV files.
# Since your CSVs use Russian, these are in Russian to ensure colors are applied correctly.
STAGES_ORDER = [
    "Комбинирование", "Смешивание", "Формовка", "Расстойка",
    "Выпекание", "Остывание",
]

# Visualization settings
USE_HOURS_FOR_X_AXIS_IF_LONG = True
HOUR_THRESHOLD_MIN = 180  # 3 hours

# --- 2. Data Reading Function ---
def read_schedule_from_csv(filename):
    """Reads schedule data from a given CSV file path."""
    if not os.path.exists(filename):
        print(f"Error: File not found at '{filename}'")
        return None, 0
    
    schedule_data = []
    max_end_time = 0.0
    try:
        with open(filename, 'r', newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            required_columns = ["Batch_ID", "Stage", "Start_Time_Min", "End_Time_Min"]
            if not all(col in reader.fieldnames for col in required_columns):
                missing = [col for col in required_columns if col not in reader.fieldnames]
                print(f"Error: CSV '{filename}' is missing required columns: {missing}")
                return None, 0
            
            for row in reader:
                try:
                    task_data = {
                        'Batch': row['Batch_ID'], 'Stage': row['Stage'],
                        'Start': int(row['Start_Time_Min']), 'End': int(row['End_Time_Min'])
                    }
                    task_data['Duration'] = task_data['End'] - task_data['Start']
                    schedule_data.append(task_data)
                    if task_data['End'] > max_end_time:
                        max_end_time = task_data['End']
                except (ValueError, KeyError) as e:
                    print(f"Warning: Skipping row due to data error in '{filename}': {row}, {e}")
        return schedule_data, float(max_end_time)
    except Exception as e:
        print(f"Critical error reading file '{filename}': {e}")
        return None, 0

# --- 3. Visualization Function ---
def plot_gantt_chart(schedule_results, makespan_minutes, stages_order, source_filepath):
    """Creates and saves a Gantt chart with improved readability and sizing."""
    if not schedule_results or makespan_minutes <= 0:
        print("No valid data available to generate a chart.")
        return

    # --- Prepare data for sorting and plotting ---
    tasks_by_batch = collections.defaultdict(list)
    for task in schedule_results:
        tasks_by_batch[task['Batch']].append(task)
    
    batch_start_times = {
        batch_id: min(t['Start'] for t in tasks) if tasks else float('inf')
        for batch_id, tasks in tasks_by_batch.items()
    }
    sorted_batches = sorted(tasks_by_batch.keys(), key=lambda b: (batch_start_times[b], b))
    num_batches = len(sorted_batches)
    batch_to_y = {batch: i for i, batch in enumerate(sorted_batches)}

    # --- Setup colors ---
    cmap = plt.get_cmap('tab20')
    stage_colors = {stage: cmap(i % cmap.N) for i, stage in enumerate(stages_order)}
    stage_colors_with_default = collections.defaultdict(lambda: 'grey', stage_colors)

    # --- Figure size calculation for consistent output ---
    fig_width = 15
    fig_height = max(8, num_batches * 0.3)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    # --- Determine time units for X-axis ---
    use_hours = USE_HOURS_FOR_X_AXIS_IF_LONG and makespan_minutes > HOUR_THRESHOLD_MIN
    time_divisor = 60.0 if use_hours else 1.0
    time_unit_label = "Time (hours)" if use_hours else "Time (minutes)"
    makespan_display = makespan_minutes / time_divisor

    # --- Draw task bars ---
    for batch_name in sorted_batches:
        y_pos = batch_to_y[batch_name]
        for task in tasks_by_batch[batch_name]:
            start, duration = task['Start'] / time_divisor, task['Duration'] / time_divisor
            if duration <= 0: continue
            color = stage_colors_with_default[task['Stage']]
            ax.barh(y=y_pos, width=duration, left=start, height=0.6, align='center',
                    color=color, edgecolor='black', linewidth=0.5)

    # --- Configure axes and labels ---
    ax.set_yticks(range(num_batches))
    if num_batches > 50:
        tick_labels = [b if i % 5 == 0 else "" for i, b in enumerate(sorted_batches)]
        ax.set_yticklabels(tick_labels, fontsize=8)
    else:
        ax.set_yticklabels(sorted_batches, fontsize=9)

    ax.invert_yaxis()
    ax.set_xlabel(time_unit_label, fontsize=12)
    ax.set_ylabel("Production Batch", fontsize=12)
    ax.set_xlim(0, math.ceil(makespan_display))
    ax.grid(True, which='major', axis='x', linestyle='--', color='gray', alpha=0.7)

    # --- Title ---
    base_name = os.path.basename(source_filepath)
    makespan_formatted = str(datetime.timedelta(minutes=makespan_minutes))
    ax.set_title(
        f"Gantt Chart for: {base_name}\n"
        f"Total Makespan: {makespan_minutes:.1f} min ({makespan_formatted})",
        fontsize=14, pad=20, weight='bold'
    )

    # --- Legend ---
    legend_patches = [mpatches.Patch(color=color, label=stage) for stage, color in stage_colors.items()]
    ax.legend(handles=legend_patches, title="Stages", bbox_to_anchor=(1.01, 1),
              loc='upper left', fontsize=10, title_fontsize=12)
    
    # --- NEW: Save the chart to a dedicated folder inside the data source directory ---
    source_dir = os.path.dirname(source_filepath)
    output_dir_path = os.path.join(source_dir, CHART_OUTPUT_DIR_NAME)
    os.makedirs(output_dir_path, exist_ok=True)
    
    output_filename = f"gantt_{os.path.splitext(base_name)[0]}.png"
    final_output_path = os.path.join(output_dir_path, output_filename)
    
    plt.savefig(final_output_path, dpi=300, bbox_inches='tight')
    print(f"Chart successfully saved to: {final_output_path}")
    plt.close(fig)

# --- 4. Main Execution Block ---
if __name__ == "__main__":
    if not os.path.isdir(DATA_SOURCE_FOLDER):
        print(f"Error: The specified data source folder does not exist: '{DATA_SOURCE_FOLDER}'")
        exit()

    for file_name in INPUT_FILES:
        full_path = os.path.join(DATA_SOURCE_FOLDER, file_name)
        print(f"\n--- Processing file: {full_path} ---")
        
        schedule_data, makespan = read_schedule_from_csv(full_path)
        
        if schedule_data and makespan > 0:
            print("Visualizing schedule...")
            plot_gantt_chart(schedule_data, makespan, STAGES_ORDER, full_path)
        else:
            print(f"Skipping visualization for {file_name} due to missing data or errors.")
    
    print("\n--- All files processed. ---")