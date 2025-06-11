import csv
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.font_manager as fm
import datetime
import math
import collections
import os
import re

# --- 1. CONFIGURATION ---

# Find the directory where this script is located
try:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    # Fallback for interactive environments (like Jupyter)
    SCRIPT_DIR = os.getcwd()

# Define the output directory for the charts
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'gantt_charts')

# Dictionary mapping friendly algorithm names to their schedule files
SCHEDULE_FILES = {
    'CP-SAT': 'production_schedule_v2.csv',
    'Genetic Algorithm': 'ga_production_schedule1.csv',
    'DQN': 'dqn_production_schedule.csv'
}

# Define the order and colors for production stages
STAGES_ORDER = [
    "Комбинирование", "Смешивание", "Формовка", "Расстойка",
    "Выпекание", "Остывание",
]

# English translation for the legend
STAGES_LEGEND_EN = {
    "Комбинирование": "Combining",
    "Смешивание": "Mixing",
    "Формовка": "Moulding",
    "Расстойка": "Proofing",
    "Выпекание": "Baking",
    "Остывание": "Cooling"
}


# --- 2. DATA READING FUNCTION (in English) ---

def read_schedule_from_csv(filename):
    """Reads schedule data from a given CSV file."""
    schedule_data = []
    max_end_time = 0.0
    
    if not os.path.exists(filename):
        print(f"Error: File '{os.path.basename(filename)}' not found in script directory.")
        return None, 0
        
    try:
        with open(filename, 'r', newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            required_cols = ["Batch_ID", "Stage", "Start_Time_Min", "End_Time_Min"]
            if not all(col in reader.fieldnames for col in required_cols):
                missing = [col for col in required_cols if col not in reader.fieldnames]
                print(f"Error: CSV file '{os.path.basename(filename)}' is missing required columns: {missing}")
                return None, 0

            for i, row in enumerate(reader, 1):
                try:
                    task = {
                        'Batch': row['Batch_ID'],
                        'Stage': row['Stage'],
                        'Start': int(row['Start_Time_Min']),
                        'End': int(row['End_Time_Min'])
                    }
                    task['Duration'] = task['End'] - task['Start']
                    
                    if task['Start'] < 0 or task['Duration'] < 0:
                        print(f"Warning: Invalid time values in row {i}. Skipping: {row}")
                        continue
                        
                    schedule_data.append(task)
                    if task['End'] > max_end_time:
                        max_end_time = task['End']
                except (ValueError, KeyError) as e:
                    print(f"Warning: Error processing row {i}: {e}. Skipping: {row}")
        
        if not schedule_data:
            print("Warning: No valid task data was found in the file.")
            return None, 0
            
        return schedule_data, float(max_end_time)
        
    except Exception as e:
        print(f"Fatal Error reading file '{os.path.basename(filename)}': {e}")
        return None, 0


# --- 3. GANTT CHART PLOTTING FUNCTION (MODIFIED) ---

def create_and_save_gantt_chart(schedule_results, makespan, algorithm_name, output_path):
    """
    Creates and saves a Gantt chart with improved readability.
    - Y-axis labels (batch names) are shortened.
    - Text inside bars is removed.
    - Saves the plot to a file instead of showing it.
    """
    if not schedule_results or makespan <= 0:
        print(f"No data to visualize for {algorithm_name}.")
        return

    # Group tasks by batch
    tasks_by_batch = collections.defaultdict(list)
    for task in schedule_results:
        tasks_by_batch[task['Batch']].append(task)
    
    # Calculate start time for each batch to sort the Y-axis
    batch_start_times = {batch: min(t['Start'] for t in tasks) for batch, tasks in tasks_by_batch.items() if tasks}
    
    # Sort batch names by their first task's start time, then alphabetically
    sorted_batches = sorted(tasks_by_batch.keys(), key=lambda b: (batch_start_times.get(b, float('inf')), b))
    
    num_batches = len(sorted_batches)
    batch_to_y = {batch: i for i, batch in enumerate(sorted_batches)}

    # --- KEY CHANGE: Shorten batch names for the Y-axis ---
    # Example: "Бородинский_batch_3" -> "Borodinsky 3"
    def shorten_name(name):
        name = name.replace('_batch_', ' ').replace('Бородинский', 'Borodinsky').replace('Формовой', 'Formovoy').replace('Сэндвич', 'Sandwich')
        # Remove any remaining non-alphanumeric characters except spaces
        name = re.sub(r'[^a-zA-Z0-9\s]', '', name)
        return name
        
    shortened_labels = [shorten_name(b) for b in sorted_batches]
    
    # Generate colors for stages
    cmap = plt.get_cmap('tab20')
    stage_colors = {stage: cmap(i % cmap.N) for i, stage in enumerate(STAGES_ORDER)}

    # --- Create the plot ---
    fig_height = max(8, num_batches * 0.35) # Dynamic height
    fig_width = max(15, makespan / 10) # Dynamic width
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    for batch_name in sorted_batches:
        y_pos = batch_to_y[batch_name]
        for task in sorted(tasks_by_batch[batch_name], key=lambda t: t['Start']):
            if task['Duration'] <= 0: continue
            color = stage_colors.get(task['Stage'], 'grey')
            ax.barh(y=y_pos, width=task['Duration'], left=task['Start'], height=0.6,
                    color=color, edgecolor='black', linewidth=0.5, alpha=0.9)
    
    # --- KEY CHANGE: The text inside the bars has been removed ---
    
    # --- Configure axes and labels ---
    ax.set_yticks(range(num_batches))
    # --- KEY CHANGE: Use shortened labels and smaller font ---
    ax.set_yticklabels(shortened_labels, fontsize=9)
    ax.invert_yaxis()

    ax.set_xlabel("Time (minutes)", fontsize=12)
    ax.set_ylabel("Product Batch (sorted by start time)", fontsize=12)
    ax.set_xlim(0, math.ceil(makespan))
    ax.grid(axis='x', linestyle='--', color='gray', alpha=0.6)

    # --- Title and Legend ---
    total_time = datetime.timedelta(minutes=makespan)
    title_str = f"Gantt Chart for {algorithm_name} Schedule\nMakespan: {makespan:.1f} minutes ({str(total_time)})"
    ax.set_title(title_str, fontsize=16, fontweight='bold', pad=20)
    
    legend_patches = []
    # Create legend based on stages present in the data, using English names
    stages_in_data = sorted(list(set(t['Stage'] for t in schedule_results)),
                            key=lambda s: STAGES_ORDER.index(s) if s in STAGES_ORDER else float('inf'))
                            
    for stage in stages_in_data:
        label = STAGES_LEGEND_EN.get(stage, stage) # Get English label
        color = stage_colors.get(stage, 'grey')
        legend_patches.append(mpatches.Patch(color=color, label=label))
    
    ax.legend(handles=legend_patches, title="Production Stages",
              bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=10)

    # --- KEY CHANGE: Save the figure ---
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig) # Close the figure to free memory
    print(f"Successfully saved chart to: {os.path.basename(output_path)}")


# --- 4. MAIN EXECUTION BLOCK ---

if __name__ == "__main__":
    # Create the output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Charts will be saved to: '{OUTPUT_DIR}'\n")

    # Loop through all configured files
    for alg_name, csv_file in SCHEDULE_FILES.items():
        print(f"--- Processing: {alg_name} ({csv_file}) ---")
        
        # Construct full file paths
        input_csv_path = os.path.join(SCRIPT_DIR, csv_file)
        output_png_path = os.path.join(OUTPUT_DIR, f"{alg_name.lower().replace(' ', '_')}_gantt.png")
        
        # Read data
        schedule_data, makespan = read_schedule_from_csv(input_csv_path)

        # If data is valid, create and save the chart
        if schedule_data and makespan > 0:
            create_and_save_gantt_chart(
                schedule_results=schedule_data,
                makespan=makespan,
                algorithm_name=alg_name,
                output_path=output_png_path
            )
        else:
            print(f"Could not generate chart for {alg_name} due to data reading errors.\n")
        
        print("-" * 30 + "\n")