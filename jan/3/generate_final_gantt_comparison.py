import csv
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.font_manager as fm
import datetime
import math
import collections
import os
from pathlib import Path
import numpy as np

# Configuration
STAGES = [
    "Combining", "Mixing", "Forming", "Proofing",
    "Baking", "Cooling",
]

# Stage translations and abbreviations
STAGE_TRANSLATIONS = {
    "Комбинирование": "Combining",
    "Смешивание": "Mixing", 
    "Формовка": "Forming",
    "Расстойка": "Proofing",
    "Выпекание": "Baking",
    "Остывание": "Cooling"
}

STAGE_ABBREVIATIONS = {
    "Combining": "C",
    "Mixing": "M", 
    "Forming": "F",
    "Proofing": "P",
    "Baking": "B",
    "Cooling": "Cool"
}

# CSV files to process
CSV_FILES = [
    'dqn_production_schedule.csv',
    'ga_production_schedule1.csv', 
    'production_schedule_v2.csv'
]

# Algorithm names for titles
ALGORITHM_NAMES = {
    'dqn_production_schedule.csv': 'DQN Algorithm',
    'ga_production_schedule1.csv': 'Genetic Algorithm',
    'production_schedule_v2.csv': 'Baseline Schedule'
}

def setup_directories():
    """Create necessary directories"""
    script_dir = Path(__file__).parent if '__file__' in globals() else Path.cwd()
    
    # Create output directories
    charts_dir = script_dir / 'gantt_charts'
    comparison_dir = script_dir / 'comparison_charts'
    
    charts_dir.mkdir(exist_ok=True)
    comparison_dir.mkdir(exist_ok=True)
    
    return script_dir, charts_dir, comparison_dir

def read_schedule_from_csv(filename):
    """Read schedule data from CSV file"""
    schedule_data = []
    max_end_time = 0.0
    
    if not os.path.exists(filename):
        print(f"Error: File '{filename}' not found.")
        return None, 0
    
    try:
        with open(filename, 'r', newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            print(f"Reading data from file: '{filename}'...")
            
            required_columns = ["Batch_ID", "Stage", "Start_Time_Min", "End_Time_Min"]
            if not all(col in reader.fieldnames for col in required_columns):
                missing = [col for col in required_columns if col not in reader.fieldnames]
                print(f"Error: Missing required columns in CSV: {missing}")
                return None, 0
            
            line_num = 1
            for row in reader:
                line_num += 1
                try:
                    start_time = int(row['Start_Time_Min'])
                    end_time = int(row['End_Time_Min'])
                    duration = end_time - start_time
                    
                    if start_time < 0 or end_time < 0 or duration < 0:
                        print(f"Warning: Invalid time values in line {line_num}. Skipping: {row}")
                        continue
                    
                    # Translate stage name
                    stage = STAGE_TRANSLATIONS.get(row['Stage'], row['Stage'])
                    
                    task_data = {
                        'Batch': row['Batch_ID'], 
                        'Stage': stage,
                        'Start': start_time, 
                        'End': end_time, 
                        'Duration': duration
                    }
                    schedule_data.append(task_data)
                    
                    if end_time > max_end_time:
                        max_end_time = end_time
                        
                except Exception as e:
                    print(f"Error processing line {line_num}: {e}. Skipping: {row}")
            
            if not schedule_data:
                print("Warning: No valid task data found.")
                return None, 0
            
            print(f"Data successfully read. Tasks: {len(schedule_data)}. Makespan: {max_end_time} min.")
            return schedule_data, float(max_end_time)
            
    except Exception as e:
        print(f"Error reading file '{filename}': {e}")
        return None, 0

def create_compact_gantt_chart(schedule_results, makespan_minutes, title, filename, charts_dir):
    """Create a compact Gantt chart suitable for scientific papers"""
    if not schedule_results or makespan_minutes <= 0:
        print("No data for visualization or invalid makespan.")
        return False

    # Font setup for scientific publications
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'figure.titlesize': 13
    })

    # Group tasks by batch and get unique product types
    tasks_by_batch = collections.defaultdict(list)
    product_types = set()
    
    for task in schedule_results:
        tasks_by_batch[task['Batch']].append(task)
        # Extract product type from batch name
        if '_batch_' in task['Batch']:
            product_type = task['Batch'].split('_batch_')[0]
            product_types.add(product_type)

    # Limit to first 15 batches for readability
    batch_start_times = {}
    for batch_id, tasks in tasks_by_batch.items():
        if tasks:
            min_start = min(task['Start'] for task in tasks)
            batch_start_times[batch_id] = min_start

    # Sort and limit batches
    sorted_batches = sorted(tasks_by_batch.keys(), 
                           key=lambda b: (batch_start_times.get(b, float('inf')), b))
    
    # Limit to reasonable number for readability
    max_batches = 20
    if len(sorted_batches) > max_batches:
        sorted_batches = sorted_batches[:max_batches]
        print(f"Limiting visualization to first {max_batches} batches for readability")

    num_batches = len(sorted_batches)
    batch_to_y = {batch: i for i, batch in enumerate(sorted_batches)}

    # Scientific color palette
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    stage_colors = {stage: colors[i % len(colors)] for i, stage in enumerate(STAGES)}

    # Create compact figure
    fig, ax = plt.subplots(figsize=(12, max(6, num_batches * 0.25 + 2)))

    # Plot tasks
    for batch_name in sorted_batches:
        y_pos = batch_to_y[batch_name]
        sorted_tasks = sorted(tasks_by_batch[batch_name], key=lambda t: t['Start'])

        for task in sorted_tasks:
            stage = task['Stage']
            start = task['Start']
            duration = task['Duration']
            
            if duration <= 0:
                continue
                
            color = stage_colors.get(stage, 'gray')
            
            # Create bar with pattern for better distinction
            bar = ax.barh(y=y_pos, width=duration, left=start, height=0.8, 
                         align='center', color=color, edgecolor='black', 
                         linewidth=0.5, alpha=0.8)

            # Add stage abbreviation for longer tasks
            if duration > makespan_minutes / 30:
                stage_abbr = STAGE_ABBREVIATIONS.get(stage, stage[:2])
                # Choose text color based on background
                text_color = 'white' if sum(plt.colors.to_rgb(color)) < 1.5 else 'black'
                ax.text(start + duration / 2, y_pos, stage_abbr, 
                       ha='center', va='center', color=text_color,
                       fontsize=8, weight='bold')

    # Create compact batch labels
    batch_labels = []
    for batch in sorted_batches:
        if '_batch_' in batch:
            product, batch_num = batch.split('_batch_')
            # Shorten product names
            product_short = {
                'Бородинский': 'B',
                'Формовой': 'F', 
                'Сэндвич': 'S'
            }.get(product, product[0])
            batch_labels.append(f"{product_short}{batch_num}")
        else:
            batch_labels.append(batch[:6])

    # Setup axes
    ax.set_yticks(range(num_batches))
    ax.set_yticklabels(batch_labels)
    ax.invert_yaxis()

    ax.set_xlabel("Time (minutes)")
    ax.set_ylabel("Batch ID")

    # Set reasonable x-axis limits
    ax.set_xlim(0, makespan_minutes * 1.05)
    
    # Add subtle grid
    ax.grid(True, axis='x', linestyle=':', alpha=0.5)

    # Compact title
    ax.set_title(f"{title}\nMakespan: {makespan_minutes:.0f} min", pad=10)

    # Create compact legend
    legend_elements = []
    stages_in_results = sorted(set(task['Stage'] for task in schedule_results))
    
    for stage in stages_in_results:
        if stage in stage_colors:
            color = stage_colors[stage]
            legend_elements.append(mpatches.Patch(color=color, label=stage))

    if legend_elements:
        ax.legend(handles=legend_elements, loc='upper right', 
                 bbox_to_anchor=(1.0, 1.0), frameon=True, fancybox=False)

    plt.tight_layout()
    
    # Save with high quality for publications
    output_path = charts_dir / filename
    plt.savefig(output_path, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    print(f"Chart saved: {output_path}")
    plt.close()
    
    return True

def create_publication_comparison(all_schedules, comparison_dir):
    """Create a publication-ready comparison chart"""
    if len(all_schedules) < 2:
        print("Need at least 2 schedules for comparison")
        return

    # Scientific publication style
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'figure.titlesize': 14
    })
    
    fig, axes = plt.subplots(len(all_schedules), 1, 
                            figsize=(14, 3 * len(all_schedules)))
    if len(all_schedules) == 1:
        axes = [axes]
    
    # Scientific color palette
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    stage_colors = {stage: colors[i % len(colors)] for i, stage in enumerate(STAGES)}
    
    max_makespan = max(data['makespan'] for data in all_schedules.values())
    
    for idx, (filename, data) in enumerate(all_schedules.items()):
        ax = axes[idx]
        schedule_results = data['schedule']
        makespan_minutes = data['makespan']
        algorithm_name = ALGORITHM_NAMES.get(filename, filename)
        
        # Group and limit batches
        tasks_by_batch = collections.defaultdict(list)
        for task in schedule_results:
            tasks_by_batch[task['Batch']].append(task)
        
        batch_start_times = {}
        for batch_id, tasks in tasks_by_batch.items():
            if tasks:
                batch_start_times[batch_id] = min(task['Start'] for task in tasks)
        
        sorted_batches = sorted(tasks_by_batch.keys(), 
                               key=lambda b: (batch_start_times.get(b, float('inf')), b))
        
        # Limit batches for comparison readability
        max_batches = 15
        if len(sorted_batches) > max_batches:
            sorted_batches = sorted_batches[:max_batches]
        
        # Plot tasks
        for batch_idx, batch_name in enumerate(sorted_batches):
            sorted_tasks = sorted(tasks_by_batch[batch_name], key=lambda t: t['Start'])
            
            for task in sorted_tasks:
                color = stage_colors.get(task['Stage'], 'gray')
                ax.barh(y=batch_idx, width=task['Duration'], left=task['Start'], 
                       height=0.7, align='center', color=color, 
                       edgecolor='black', linewidth=0.3, alpha=0.8)
        
        # Compact batch labels
        batch_labels = []
        for batch in sorted_batches:
            if '_batch_' in batch:
                product, batch_num = batch.split('_batch_')
                product_short = {
                    'Бородинский': 'B',
                    'Формовой': 'F',
                    'Сэндвич': 'S'
                }.get(product, product[0])
                batch_labels.append(f"{product_short}{batch_num}")
            else:
                batch_labels.append(batch[:4])
        
        # Setup axes
        ax.set_yticks(range(len(sorted_batches)))
        ax.set_yticklabels(batch_labels, fontsize=9)
        ax.invert_yaxis()
        
        ax.set_xlim(0, max_makespan)
        ax.grid(True, axis='x', linestyle=':', alpha=0.3)
        
        # Compact titles
        ax.text(0.02, 0.95, f"{algorithm_name} ({makespan_minutes:.0f} min)", 
                transform=ax.transAxes, fontsize=11, fontweight='bold',
                verticalalignment='top',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        # Only show x-label on bottom chart
        if idx == len(all_schedules) - 1:
            ax.set_xlabel("Time (minutes)")
        else:
            ax.set_xticklabels([])
        
        # Only show y-label on middle chart
        if idx == len(all_schedules) // 2:
            ax.set_ylabel("Batch ID")
    
    # Main title
    fig.suptitle("Production Schedule Algorithm Comparison", fontsize=14, fontweight='bold')
    
    # Single legend for all subplots
    legend_elements = []
    all_stages = set()
    for data in all_schedules.values():
        all_stages.update(task['Stage'] for task in data['schedule'])
    
    for stage in sorted(all_stages):
        if stage in stage_colors:
            color = stage_colors[stage]
            legend_elements.append(mpatches.Patch(color=color, label=stage))
    
    fig.legend(handles=legend_elements, loc='center right', 
              bbox_to_anchor=(0.98, 0.5), title="Production Stages")
    
    plt.tight_layout()
    plt.subplots_adjust(right=0.85, top=0.93)
    
    # Save for publication
    output_path = comparison_dir / 'algorithm_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    print(f"Publication comparison saved: {output_path}")
    plt.close()

def create_summary_table(all_schedules, comparison_dir):
    """Create a summary table for the paper"""
    if not all_schedules:
        return
    
    # Calculate metrics
    results = []
    for filename, data in all_schedules.items():
        algorithm = ALGORITHM_NAMES.get(filename, filename)
        makespan = data['makespan']
        num_batches = len(set(task['Batch'] for task in data['schedule']))
        
        results.append({
            'Algorithm': algorithm,
            'Makespan (min)': makespan,
            'Batches': num_batches
        })
    
    # Sort by makespan
    results.sort(key=lambda x: x['Makespan (min)'])
    
    # Create table figure
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare table data
    table_data = []
    best_makespan = results[0]['Makespan (min)']
    
    for i, result in enumerate(results):
        makespan = result['Makespan (min)']
        improvement = ((makespan - best_makespan) / best_makespan * 100) if makespan != best_makespan else 0
        
        table_data.append([
            result['Algorithm'],
            f"{makespan:.1f}",
            str(result['Batches']),
            f"{improvement:+.1f}%" if improvement != 0 else "Best"
        ])
    
    # Create table
    table = ax.table(cellText=table_data,
                    colLabels=['Algorithm', 'Makespan (min)', 'Batches', 'vs Best (%)'],
                    cellLoc='center',
                    loc='center',
                    bbox=[0.1, 0.3, 0.8, 0.4])
    
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 2)
    
    # Style the table
    for i in range(len(table_data) + 1):
        for j in range(4):
            cell = table[(i, j)]
            if i == 0:  # Header
                cell.set_facecolor('#40466e')
                cell.set_text_props(weight='bold', color='white')
            elif i == 1:  # Best result
                cell.set_facecolor('#90EE90')
            else:
                cell.set_facecolor('#f0f0f0')
    
    plt.title('Algorithm Performance Summary', fontsize=14, fontweight='bold', pad=20)
    
    # Save table
    output_path = comparison_dir / 'performance_summary.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    print(f"Performance summary saved: {output_path}")
    plt.close()

def main():
    """Main execution function"""
    # Setup directories
    script_dir, charts_dir, comparison_dir = setup_directories()
    
    print("=== Scientific Publication Gantt Chart Generator ===")
    print(f"Script directory: {script_dir}")
    print(f"Charts output: {charts_dir}")
    print(f"Comparison output: {comparison_dir}")
    
    all_schedules = {}
    
    # Process each CSV file
    for csv_file in CSV_FILES:
        csv_path = script_dir / csv_file
        
        if not csv_path.exists():
            print(f"Warning: {csv_file} not found, skipping...")
            continue
        
        print(f"\n--- Processing {csv_file} ---")
        schedule_data, makespan = read_schedule_from_csv(csv_path)
        
        if schedule_data and makespan > 0:
            # Store data for comparison
            all_schedules[csv_file] = {
                'schedule': schedule_data,
                'makespan': makespan
            }
            
            # Create individual compact chart
            algorithm_name = ALGORITHM_NAMES.get(csv_file, csv_file)
            chart_filename = f"{csv_file.replace('.csv', '')}_compact.png"
            
            success = create_compact_gantt_chart(
                schedule_data, makespan, algorithm_name, 
                chart_filename, charts_dir
            )
            
            if success:
                print(f"✓ Compact chart created for {csv_file}")
            else:
                print(f"✗ Failed to create chart for {csv_file}")
        else:
            print(f"✗ Failed to process {csv_file}")
    
    # Create publication-ready comparison charts
    if len(all_schedules) > 1:
        print(f"\n--- Creating Publication Charts ---")
        create_publication_comparison(all_schedules, comparison_dir)
        create_summary_table(all_schedules, comparison_dir)
        print("✓ Publication charts created")
    else:
        print("Need at least 2 valid schedules for comparison")
    
    print(f"\n=== Summary ===")
    print(f"Compact individual charts: {len(all_schedules)} created")
    print(f"Publication comparison: {'Created' if len(all_schedules) > 1 else 'Skipped'}")
    
    # Print results summary
    if all_schedules:
        print("\nMakespan Results:")
        sorted_results = sorted(all_schedules.items(), key=lambda x: x[1]['makespan'])
        for csv_file, data in sorted_results:
            algorithm = ALGORITHM_NAMES.get(csv_file, csv_file)
            print(f"  {algorithm}: {data['makespan']:.1f} minutes")

if __name__ == "__main__":
    main()