import os
import pandas as pd
import numpy as np
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as mticker

# --- НАСТРОЙКИ ---

BASE_DIR = 'jan'
DATES = ['3', '4', '6', '7', '8', '10']
ALGO_MAP = {
    'CP-SAT (OR-Tools)': 'production_schedule_v2.csv',
    'Genetic Algorithm (GA)': 'ga_production_schedule1.csv',
    'Deep Q-Network (DQN)': 'dqn_production_schedule.csv',
}
MACHINES_AVAILABLE = { 
    "Комбинирование": 2, "Смешивание": 3, "Формовка": 2, 
    "Расстойка": 8, "Выпекание": 6, "Остывание": 50 
}

# --- 1. ФУНКЦИЯ ЗАГРУЗКИ (без изменений) ---
def load_makespan_data(base_dir, dates, algo_map):
    all_data = []
    print("Чтение данных и вычисление makespan...")
    for date in dates:
        date_path = os.path.join(base_dir, date)
        if not os.path.isdir(date_path):
            print(f"Предупреждение: Папка {date_path} не найдена.")
            continue
        for algo_name, filename in algo_map.items():
            file_path = os.path.join(date_path, filename)
            makespan_min = 0
            if os.path.exists(file_path):
                try:
                    df = pd.read_csv(file_path)
                    if not df.empty and 'End_Time_Min' in df.columns:
                        makespan_min = df['End_Time_Min'].max()
                except Exception as e:
                    print(f"Ошибка чтения файла {file_path}: {e}")
            else:
                print(f"Предупреждение: Файл {file_path} не найден.")
            all_data.append({
                'date': f'Янв {date}',
                'algorithm': algo_name,
                'makespan_min': makespan_min,
                'makespan_hours': makespan_min / 60 if makespan_min > 0 else 0
            })
    print("Данные успешно загружены.")
    return pd.DataFrame(all_data)

# --- 2. ИСПРАВЛЕННАЯ ФУНКЦИЯ ДЛЯ СГЛАЖЕННОГО ГРАФИКА ---

def plot_makespan_comparison_linechart_smooth(df):
    """
    Строит УЛУЧШЕННЫЙ линейный график: сглаженный, сжатый и с акцентом на разницу.
    """
    print("Создание улучшенного графика Makespan (Линейный, сглаженный)...")
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 8))

    algorithms = df['algorithm'].unique()
    dates = df['date'].cat.categories # Получаем категории в правильном порядке

    palette = sns.color_palette('viridis', n_colors=len(algorithms))
    color_map = {algo: color for algo, color in zip(algorithms, palette)}
    marker_map = {algo: marker for algo, marker in zip(algorithms, ['o', 'X', 's'])}

    for algo in algorithms:
        algo_df = df[df['algorithm'] == algo]
        
        x_points = np.arange(len(algo_df))
        y_points = algo_df['makespan_hours'].values

        valid_indices = np.where(y_points > 0)[0]
        if len(valid_indices) < 2:
            ax.plot(x_points, y_points, marker=marker_map[algo], color=color_map[algo], label=algo, linewidth=2.5, markersize=8)
        else:
            spline = CubicSpline(x_points[valid_indices], y_points[valid_indices])
            x_smooth = np.linspace(valid_indices.min(), valid_indices.max(), 300)
            y_smooth = spline(x_smooth)
            ax.plot(x_smooth, y_smooth, color=color_map[algo], label=algo, linewidth=2.5)
            ax.scatter(x_points, y_points, color=color_map[algo], marker=marker_map[algo], s=100, zorder=5)

        # ИСПРАВЛЕНИЕ: Используем числовые координаты x_points для размещения текста
        for x_coord, y_coord in zip(x_points, y_points):
            if y_coord > 0.1:
                ax.text(x_coord, y_coord + 0.35, f"{y_coord:.1f}", 
                        ha='center', va='bottom', fontsize=9, fontweight='bold', zorder=6)

    ax.set_title('Динамика эффективности алгоритмов (Makespan)', fontsize=16, fontweight='bold')
    ax.set_xlabel('Дата', fontsize=12)
    ax.set_ylabel('Общее время выполнения (часы)', fontsize=12)
    
    min_val = df[df['makespan_hours'] > 0]['makespan_hours'].min()
    max_val = df['makespan_hours'].max()
    ax.set_ylim(min_val - 2, max_val + 2)

    ax.legend(title='Алгоритм', fontsize=11)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    ax.set_xticks(np.arange(len(dates)))
    ax.set_xticklabels(dates)

    plt.tight_layout()
    output_filename = 'comparison_makespan_line_smooth.png'
    plt.savefig(output_filename, dpi=300)
    print(f"График сохранен в файл: {output_filename}")
    plt.close(fig)

# Остальные функции (гистограмма, утилизация, время ожидания) остаются без изменений
def plot_makespan_comparison_barchart(df):
    print("Создание графика сравнения Makespan (Гистограмма)...")
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(14, 8))
    sns.barplot(data=df, x='date', y='makespan_hours', hue='algorithm', ax=ax, palette='viridis')
    for p in ax.patches:
        if p.get_height() > 0.1:
            ax.annotate(f"{p.get_height():.1f}",(p.get_x() + p.get_width() / 2., p.get_height()),ha='center', va='center', xytext=(0, 9), textcoords='offset points', fontsize=9, fontweight='bold')
    ax.set_title('Сравнение эффективности алгоритмов по дням (Makespan)', fontsize=16, fontweight='bold')
    ax.set_xlabel('Дата', fontsize=12); ax.set_ylabel('Общее время выполнения (часы)', fontsize=12)
    ax.legend(title='Алгоритм', fontsize=11); ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.grid(axis='y', linestyle='--', alpha=0.7); ax.set_ylim(0, df['makespan_hours'].max() * 1.15)
    plt.tight_layout(); output_filename = 'comparison_makespan_bar.png'
    plt.savefig(output_filename, dpi=300); print(f"График сохранен в файл: {output_filename}"); plt.close(fig)

def calculate_and_plot_utilization(base_dir, date, algo_map, machines):
    print(f"Создание графика утилизации оборудования для {date} января...")
    utilization_data = []
    for algo_name, filename in algo_map.items():
        file_path = os.path.join(base_dir, date, filename)
        if not os.path.exists(file_path): continue
        df = pd.read_csv(file_path)
        if df.empty: continue
        makespan = df['End_Time_Min'].max()
        for stage, count in machines.items():
            stage_df = df[df['Stage'] == stage]
            total_duration = stage_df['Duration_Min'].sum()
            total_available_time = makespan * count
            utilization = (total_duration / total_available_time) if total_available_time > 0 else 0
            utilization_data.append({'algorithm': algo_name, 'stage': stage, 'utilization': utilization})
    if not utilization_data: print(f"Нет данных для построения графика утилизации для {date} января."); return
    util_df = pd.DataFrame(utilization_data)
    g = sns.catplot(data=util_df, kind="bar", x="stage", y="utilization", col="algorithm", palette="plasma", height=5, aspect=1.2)
    g.fig.suptitle(f'Утилизация оборудования по этапам (Январь {date})', y=1.03, fontsize=16, fontweight='bold')
    g.set_axis_labels("Этап производства", "Утилизация (%)"); g.set_xticklabels(rotation=45, ha='right')
    for ax in g.axes.flat:
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
        for p in ax.patches:
            ax.annotate(f"{p.get_height():.0%}", (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='bottom', fontsize=8)
    plt.tight_layout(rect=[0, 0, 1, 0.97]); output_filename = f'comparison_utilization_{date}jan.png'
    plt.savefig(output_filename, dpi=300); print(f"График сохранен в файл: {output_filename}"); plt.close()

def calculate_and_plot_wait_times(base_dir, date, algo_map):
    print(f"Создание графика времени ожидания для {date} января...")
    wait_time_data = []
    for algo_name, filename in algo_map.items():
        file_path = os.path.join(base_dir, date, filename)
        if not os.path.exists(file_path): continue
        df = pd.read_csv(file_path)
        if df.empty: continue
        df_sorted = df.sort_values(['Batch_ID', 'Start_Time_Min'])
        df_sorted['prev_end_time'] = df_sorted.groupby('Batch_ID')['End_Time_Min'].shift(1)
        df_sorted['wait_time'] = df_sorted['Start_Time_Min'] - df_sorted['prev_end_time']
        valid_waits = df_sorted[df_sorted['wait_time'] >= 0]['wait_time']
        for wait in valid_waits: wait_time_data.append({'algorithm': algo_name, 'wait_time_min': wait})
    if not wait_time_data: print(f"Нет данных для построения графика ожидания для {date} января."); return
    wait_df = pd.DataFrame(wait_time_data)
    plt.style.use('seaborn-v0_8-whitegrid'); fig, ax = plt.subplots(figsize=(12, 7))
    sns.boxplot(data=wait_df, x='algorithm', y='wait_time_min', ax=ax, palette='coolwarm')
    ax.set_title(f'Распределение времени ожидания партий между этапами (Январь {date})', fontsize=16, fontweight='bold')
    ax.set_xlabel('Алгоритм', fontsize=12); ax.set_ylabel('Время ожидания (минуты)', fontsize=12)
    ax.set_ylim(0, 40); ax.grid(axis='y', linestyle='--', alpha=0.7); plt.tight_layout()
    output_filename = f'comparison_waittime_{date}jan.png'
    plt.savefig(output_filename, dpi=300); print(f"График сохранен в файл: {output_filename}"); plt.close(fig)

# --- 3. ГЛАВНЫЙ БЛОК ИСПОЛНЕНИЯ ---

if __name__ == '__main__':
    makespan_df = load_makespan_data(BASE_DIR, DATES, ALGO_MAP)
    if not makespan_df.empty:
        # ИСПРАВЛЕНИЕ: Устанавливаем правильный порядок дат и сортируем
        date_order = [f'Янв {d}' for d in DATES]
        makespan_df['date'] = pd.Categorical(makespan_df['date'], categories=date_order, ordered=True)
        makespan_df = makespan_df.sort_values('date')

        print("\nСводная таблица Makespan (в часах):")
        print(makespan_df.pivot_table(index='date', columns='algorithm', values='makespan_hours').round(2))
        
        plot_makespan_comparison_barchart(makespan_df)
        plot_makespan_comparison_linechart_smooth(makespan_df)
    
    representative_date = DATES[-1]
    print(f"\n--- Детальный анализ для {representative_date} января ---")
    calculate_and_plot_utilization(BASE_DIR, representative_date, ALGO_MAP, MACHINES_AVAILABLE)
    calculate_and_plot_wait_times(BASE_DIR, representative_date, ALGO_MAP)
    
    print("\nАнализ завершен. Все графики сохранены в текущей папке.")