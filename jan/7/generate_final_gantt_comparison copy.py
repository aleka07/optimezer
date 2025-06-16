import csv
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.font_manager as fm
import datetime
import math
import collections
import os # Убедитесь, что этот импорт есть в начале файла

# --- Имя входного файла (теперь определяется относительно папки со скриптом) ---
# 1. Получаем абсолютный путь к папке, где лежит этот скрипт
try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    # Этот блок сработает, если вы запускаете код в интерактивной среде, где __file__ не определен
    script_dir = os.getcwd()

# 2. Соединяем путь к этой папке с именем нашего файла
INPUT_CSV_FILE = os.path.join(script_dir, 'production_schedule_v2.csv')
# INPUT_CSV_FILE = os.path.join(script_dir, 'ga_production_schedule1.csv')
# INPUT_CSV_FILE = os.path.join(script_dir, 'dqn_production_schedule.csv')

# Список этапов для последовательности и цветов
STAGES = [
    "Комбинирование", "Смешивание", "Формовка", "Расстойка",
    "Выпекание", "Остывание",
]

# --- Функция Чтения Данных из CSV ---
def read_schedule_from_csv(filename):
    schedule_data = []
    max_end_time = 0.0
    if not os.path.exists(filename):
        print(f"Ошибка: Файл '{filename}' не найден.")
        return None, 0
    try:
        with open(filename, 'r', newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            print(f"Чтение данных из файла: '{filename}'...")
            required_columns = ["Batch_ID", "Stage", "Start_Time_Min", "End_Time_Min"]
            if not all(col in reader.fieldnames for col in required_columns):
                missing = [col for col in required_columns if col not in reader.fieldnames]
                print(f"Ошибка: В CSV файле отсутствуют необходимые столбцы: {missing}")
                return None, 0
            line_num = 1
            for row in reader:
                line_num += 1
                try:
                    start_time = int(row['Start_Time_Min'])
                    end_time = int(row['End_Time_Min'])
                    duration = end_time - start_time
                    if start_time < 0 or end_time < 0 or duration < 0:
                         print(f"Предупреждение: Некорректные временные значения в строке {line_num}. Строка пропущена: {row}")
                         continue
                    task_data = {'Batch': row['Batch_ID'], 'Stage': row['Stage'],
                                 'Start': start_time, 'End': end_time, 'Duration': duration}
                    schedule_data.append(task_data)
                    if end_time > max_end_time: max_end_time = end_time
                except Exception as e: print(f"Ошибка обработки строки {line_num}: {e}. Строка пропущена: {row}")
        if not schedule_data:
             print("Предупреждение: Не найдено корректных данных о задачах.")
             return None, 0
        print(f"Данные успешно прочитаны. Задач: {len(schedule_data)}. Makespan: {max_end_time} мин.")
        return schedule_data, float(max_end_time)
    except Exception as e:
        print(f"Ошибка чтения файла '{filename}': {e}")
        return None, 0

# --- Функция Визуализации Диаграммы Ганта (ВСЕ ИЗМЕНЕНИЯ ЗДЕСЬ) ---

def plot_gantt_chart_by_batch_start_time(schedule_results, makespan_minutes, stages_order):
    """
    Строит линейный график: по оси X — время, по оси Y — партия (batch), этапы отмечены маркерами.
    Легенда и подписи увеличены для читаемости.
    """
    if not schedule_results or makespan_minutes <= 0:
        print("Нет данных для визуализации или makespan некорректен.")
        return

    # --- Настройка шрифтов ---
    default_font = 'DejaVu Sans'
    axis_font_prop = fm.FontProperties(family=default_font, size=16)
    tick_fontsize = 16
    legend_fontsize = 18
    title_fontsize = 20
    plt.rcParams['font.family'] = default_font

    # --- Подготовка данных ---
    tasks_by_batch = collections.defaultdict(list)
    all_batches_set = set()
    for task in schedule_results:
        tasks_by_batch[task['Batch']].append(task)
        all_batches_set.add(task['Batch'])

    batch_start_times = {}
    for batch_id, tasks in tasks_by_batch.items():
        if tasks:
            min_start = min(task['Start'] for task in tasks)
            batch_start_times[batch_id] = min_start
        else:
            batch_start_times[batch_id] = float('inf')

    def sort_key(batch_name):
        return (batch_start_times.get(batch_name, float('inf')), batch_name)

    sorted_batches = sorted(list(all_batches_set), key=sort_key)
    num_batches = len(sorted_batches)
    batch_to_y = {batch: i for i, batch in enumerate(sorted_batches)}

    cmap = plt.get_cmap('tab20')
    num_colors = cmap.N
    stage_colors = {stage: cmap(i % num_colors) for i, stage in enumerate(stages_order)}
    stage_colors_with_default = collections.defaultdict(lambda: 'grey', stage_colors)

    fig, ax = plt.subplots(figsize=(max(18, makespan_minutes / 18), max(10, num_batches * 0.5)))

    # Для легенды: собираем уникальные этапы
    legend_handles = {}

    for batch_name in sorted_batches:
        y_pos = batch_to_y[batch_name]
        sorted_tasks = sorted(tasks_by_batch[batch_name], key=lambda t: t['Start'])
        x_points = []
        y_points = []
        colors = []
        for task in sorted_tasks:
            stage = task['Stage']
            start = task['Start']
            end = task['End']
            color = stage_colors_with_default[stage]
            x_points.append(start)
            y_points.append(y_pos)
            colors.append(color)
            # Для легенды
            if stage not in legend_handles:
                legend_handles[stage] = mpatches.Patch(color=color, label=stage)
        # Соединяем этапы линией
        if len(x_points) > 1:
            ax.plot(x_points, y_points, color='black', linewidth=2, alpha=0.5, zorder=1)
        # Рисуем маркеры этапов
        ax.scatter(x_points, y_points, c=colors, s=120, edgecolor='black', linewidth=1.2, zorder=2)

    # --- Оформление осей ---
    ax.set_yticks(range(num_batches))
    short_batch_labels = [f"B_{b.split('_')[-1]}" for b in sorted_batches]
    ax.set_yticklabels(short_batch_labels, fontproperties=axis_font_prop, fontsize=tick_fontsize)
    ax.invert_yaxis()
    ax.set_xlabel("Время (минуты)", fontproperties=axis_font_prop, fontsize=title_fontsize, labelpad=12)
    ax.set_ylabel("Партия", fontproperties=axis_font_prop, fontsize=title_fontsize, labelpad=12)
    ax.set_xlim(0, math.ceil(makespan_minutes))
    ax.xaxis.grid(True, linestyle='--', color='gray', alpha=0.6)
    ax.tick_params(axis='x', labelsize=tick_fontsize)
    ax.tick_params(axis='y', labelsize=tick_fontsize)

    # --- Заголовок ---
    total_seconds = int(makespan_minutes * 60)
    tdelta = datetime.timedelta(seconds=total_seconds)
    makespan_formatted = str(tdelta)
    ax.set_title(f"План производства | Makespan: {makespan_minutes:.1f} мин ({makespan_formatted})",
                 fontproperties=axis_font_prop, fontsize=title_fontsize, pad=18)

    # --- Легенда ---
    if legend_handles:
        handles = [legend_handles[stage] for stage in stages_order if stage in legend_handles]
        ax.legend(handles=handles, title="Этапы производства", title_fontsize=legend_fontsize,
                  fontsize=legend_fontsize, bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)
        plt.subplots_adjust(right=0.78)
    else:
        print("Нет данных для отображения легенды.")

    plt.tight_layout(rect=[0, 0, 0.95, 0.97])

    # --- Сохранение диаграммы в файл ---
    try:
        output_dir_name = 'gantt_charts'
        output_dir = os.path.join(script_dir, output_dir_name)
        os.makedirs(output_dir, exist_ok=True)
        base_name = os.path.basename(INPUT_CSV_FILE)
        file_name_without_ext = os.path.splitext(base_name)[0]
        output_filename = f"{file_name_without_ext}_lineplot.png"
        output_path = os.path.join(output_dir, output_filename)
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\nДиаграмма успешно сохранена в файл:\n{output_path}")
    except Exception as e:
        print(f"\nНе удалось сохранить диаграмму. Ошибка: {e}")

    plt.show()

# --- Основной блок выполнения ---
if __name__ == "__main__":
    schedule_data, makespan = read_schedule_from_csv(INPUT_CSV_FILE)

    if schedule_data and makespan > 0:
        print("\nЗапуск визуализации расписания...")
        plot_gantt_chart_by_batch_start_time(schedule_data, makespan, STAGES)
        print("Визуализация завершена.")
    else:
        print("\nВизуализация не может быть построена из-за ошибок при чтении данных.")