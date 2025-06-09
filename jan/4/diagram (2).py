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
INPUT_CSV_FILE = os.path.join(script_dir, 'milp_production_schedule.csv')

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

# --- Функция Визуализации Диаграммы Ганта (ИЗМЕНЕНА СОРТИРОВКА ПАРТИЙ) ---

def plot_gantt_chart_by_batch_start_time(schedule_results, makespan_minutes, stages_order):
    """
    Создает диаграмму Ганта, где партии на оси Y отсортированы по времени
    начала их первой операции.

    Args:
        schedule_results (list): Список словарей с задачами.
        makespan_minutes (float): Общее время выполнения (makespan).
        stages_order (list): Список названий этапов для цвета.
    """
    if not schedule_results or makespan_minutes <= 0:
        print("Нет данных для визуализации или makespan некорректен.")
        return

    # --- Настройка шрифта ---
    font_path = None
    default_font = 'DejaVu Sans'
    try:
        if font_path:
            axis_font_prop = fm.FontProperties(fname=font_path, size=10)
            bar_font_prop = fm.FontProperties(fname=font_path, size=7)
            plt.rcParams['font.family'] = axis_font_prop.get_name()
        else:
            axis_font_prop = fm.FontProperties(family=default_font, size=10)
            bar_font_prop = fm.FontProperties(family=default_font, size=7)
            plt.rcParams['font.family'] = default_font
        print(f"Используется шрифт: {plt.rcParams['font.family']} для диаграммы.")
    except RuntimeError:
        print(f"Шрифт '{default_font}' не найден. Кириллица может отображаться некорректно.")
        axis_font_prop = fm.FontProperties(size=10)
        bar_font_prop = fm.FontProperties(size=7)

    # --- Подготовка данных ---
    # Группируем задачи по партиям
    tasks_by_batch = collections.defaultdict(list)
    all_batches_set = set()
    for task in schedule_results:
        tasks_by_batch[task['Batch']].append(task)
        all_batches_set.add(task['Batch'])

    # === ИЗМЕНЕНИЕ: Вычисление времени начала для каждой партии ===
    batch_start_times = {}
    for batch_id, tasks in tasks_by_batch.items():
        if tasks:
            min_start = min(task['Start'] for task in tasks)
            batch_start_times[batch_id] = min_start
        else:
            batch_start_times[batch_id] = float('inf') # Если у партии нет задач

    # === ИЗМЕНЕНИЕ: Сортировка партий по времени начала ===
    # Сначала по времени начала, затем по ID партии для стабильности
    def sort_key(batch_name):
        return (batch_start_times.get(batch_name, float('inf')), batch_name)

    sorted_batches = sorted(list(all_batches_set), key=sort_key)
    # =========================================================

    num_batches = len(sorted_batches)
    # Обновляем Y-координаты на основе новой сортировки
    batch_to_y = {batch: i for i, batch in enumerate(sorted_batches)}

    # Генерация цветов (без изменений)
    cmap = plt.get_cmap('tab20')
    num_colors = cmap.N
    stage_colors = {stage: cmap(i % num_colors) for i, stage in enumerate(stages_order)}
    stage_colors_with_default = collections.defaultdict(lambda: 'grey', stage_colors)

    # --- Создание диаграммы ---
    fig, ax = plt.subplots(figsize=(max(15, makespan_minutes / 25), max(8, num_batches * 0.4)))

    for batch_name in sorted_batches: # Используем новый отсортированный список
        y_pos = batch_to_y[batch_name]
        sorted_tasks = sorted(tasks_by_batch[batch_name], key=lambda t: t['Start'])

        for task in sorted_tasks:
            stage = task['Stage']; start = task['Start']; duration = task['Duration']
            if duration <= 0: continue
            color = stage_colors_with_default[stage]
            ax.barh(y=y_pos, width=duration, left=start, height=0.6, align='center',
                    color=color, edgecolor='black', linewidth=0.5, alpha=0.9)

            min_duration_for_text = makespan_minutes / 45
            if duration > min_duration_for_text:
                 stage_abbr = stage[:4] + '.' if len(stage) > 4 else stage
                 text_color = 'white' if sum(color[:3]) < 1.5 else 'black'
                 ax.text(start + duration / 2, y_pos, stage_abbr, ha='center', va='center',
                         color=text_color, fontsize=bar_font_prop.get_size(),
                         weight='bold', fontproperties=bar_font_prop)

    # --- Настройка внешнего вида (используем новые sorted_batches для меток оси Y) ---
    ax.set_yticks(range(num_batches))
    ax.set_yticklabels(sorted_batches, fontproperties=axis_font_prop) # Метки теперь в новом порядке
    ax.invert_yaxis()

    ax.set_xlabel("Время (минуты)", fontproperties=axis_font_prop)
    ax.set_ylabel("Партия продукции (сортировка по нач. времени)", fontproperties=axis_font_prop) # Обновили подпись оси Y

    ax.set_xlim(0, math.ceil(makespan_minutes))
    ax.xaxis.grid(True, linestyle='--', color='gray', alpha=0.6)

    total_seconds = int(makespan_minutes * 60)
    tdelta = datetime.timedelta(seconds=total_seconds)
    makespan_formatted = str(tdelta)

    title_font_prop = axis_font_prop.copy(); title_font_prop.set_size(14)
    ax.set_title(f"Диаграмма Ганта по Партиям (из файла {INPUT_CSV_FILE})\nОбщее время: {makespan_minutes:.1f} мин ({makespan_formatted})",
                 fontproperties=title_font_prop, pad=15)

    # --- Легенда ---
    legend_patches = []
    stages_in_results = sorted(list(set(item['Stage'] for item in schedule_results)),
                               key=lambda s: stages_order.index(s) if s in stages_order else float('inf'))
    for stage in stages_in_results:
        color = stage_colors_with_default[stage]
        if stage: legend_patches.append(mpatches.Patch(color=color, label=stage))

    if legend_patches:
        legend_title_prop = axis_font_prop.copy(); legend_title_prop.set_weight('bold')
        ax.legend(handles=legend_patches, title="Этапы производства",
                bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0.,
                prop=axis_font_prop, title_fontproperties=legend_title_prop)
        plt.subplots_adjust(right=0.82)
    else: print("Нет данных для отображения легенды.")

    plt.tight_layout(rect=[0, 0, 0.9, 0.95])
    plt.show()

# --- Основной блок выполнения ---
if __name__ == "__main__":
    schedule_data, makespan = read_schedule_from_csv(INPUT_CSV_FILE)

    if schedule_data and makespan > 0:
        print("\nЗапуск визуализации расписания (сортировка партий по времени начала)...")
        # Вызываем новую функцию
        plot_gantt_chart_by_batch_start_time(schedule_data, makespan, STAGES)
        print("Визуализация завершена.")
    else:
        print("\nВизуализация не может быть построена из-за ошибок при чтении данных.")