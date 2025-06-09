import csv
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.font_manager as fm
import datetime
import math
import collections
import os

# --- Имя входного файла ---
INPUT_CSV_FILE = 'fifo_production_schedule.csv' # Должен совпадать с OUTPUT_CSV_FILE из time_minimizer.py
# INPUT_CSV_FILE = 'production_schedule_v2.csv' # Должен совпадать с OUTPUT_CSV_FILE из time_minimizer.py
# INPUT_CSV_FILE = 'scheduler_heuristic.csv' # Должен совпадать с OUTPUT_CSV_FILE из time_minimizer.py
# INPUT_CSV_FILE = 'production_schedule_v3_individual_machines.csv' # Должен совпадать с OUTPUT_CSV_FILE из time_minimizer.py



# Список этапов для последовательности и цветов (должен совпадать с STAGES из time_minimizer.py)
STAGES_FOR_DIAGRAM = [
    "Комбинирование", "Смешивание", "Формовка", "Расстойка",
    "Выпекание", "Остывание", "Упаковка",
]

# --- Функция Чтения Данных из CSV ---
def read_schedule_from_csv(filename):
    schedule_data = []
    max_end_time = 0.0
    if not os.path.exists(filename):
        print(f"Ошибка: Файл '{filename}' не найден. Запустите сначала time_minimizer.py")
        return None, 0
    try:
        with open(filename, 'r', newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            print(f"Чтение данных из файла: '{filename}'...")
            required_columns = ["Batch_ID", "Stage", "Start_Time_Min", "End_Time_Min", "Duration_Min"]
            if not reader.fieldnames: # Проверка на пустой файл или отсутствие заголовков
                 print(f"Ошибка: Файл '{filename}' пуст или не содержит заголовков.")
                 return None, 0
            if not all(col in reader.fieldnames for col in required_columns):
                missing = [col for col in required_columns if col not in reader.fieldnames]
                print(f"Ошибка: В CSV файле отсутствуют необходимые столбцы: {missing}")
                return None, 0

            line_num = 1
            for row in reader:
                line_num += 1
                try:
                    # Преобразуем в float, затем в int, чтобы избежать проблем с ".0"
                    start_time = int(float(row['Start_Time_Min']))
                    end_time = int(float(row['End_Time_Min']))
                    duration = int(float(row['Duration_Min'])) # или end_time - start_time

                    if start_time < 0 or end_time < 0 or duration < 0:
                         print(f"Предупреждение: Некорректные временные значения в строке {line_num}. Строка пропущена: {row}")
                         continue
                    if duration == 0 and start_time == end_time: # Пропускаем задачи с нулевой длительностью, если они есть
                        # print(f"Информация: Пропущена задача с нулевой длительностью в строке {line_num}: {row}")
                        continue

                    task_data = {'Batch': row['Batch_ID'], 'Stage': row['Stage'],
                                 'Start': start_time, 'End': end_time, 'Duration': duration}
                    schedule_data.append(task_data)
                    if end_time > max_end_time: max_end_time = end_time
                except ValueError as ve:
                    print(f"Ошибка преобразования значения в строке {line_num}: {ve}. Строка пропущена: {row}")
                except Exception as e:
                    print(f"Ошибка обработки строки {line_num}: {e}. Строка пропущена: {row}")

        if not schedule_data:
             print("Предупреждение: Не найдено корректных данных о задачах для визуализации в файле.")
             return None, 0
        print(f"Данные успешно прочитаны. Задач для диаграммы: {len(schedule_data)}. Makespan (из CSV): {max_end_time} мин.")
        return schedule_data, float(max_end_time)
    except Exception as e:
        print(f"Критическая ошибка чтения файла '{filename}': {e}")
        return None, 0

# --- Функция Визуализации Диаграммы Ганта ---
def plot_gantt_chart_by_batch_start_time(schedule_results, makespan_minutes, stages_order_for_colors):
    if not schedule_results or makespan_minutes <= 0:
        print("Нет данных для визуализации или makespan некорректен.")
        return

    # --- Настройка шрифта ---
    font_path = None # Укажите путь к .ttf файлу, если стандартный не поддерживает кириллицу
    default_font = 'DejaVu Sans' # Стандартный шрифт, который часто есть и поддерживает кириллицу
    axis_font_prop = None
    bar_font_prop = None
    try:
        if font_path and os.path.exists(font_path):
            axis_font_prop = fm.FontProperties(fname=font_path, size=10)
            bar_font_prop = fm.FontProperties(fname=font_path, size=7)
        else:
            # Попытка использовать стандартные шрифты, которые могут поддерживать кириллицу
            available_fonts = fm.findSystemFonts(fontpaths=None, fontext='ttf')
            cyrillic_support_fonts = ['DejaVu Sans', 'Arial', 'Verdana', 'Tahoma', 'Times New Roman'] # Приоритет
            
            selected_font_family = default_font # По умолчанию
            for font_name_pref in cyrillic_support_fonts:
                if any(font_name_pref.lower() in f.lower() for f in available_fonts):
                    try:
                        # Проверим, можно ли его реально использовать
                        fm.FontProperties(family=font_name_pref)
                        selected_font_family = font_name_pref
                        break
                    except RuntimeError:
                        continue
            
            axis_font_prop = fm.FontProperties(family=selected_font_family, size=10)
            bar_font_prop = fm.FontProperties(family=selected_font_family, size=7)

        plt.rcParams['font.family'] = axis_font_prop.get_name()
        print(f"Используется шрифт: {plt.rcParams['font.family']} для диаграммы.")
    except Exception as e:
        print(f"Ошибка настройки шрифта ('{default_font}' или указанный): {e}. Кириллица может отображаться некорректно.")
        # Fallback на системный шрифт по умолчанию
        axis_font_prop = fm.FontProperties(size=10)
        bar_font_prop = fm.FontProperties(size=7)


    # --- Подготовка данных ---
    tasks_by_batch = collections.defaultdict(list)
    all_batches_set = set()
    for task in schedule_results:
        tasks_by_batch[task['Batch']].append(task)
        all_batches_set.add(task['Batch'])

    batch_start_times = {}
    for batch_id, tasks_for_batch in tasks_by_batch.items():
        if tasks_for_batch:
            min_start = min(t['Start'] for t in tasks_for_batch)
            batch_start_times[batch_id] = min_start
        else:
            batch_start_times[batch_id] = float('inf')

    def sort_key_batch(batch_name):
        return (batch_start_times.get(batch_name, float('inf')), batch_name)

    sorted_batches_names = sorted(list(all_batches_set), key=sort_key_batch)
    num_batches = len(sorted_batches_names)
    batch_to_y_map = {batch_name: i for i, batch_name in enumerate(sorted_batches_names)}

    # Генерация цветов
    try:
        cmap = plt.get_cmap('tab20') # Хороший набор из 20 цветов
    except ValueError:
        cmap = plt.get_cmap('viridis') # Запасной вариант, если tab20 нет
        
    num_colors_available = cmap.N
    
    # Создаем цвета для этапов, которые есть в stages_order_for_colors
    # Остальные (если вдруг появятся в CSV, но не в STAGES_FOR_DIAGRAM) будут серыми
    stage_colors = {
        stage: cmap(i % num_colors_available)
        for i, stage in enumerate(stages_order_for_colors)
    }
    stage_colors_with_default = collections.defaultdict(lambda: 'grey', stage_colors)


    # --- Создание диаграммы ---
    fig_height = max(8, num_batches * 0.45)  # Динамическая высота
    fig_width = max(15, makespan_minutes / 20) # Динамическая ширина
    if fig_width > 50: fig_width = 50 # Ограничение на максимальную ширину
    if fig_height > 30: fig_height = 30 # Ограничение на максимальную высоту

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    for batch_name_iter in sorted_batches_names:
        y_pos = batch_to_y_map[batch_name_iter]
        # Сортируем задачи внутри партии по времени начала для корректного наложения (хотя они должны быть уже отсортированы)
        # Это не обязательно, если AddCumulative работает правильно, но не повредит
        # tasks_for_this_batch_sorted = sorted(tasks_by_batch[batch_name_iter], key=lambda t: t['Start'])

        for task in tasks_by_batch[batch_name_iter]: # Используем задачи как есть, т.к. они не должны перекрываться для одной партии
            stage = task['Stage']; start = task['Start']; duration = task['Duration']
            if duration <= 0: continue # Пропускаем задачи с нулевой или отрицательной длительностью

            color_for_stage = stage_colors_with_default[stage]
            ax.barh(y=y_pos, width=duration, left=start, height=0.6, align='center',
                    color=color_for_stage, edgecolor='black', linewidth=0.5, alpha=0.9)

            # Добавление текста на плашки (сокращенное название этапа)
            # Определяем, достаточно ли длинная плашка для текста
            # min_pixels_for_text = 30 # примерно
            # min_duration_for_text_pixels = (min_pixels_for_text / fig.dpi) * (makespan_minutes / fig_width)
            min_duration_for_text_dynamic = makespan_minutes / (fig_width * 2) # Эвристика: если плашка занимает хотя бы 1/ (ширина_фигуры * 2) от общего времени
            if duration > min_duration_for_text_dynamic and duration > 2 : # И хотя бы 2 минуты
                 stage_abbr = stage[:3] + '.' if len(stage) > 3 else stage # Сокращение
                 # Определяем цвет текста в зависимости от фона плашки для лучшей читаемости
                 text_color = 'white' if sum(color_for_stage[:3]) < 1.5 else 'black' # RGB (0-1), если сумма < 1.5 - темный фон
                 ax.text(start + duration / 2, y_pos, stage_abbr, ha='center', va='center',
                         color=text_color, fontsize=bar_font_prop.get_size(),
                         fontproperties=bar_font_prop, weight='bold')

    # --- Настройка внешнего вида ---
    ax.set_yticks(range(num_batches))
    ax.set_yticklabels(sorted_batches_names, fontproperties=axis_font_prop)
    ax.invert_yaxis() # Первая партия сверху

    ax.set_xlabel("Время (минуты)", fontproperties=axis_font_prop)
    ax.set_ylabel("Партия продукции (сортировка по времени начала)", fontproperties=axis_font_prop)

    ax.set_xlim(0, math.ceil(makespan_minutes)) # Округляем makespan вверх для оси X
    ax.xaxis.grid(True, linestyle='--', color='gray', alpha=0.6)

    # Форматирование makespan для заголовка
    total_seconds_makespan_diag = int(makespan_minutes * 60)
    tdelta_diag = datetime.timedelta(seconds=total_seconds_makespan_diag)
    days_diag = tdelta_diag.days
    hours_diag, remainder_diag = divmod(tdelta_diag.seconds, 3600)
    minutes_diag, seconds_diag = divmod(remainder_diag, 60)
    makespan_formatted_diag = ""
    if days_diag > 0: makespan_formatted_diag += f"{days_diag} дн "
    makespan_formatted_diag += f"{hours_diag:02}:{minutes_diag:02}:{seconds_diag:02}"


    title_font_prop = axis_font_prop.copy(); title_font_prop.set_size(14); title_font_prop.set_weight('bold')
    ax.set_title(f"Диаграмма Ганта по Партиям (из файла {INPUT_CSV_FILE})\nОбщее время: {makespan_minutes:.1f} мин ({makespan_formatted_diag})",
                 fontproperties=title_font_prop, pad=20)

    # --- Легенда ---
    legend_patches = []
    # Берем этапы, которые реально есть в результатах, и сортируем их согласно STAGES_FOR_DIAGRAM
    stages_present_in_results = sorted(list(set(item['Stage'] for item in schedule_results)),
                               key=lambda s: stages_order_for_colors.index(s) if s in stages_order_for_colors else float('inf'))

    for stage_in_legend in stages_present_in_results:
        color = stage_colors_with_default[stage_in_legend]
        if stage_in_legend: # Убедимся, что имя этапа не пустое
            legend_patches.append(mpatches.Patch(color=color, label=stage_in_legend))

    if legend_patches:
        legend_title_prop = axis_font_prop.copy(); legend_title_prop.set_weight('bold')
        ax.legend(handles=legend_patches, title="Этапы производства",
                  bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0.,
                  prop=axis_font_prop, title_fontproperties=legend_title_prop)
        plt.subplots_adjust(right=0.82 if num_batches < 30 else 0.85) # Дать больше места легенде если много партий
    else:
        print("Нет данных для отображения легенды.")

    plt.tight_layout(rect=[0, 0, 0.9 if legend_patches else 1, 0.95]) # rect=[left, bottom, right, top]
    plt.show()

# --- Основной блок выполнения ---
if __name__ == "__main__":
    schedule_data_from_file, makespan_from_file = read_schedule_from_csv(INPUT_CSV_FILE)

    if schedule_data_from_file and makespan_from_file > 0:
        print("\nЗапуск визуализации расписания (сортировка партий по времени начала)...")
        plot_gantt_chart_by_batch_start_time(schedule_data_from_file, makespan_from_file, STAGES_FOR_DIAGRAM)
        print("Визуализация завершена.")
    else:
        print("\nВизуализация не может быть построена из-за ошибок при чтении данных или нулевого makespan.")