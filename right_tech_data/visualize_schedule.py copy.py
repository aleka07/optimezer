import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import math

# --- НАСТРОЙКИ ---
# Имя CSV файла, который был создан скриптом оптимизации
INPUT_CSV_FILE = 'production_schedule_optimized.csv'
# Имя файла, в который будет сохранена диаграмма
OUTPUT_IMAGE_FILE = 'gantt_chart_copy.png'
    
# # в файле visualize_schedule.py
# INPUT_CSV_FILE = 'fifo_schedule.csv' # Укажите имя нового файла
# OUTPUT_IMAGE_FILE = 'gantt_chart_fifo.png' # Задайте новое имя для картинки

# INPUT_CSV_FILE = 'fifo_schedule_correct.csv'
# OUTPUT_IMAGE_FILE = 'fifo_summary_correct_copy.png'

  
def create_gantt_chart(csv_path, output_path):
    """
    Читает CSV файл с расписанием и создает диаграмму Ганта.
    """
    # --- 1. Загрузка и проверка данных ---
    try:
        df = pd.read_csv(csv_path)
        print(f"Файл '{csv_path}' успешно загружен. Найдено {len(df)} операций.")
    except FileNotFoundError:
        print(f"ОШИБКА: Файл '{csv_path}' не найден. Сначала запустите скрипт оптимизации 'optimize_bakery.py'.")
        return

    if df.empty:
        print("CSV файл пуст. Диаграмма не может быть построена.")
        return

    # --- 2. Подготовка данных для построения ---
    # Определяем цвета для каждого этапа
    stage_colors = {
        'Combination': '#1f77b4',  # Синий
        'Mixing': '#ff7f0e',      # Оранжевый
        'Molding': '#2ca02c',     # Зеленый
        'Proofing': '#d62728',    # Красный
        'Baking': '#9467bd',      # Фиолетовый
        'Cooling': '#8c564b'       # Коричневый
    }
    
    # Используем минуты для более точного отображения
    df['start_min'] = df['Start_Time_Min']
    df['duration_min'] = df['Duration_Min']

    # Получаем уникальные партии и сортируем их для оси Y
    # Сортируем по времени начала первой операции для каждой партии
    batch_start_times = df.groupby('Batch_ID')['Start_Time_Min'].min().sort_values()
    unique_batches = batch_start_times.index.tolist()
    
    # --- 3. Построение диаграммы ---
    # Увеличиваем размер фигуры для лучшей читабельности
    fig, ax = plt.subplots(figsize=(32, max(16, len(unique_batches) * 0.7)))

    # Итерируемся по каждой операции в расписании и рисуем ее как полосу на диаграмме
    for index, row in df.iterrows():
        batch_id = row['Batch_ID']
        stage = row['Stage']
        
        # Находим вертикальную позицию (y) для текущей партии
        y_pos = unique_batches.index(batch_id)
        
        # Рисуем горизонтальную полосу (задачу)
        ax.barh(
            y=y_pos,
            width=row['duration_min'],
            left=row['start_min'],
            height=0.85,  # Увеличиваем высоту баров
            color=stage_colors.get(stage, 'grey'),
            edgecolor='black',
            linewidth=1.2  # Увеличиваем толщину границ для лучшей видимости
        )

    # --- 4. Настройка внешнего вида диаграммы ---
    # Настройка оси Y (названия партий)
    ax.set_yticks(range(len(unique_batches)))
    ax.set_yticklabels(unique_batches, fontsize=16, fontweight='bold')  # Увеличиваем размер шрифта
    ax.invert_yaxis()  # Первая партия будет сверху

    # Настройка оси X (время в минутах)
    ax.set_xlabel('Время (минуты)', fontsize=18, fontweight='bold')
    
    # Добавляем более детальную и ясную сетку
    ax.grid(True, which='major', axis='x', linestyle='-', alpha=0.3, color='gray')
    ax.grid(True, which='minor', axis='x', linestyle='--', alpha=0.2, color='gray')
    ax.grid(True, axis='y', linestyle='-', alpha=0.2, color='gray')
    
    # Настраиваем частоту меток на оси X для лучшей читаемости
    max_time = df['End_Time_Min'].max()
    if max_time <= 120:  # До 2 часов
        major_ticks = range(0, int(max_time) + 1, 30)  # Каждые 30 минут
        minor_ticks = range(0, int(max_time) + 1, 10)  # Каждые 10 минут
    elif max_time <= 480:  # До 8 часов
        major_ticks = range(0, int(max_time) + 1, 60)  # Каждый час
        minor_ticks = range(0, int(max_time) + 1, 30)  # Каждые 30 минут
    else:  # Более 8 часов
        major_ticks = range(0, int(max_time) + 1, 120)  # Каждые 2 часа
        minor_ticks = range(0, int(max_time) + 1, 60)   # Каждый час
    
    ax.set_xticks(major_ticks)
    ax.set_xticks(minor_ticks, minor=True)

    # Добавление заголовка с общим временем производства
    makespan_min = df['End_Time_Min'].max()
    makespan_hours = makespan_min / 60
    days = int(makespan_hours // 24)
    hours = int(makespan_hours % 24)
    minutes = int(makespan_min % 60)
    
    title_str = f'Диаграмма Ганта для производственного плана\n'
    title_str += f'Общее время (Makespan): {makespan_min:.0f} минут ({makespan_hours:.2f} часов | {days} дн {hours} ч {minutes} мин)'
    ax.set_title(title_str, fontsize=24, fontweight='bold', pad=30)

    # Увеличиваем размер шрифта для меток осей для лучшей читабельности
    ax.tick_params(axis='both', which='major', labelsize=17, width=2, length=8)
    ax.tick_params(axis='x', which='minor', labelsize=14, width=1.5, length=5)
    
    # Создание легенды для цветов с ОЧЕНЬ увеличенным размером
    legend_patches = [mpatches.Patch(color=color, label=stage) for stage, color in stage_colors.items()]
    ax.legend(handles=legend_patches, 
              bbox_to_anchor=(1.01, 1), 
              loc='upper left', 
              fontsize=20,  # МАКСИМАЛЬНО большой шрифт для легенды
              title='Этапы производства',
              title_fontsize=22,  # Огромный заголовок легенды
              frameon=True,
              fancybox=True,
              shadow=True,
              borderpad=1.5,  # Ещё больше отступа внутри рамки
              labelspacing=1.5,  # Ещё больше расстояния между элементами
              handlelength=3,  # Длиннее цветные квадратики
              handleheight=2)  # Выше цветные квадратики

    # Оптимизация расположения элементов
    plt.tight_layout(rect=[0, 0, 0.88, 1]) # Оставляем место справа для легенды

    # --- 5. Сохранение и отображение ---
    try:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\nДиаграмма Ганта успешно сохранена в файл: '{output_path}'")
    except Exception as e:
        print(f"\nОшибка при сохранении файла: {e}")
        
    # plt.show() # Раскомментируйте, если хотите, чтобы диаграмма открылась на экране

if __name__ == '__main__':
    create_gantt_chart(INPUT_CSV_FILE, OUTPUT_IMAGE_FILE)