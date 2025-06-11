import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- Шаг 1: Сбор данных ---
# Предполагается, что скрипт находится в папке, содержащей папку 'jan'
base_dir = 'jan'
dates = ['3', '4', '6', '7', '8', '10']
algorithms = {
    'CP-SAT': 'production_schedule_v2.csv',
    'GA': 'ga_production_schedule1.csv',
    'DQN': 'dqn_production_schedule.csv'
}

results = []

for date in dates:
    date_path = os.path.join(base_dir, date)
    if not os.path.isdir(date_path):
        print(f"Папка для даты {date} не найдена.")
        continue
        
    for alg_name, file_name in algorithms.items():
        file_path = os.path.join(date_path, file_name)
        try:
            df = pd.read_csv(file_path)
            # Makespan - это максимальное время окончания задачи
            makespan = df['End_Time_Min'].max() if not df.empty else 0
            results.append({'Date': f'Январь {date}', 'Algorithm': alg_name, 'Makespan_min': makespan})
        except FileNotFoundError:
            print(f"Файл не найден: {file_path}")
        except Exception as e:
            print(f"Ошибка при чтении файла {file_path}: {e}")

# Проверка, что данные были собраны
if not results:
    print("Данные для построения графика не найдены. Проверьте структуру папок и имена файлов.")
else:
    results_df = pd.DataFrame(results)
    # Конвертация в часы для лучшей читаемости
    results_df['Makespan_hours'] = results_df['Makespan_min'] / 60.0

    # --- Шаг 2: Визуализация ---
    plt.style.use('seaborn-v0_8-whitegrid') # Профессиональный стиль
    fig, ax = plt.subplots(figsize=(14, 8))

    # Использование seaborn для легкого создания сгруппированной гистограммы
    sns.barplot(data=results_df, x='Date', y='Makespan_hours', hue='Algorithm',
                palette='viridis', ax=ax, edgecolor='black')

    # Настройка графика
    ax.set_title('Сравнение эффективности алгоритмов по дням', fontsize=16, fontweight='bold')
    ax.set_xlabel('Дата производственного задания', fontsize=12)
    ax.set_ylabel('Общее время выполнения (Makespan), часы', fontsize=12)
    ax.legend(title='Алгоритм', fontsize=11)
    ax.tick_params(axis='x', rotation=45) # Поворот подписей дат для лучшей читаемости
    
    # Добавление значений на столбцы для наглядности
    for container in ax.containers:
        ax.bar_label(container, fmt='%.1f', label_type='edge', fontsize=9, padding=3)

    plt.tight_layout() # Автоматическая подгонка элементов графика
    
    # Сохранение графика в файл
    output_filename = 'algorithms_comparison_barchart.png'
    plt.savefig(output_filename, dpi=300) # Высокое разрешение для статьи
    plt.show()

    print(f"График сохранен в файл: {output_filename}")