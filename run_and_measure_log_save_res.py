import os
import subprocess
import time
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import ScalarFormatter

# --- 1. Конфигурация ---
BASE_DIR = 'jan'
DATES = ['3', '4', '6', '7', '8', '10']
RESULTS_FILENAME = 'execution_times.csv'  # Файл для сохранения/загрузки результатов

# Словарь со скриптами
SCRIPTS_TO_RUN = {
    'CP-SAT': 'time_min.py',
    'GA': 'ga copy.py',
    'DQN': 'dqn copy.py'
}

# --- НОВОВВЕДЕНИЕ: Задаем строгий порядок для консистентности ---
# Этот порядок будет использоваться и при сборе данных, и при построении графика
ALGORITHM_ORDER = ['CP-SAT', 'GA', 'DQN']

python_executable = sys.executable
child_env = os.environ.copy()
child_env["PYTHONIOENCODING"] = "utf-8"

# --- 2. Сбор или загрузка данных ---

# --- НОВОВВЕДЕНИЕ: Проверяем, существуют ли уже сохраненные результаты ---
if os.path.exists(RESULTS_FILENAME):
    print(f"--- Найден файл с результатами '{RESULTS_FILENAME}'. Загрузка данных... ---")
    print("--- Чтобы запустить измерения заново, удалите этот файл. ---")
    results_df = pd.read_csv(RESULTS_FILENAME)

else:
    print(f"--- Файл '{RESULTS_FILENAME}' не найден. Запуск измерений... ---")
    execution_results = []
    
    print(f"--- Используется интерпретатор Python из venv: {python_executable} ---")

    for date in DATES:
        date_path = os.path.join(BASE_DIR, date)
        if not os.path.isdir(date_path):
            print(f"Предупреждение: Папка для даты '{date}' не найдена. Пропускаем.")
            continue

        print(f"\nОбработка данных за {date} января...")

        # --- ИЗМЕНЕНИЕ: Используем заданный порядок ALGORITHM_ORDER ---
        for alg_name in ALGORITHM_ORDER:
            script_name = SCRIPTS_TO_RUN[alg_name]
            full_script_path = os.path.join(date_path, script_name)

            if not os.path.isfile(full_script_path):
                print(f"  - Предупреждение: Скрипт '{script_name}' не найден в папке '{date_path}'. Пропускаем.")
                continue

            print(f"  - Запуск: {alg_name}...", end='', flush=True)

            start_time = time.time()
            try:
                subprocess.run(
                    [python_executable, script_name],
                    cwd=date_path,
                    check=True,
                    capture_output=True,
                    text=True,
                    encoding='utf-8',
                    env=child_env
                )
            except subprocess.CalledProcessError as e:
                print(f"\n    !!! ОШИБКА при выполнении '{script_name}' для даты {date}:")
                print(f"    STDOUT: {e.stdout}")
                print(f"    STDERR: {e.stderr}")
                continue

            end_time = time.time()
            duration = end_time - start_time

            print(f" Завершено за {duration:.2f} сек.")

            execution_results.append({
                'Date': f'Январь {date}',
                'Algorithm': alg_name,
                'ExecutionTime_sec': duration
            })

    print("\n--- Измерение завершено. ---")

    if not execution_results:
        print("Не удалось собрать данные. Файл с результатами не будет создан.")
        sys.exit() # Выход из скрипта, если данных нет

    results_df = pd.DataFrame(execution_results)
    
    # --- НОВОВВЕДЕНИЕ: Сохраняем результаты в CSV файл ---
    results_df.to_csv(RESULTS_FILENAME, index=False)
    print(f"--- Результаты сохранены в файл: {RESULTS_FILENAME} ---")


# --- 3. Визуализация результатов ---
print("\n--- Подготовка графика. ---")

plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(16, 9))

# --- ИЗМЕНЕНИЕ: Добавлен параметр hue_order для гарантии нужного порядка на графике ---
sns.barplot(data=results_df, x='Date', y='ExecutionTime_sec', hue='Algorithm',
            palette='viridis', ax=ax, edgecolor='black', hue_order=ALGORITHM_ORDER)

# Устанавливаем логарифмическую шкалу для оси Y
ax.set_yscale('log')

# Убираем научную нотацию (e.g., 10^1, 10^2) и показываем обычные числа (10, 100)
for axis in [ax.yaxis]:
    axis.set_major_formatter(ScalarFormatter())

# Заголовки и метки осей
ax.set_title('Сравнение времени выполнения алгоритмов (Логарифмическая шкала)', fontsize=18, fontweight='bold', pad=20)
ax.set_xlabel('Дата производственного задания', fontsize=14, labelpad=15)
ax.set_ylabel('Время выполнения, секунды (лог. шкала)', fontsize=14, labelpad=15)

# --- ИЗМЕНЕНИЕ: Указано точное положение легенды ---
ax.legend(title='Алгоритм', title_fontsize='13', fontsize='11', loc='upper right')

ax.tick_params(axis='x', labelsize=12)
ax.tick_params(axis='y', labelsize=12)

# Подписи данных
for container in ax.containers:
    ax.bar_label(container, fmt='%.1f', label_type='edge', fontsize=10, padding=3)

# Устанавливаем нижнюю границу оси Y чуть ниже минимального значения
min_val = results_df['ExecutionTime_sec'].min()
if min_val > 0:
    ax.set_ylim(bottom=min_val * 0.5)

plt.tight_layout()

output_filename = 'algorithms_execution_time_comparison_log.png'
plt.savefig(output_filename, dpi=300)
plt.show()

print(f"\nГрафик с логарифмической шкалой сохранен в файл: {output_filename}")