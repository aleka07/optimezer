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

SCRIPTS_TO_RUN = {
    'CP-SAT (time_min.py)': 'time_min.py',
    'Genetic Algorithm (ga copy.py)': 'ga copy.py',
    'DQN (dqn copy.py)': 'dqn copy.py'
}

python_executable = sys.executable
print(f"--- Используется интерпретатор Python из venv: {python_executable} ---")

child_env = os.environ.copy()
child_env["PYTHONIOENCODING"] = "utf-8"


# --- 2. Сбор данных о времени выполнения (без изменений) ---
execution_results = []

print("--- Начало измерения времени выполнения скриптов ---")

for date in DATES:
    date_path = os.path.join(BASE_DIR, date)
    if not os.path.isdir(date_path):
        print(f"Предупреждение: Папка для даты '{date}' не найдена. Пропускаем.")
        continue

    print(f"\nОбработка данных за {date} января...")

    for alg_name, script_name in SCRIPTS_TO_RUN.items():
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

print("\n--- Измерение завершено. Подготовка графика. ---")

# --- 3. Визуализация результатов (с логарифмической шкалой) ---
if not execution_results:
    print("Не удалось собрать данные для построения графика. Проверьте пути и наличие файлов.")
else:
    results_df = pd.DataFrame(execution_results)

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(16, 9))

    sns.barplot(data=results_df, x='Date', y='ExecutionTime_sec', hue='Algorithm',
                palette='plasma', ax=ax, edgecolor='black')

    # --- КЛЮЧЕВОЕ ИЗМЕНЕНИЕ: Устанавливаем логарифмическую шкалу для оси Y ---
    # Это позволяет наглядно сравнивать значения, различающиеся на порядки.
    ax.set_yscale('log')

    # Убираем научную нотацию (e.g., 10^1, 10^2) и показываем обычные числа (10, 100)
    # Это делает ось Y более читаемой для большинства людей.
    for axis in [ax.yaxis]:
        axis.set_major_formatter(ScalarFormatter())

    # --- ИЗМЕНЕНЫ ЗАГОЛОВКИ И ИМЯ ФАЙЛА ---
    ax.set_title('Сравнение времени выполнения алгоритмов (Логарифмическая шкала)', fontsize=18, fontweight='bold', pad=20)
    ax.set_xlabel('Дата производственного задания', fontsize=14, labelpad=15)
    ax.set_ylabel('Время выполнения, секунды (лог. шкала)', fontsize=14, labelpad=15)
    ax.legend(title='Алгоритм', title_fontsize='13', fontsize='11')
    ax.tick_params(axis='x', labelsize=12)
    ax.tick_params(axis='y', labelsize=12)

    # Подписи данных остаются такими же, они показывают реальные (не логарифмированные) значения
    for container in ax.containers:
        ax.bar_label(container, fmt='%.1f', label_type='edge', fontsize=10, padding=3)
    
    # Устанавливаем нижнюю границу оси Y чуть ниже минимального значения для красоты
    # Это необязательно, но предотвращает "прилипание" самых маленьких столбцов к оси X
    min_val = results_df['ExecutionTime_sec'].min()
    if min_val > 0:
        ax.set_ylim(bottom=min_val * 0.5)

    plt.tight_layout()

    # Новое имя файла, чтобы не перезаписать старый
    output_filename = 'algorithms_execution_time_comparison_log.png'
    plt.savefig(output_filename, dpi=300)
    plt.show()

    print(f"\nГрафик с логарифмической шкалой сохранен в файл: {output_filename}")