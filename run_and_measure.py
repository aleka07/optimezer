import os
import subprocess
import time
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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

# --- ДОБАВЛЕНО: Создаем окружение для дочерних процессов ---
# Копируем текущее окружение и добавляем переменную,
# которая заставляет Python использовать UTF-8 для stdout/stderr.
child_env = os.environ.copy()
child_env["PYTHONIOENCODING"] = "utf-8"


# --- 2. Сбор данных о времени выполнения ---
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
            # --- ИЗМЕНЕНО: Добавлен параметр `env=child_env` ---
            subprocess.run(
                [python_executable, script_name],
                cwd=date_path,
                check=True,
                capture_output=True,
                text=True,
                encoding='utf-8', # Это по-прежнему хорошая практика
                env=child_env      # <--- Вот ключевое изменение
            )
        except subprocess.CalledProcessError as e:
            print(f"\n    !!! ОШИБКА при выполнении '{script_name}' для даты {date}:")
            # Выводим ошибки, декодируя их явно, на всякий случай
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

# --- 3. Визуализация результатов (код без изменений) ---
if not execution_results:
    print("Не удалось собрать данные для построения графика. Проверьте пути и наличие файлов.")
else:
    results_df = pd.DataFrame(execution_results)

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(16, 9))

    sns.barplot(data=results_df, x='Date', y='ExecutionTime_sec', hue='Algorithm',
                palette='plasma', ax=ax, edgecolor='black')

    ax.set_title('Сравнение времени выполнения алгоритмов', fontsize=18, fontweight='bold', pad=20)
    ax.set_xlabel('Дата производственного задания', fontsize=14, labelpad=15)
    ax.set_ylabel('Время выполнения, секунды', fontsize=14, labelpad=15)
    ax.legend(title='Алгоритм', title_fontsize='13', fontsize='11')
    ax.tick_params(axis='x', labelsize=12)
    ax.tick_params(axis='y', labelsize=12)

    for container in ax.containers:
        ax.bar_label(container, fmt='%.1f', label_type='edge', fontsize=10, padding=3)

    plt.tight_layout()

    output_filename = 'algorithms_execution_time_comparison.png'
    plt.savefig(output_filename, dpi=300)
    plt.show()

    print(f"\nГрафик сравнения времени выполнения сохранен в файл: {output_filename}")