import math
import collections
import csv
import datetime
import json
import os

# =================================================================================
# ЧАСТЬ 1: ОПРЕДЕЛЕНИЕ ВХОДНЫХ ДАННЫХ (идентично оптимизатору)
# =================================================================================

TECH_MAP_FILE = 'Fixed.json'

def load_tech_map(filename):
    """Загружает технологическую карту из JSON и конвертирует время в минуты."""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        tech_map_minutes = {}
        for product_data in data:
            product_name = product_data['product_name']
            process = product_data['techprocess']
            if process:
                tech_map_minutes[product_name] = {stage['name']: math.ceil(stage['time'] / 60.0) for stage in process}
        
        print(f"Технологическая карта успешно загружена из '{filename}'.")
        return tech_map_minutes
    except Exception as e:
        print(f"ОШИБКА при загрузке тех. карты: {e}")
        return None

tech_map_data = load_tech_map(TECH_MAP_FILE)
if not tech_map_data:
    exit()

# --- Ресурсы и ограничения (такие же, как у оптимизатора) ---
machines_available = {
    "Combination": 2, "Mixing": 3, "Molding": 2,
    "Proofing": 8, "Baking": 6, "Cooling": 150,
}

MAX_WAIT_CONSTRAINTS = {
    ("Combination", "Mixing"): 8,
    ("Mixing", "Molding"): 8,
    ("Molding", "Proofing"): 8,
    ("Proofing", "Baking"): 5,
    ("Baking", "Cooling"): 5,
}

BATCH_SIZE = 100
PROPORTIONAL_TIME_STAGES = ["Combination", "Molding"]

# --- Используем тот же самый зафиксированный заказ для сравнения ---
orders = {
    "Хлеб «Гречишный»": 318,
    "Багет с луком": 515,
    "Булочки": 244,
    "Хлеб «Формовой»": 556,
    "Лепешка с сыром и луком": 419,
}
print("\n--- Используется зафиксированный заказ для FIFO симуляции ---")
for p, q in orders.items():
    print(f"- {p}: {q} шт.")
print("----------------------------------------------------------\n")

# --- Имена выходных файлов для FIFO ---
script_dir = os.path.dirname(os.path.abspath(__file__))
OUTPUT_CSV_FILE = os.path.join(script_dir, 'fifo_schedule.csv')
OUTPUT_TXT_FILE = os.path.join(script_dir, 'fifo_summary.txt')


# =================================================================================
# ЧАСТЬ 2: СИМУЛЯТОР FIFO
# =================================================================================

def run_fifo_simulation(batches, machines, wait_constraints):
    """
    Проводит симуляцию производственного процесса по принципу FIFO.
    """
    # Отслеживаем, когда каждая машина освободится.
    # Например: {'Mixing': [0, 0, 0], 'Baking': [0, 0, 0, 0, 0, 0]}
    machine_free_times = {
        stage: [0] * count for stage, count in machines.items()
    }
    
    schedule = []
    
    for batch in batches:
        previous_stage_end_time = 0
        
        for task in batch['tasks']:
            stage_name = task['stage_name']
            duration = task['duration']
            
            # Найти самую раннюю машину, которая освободится для этого этапа
            machine_times = machine_free_times[stage_name]
            earliest_machine_free_time = min(machine_times)
            machine_index = machine_times.index(earliest_machine_free_time)

            # Задача может начаться не раньше, чем закончится предыдущий этап И освободится машина
            start_time = max(previous_stage_end_time, earliest_machine_free_time)

            # Проверка ограничения на максимальное ожидание
            wait_time = start_time - previous_stage_end_time
            prev_stage_name = task.get('prev_stage', None)
            if prev_stage_name:
                max_wait = wait_constraints.get((prev_stage_name, stage_name))
                if max_wait is not None and wait_time > max_wait:
                    print(f"ПРЕДУПРЕЖДЕНИЕ: Нарушено ограничение ожидания для '{batch['id']}' "
                          f"между '{prev_stage_name}' и '{stage_name}'. "
                          f"Ожидание: {wait_time} мин > Макс: {max_wait} мин.")
                    # В реальной симуляции это может означать брак, но мы продолжаем расчет.

            end_time = start_time + duration
            
            # Обновляем время, когда эта машина освободится
            machine_free_times[stage_name][machine_index] = end_time
            
            # Запоминаем время окончания для следующего этапа этой же партии
            previous_stage_end_time = end_time

            schedule.append({
                "Batch_ID": batch['id'],
                "Stage": stage_name,
                "Start_Time_Min": start_time,
                "End_Time_Min": end_time,
                "Duration_Min": duration
            })

    return schedule

# --- Подготовка партий (идентична оптимизатору) ---
all_batches = []
for product, quantity in sorted(orders.items()): # Сортируем для стабильного FIFO порядка
    num_batches = math.ceil(quantity / BATCH_SIZE)
    for i in range(num_batches):
        batch_id = f"{product}_batch_{i+1}"
        # ... (здесь упрощенная логика размера, можно скопировать точную из optimize_bakery.py если нужно)
        
        tasks = []
        product_stages = tech_map_data.get(product, {})
        prev_stage = None
        for stage_name, duration in product_stages.items():
            tasks.append({
                "stage_name": stage_name, 
                "duration": 1 if duration <= 0 else duration, # Длительность не может быть 0
                "prev_stage": prev_stage
            })
            prev_stage = stage_name
        
        all_batches.append({"id": batch_id, "tasks": tasks})

# --- Запуск симуляции и обработка результатов ---
print("Запуск FIFO симуляции...")
fifo_schedule = run_fifo_simulation(all_batches, machines_available, MAX_WAIT_CONSTRAINTS)
print("Симуляция завершена.")

if fifo_schedule:
    # Расчет Makespan
    makespan_minutes = max(task['End_Time_Min'] for task in fifo_schedule)
    tdelta = datetime.timedelta(minutes=makespan_minutes)
    days = tdelta.days
    hours, remainder = divmod(tdelta.seconds, 3600)
    minutes, _ = divmod(remainder, 60)
    makespan_formatted = f"{days} дн {hours:02}:{minutes:02}" if days > 0 else f"{hours:02}:{minutes:02}"

    print(f"\n--- Результат FIFO симуляции ---")
    print(f"Общее время производства (Makespan): {makespan_minutes:.0f} минут ({makespan_formatted})")

    # --- Сохранение в файлы ---
    try:
        with open(OUTPUT_CSV_FILE, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=["Batch_ID", "Stage", "Start_Time_Min", "End_Time_Min", "Duration_Min"])
            writer.writeheader()
            writer.writerows(fifo_schedule)
        print(f"\nДетальное расписание FIFO сохранено в: '{OUTPUT_CSV_FILE}'")
    except Exception as e:
        print(f"Ошибка записи CSV: {e}")

    try:
        with open(OUTPUT_TXT_FILE, 'w', encoding='utf-8') as f:
            f.write("--- Сводка по FIFO Расписанию ---\n\n")
            f.write(f"Общее время производства (Makespan): {makespan_minutes:.0f} минут ({makespan_formatted})\n")
            f.write(f"Всего партий: {len(all_batches)}\n")
        print(f"Сводная информация сохранена в: '{OUTPUT_TXT_FILE}'")
    except Exception as e:
        print(f"Ошибка записи TXT: {e}")