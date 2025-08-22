import math
import collections
import csv
import datetime
import json
import os

# =================================================================================
# ЧАСТЬ 1: ВХОДНЫЕ ДАННЫЕ (без изменений)
# =================================================================================

TECH_MAP_FILE = 'Fixed.json'

def load_tech_map(filename):
    try:
        with open(filename, 'r', encoding='utf-8') as f: data = json.load(f)
        tech_map_minutes = {}
        for p_data in data:
            if p_data['techprocess']:
                tech_map_minutes[p_data['product_name']] = {s['name']: math.ceil(s['time'] / 60.0) for s in p_data['techprocess']}
        print(f"Технологическая карта успешно загружена из '{filename}'.")
        return tech_map_minutes
    except Exception as e:
        print(f"ОШИБКА при загрузке тех. карты: {e}")
        return None

tech_map_data = load_tech_map(TECH_MAP_FILE)
if not tech_map_data: exit()

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

orders = {
    "Хлеб «Гречишный»": 318, "Багет с луком": 515, "Булочки": 244,
    "Хлеб «Формовой»": 556, "Лепешка с сыром и луком": 419,
}
print("\n--- Используется зафиксированный заказ для FIFO симуляции ---")
for p, q in orders.items(): print(f"- {p}: {q} шт.")
print("----------------------------------------------------------\n")

script_dir = os.path.dirname(os.path.abspath(__file__))
OUTPUT_CSV_FILE = os.path.join(script_dir, 'fifo_schedule_correct.csv')
OUTPUT_TXT_FILE = os.path.join(script_dir, 'fifo_summary_correct.txt')


# =================================================================================
# ЧАСТЬ 2: ИСПРАВЛЕННЫЙ СИМУЛЯТОР FIFO
# =================================================================================

def run_fifo_simulation_with_constraints(batches, machines, wait_constraints):
    machine_free_times = {stage: [0] * count for stage, count in machines.items()}
    schedule = []
    
    for batch in batches:
        previous_stage_end_time = 0
        
        for task in batch['tasks']:
            stage_name = task['stage_name']
            duration = task['duration']
            
            # Определяем временное окно для старта
            earliest_possible_start = previous_stage_end_time
            max_wait = float('inf') # По умолчанию ждать можно бесконечно
            prev_stage_name = task.get('prev_stage', None)
            if prev_stage_name:
                constraint_key = (prev_stage_name, stage_name)
                if constraint_key in wait_constraints:
                    max_wait = wait_constraints[constraint_key]
            
            latest_possible_start = earliest_possible_start + max_wait

            # Ищем машину, которая освободится ВНУТРИ нашего окна
            best_start_time = float('inf')
            best_machine_index = -1

            possible_starts = []
            for i, machine_free_at in enumerate(machine_free_times[stage_name]):
                # Задача может начаться не раньше, чем кончится предыдущая И освободится машина
                potential_start = max(earliest_possible_start, machine_free_at)
                
                # Если этот старт укладывается в наше технологическое окно
                if potential_start <= latest_possible_start:
                    possible_starts.append({'start': potential_start, 'index': i})

            if possible_starts:
                # Если есть подходящие машины, выбираем ту, что освободится раньше всех
                best_option = min(possible_starts, key=lambda x: x['start'])
                start_time = best_option['start']
                machine_index = best_option['index']
            else:
                # НИ ОДНА МАШИНА НЕ ОСВОБОЖДАЕТСЯ ВОВРЕМЯ. НАРУШЕНИЕ НЕИЗБЕЖНО.
                # Вынуждены ждать самую первую освободившуюся машину, даже если это поздно.
                earliest_machine_free_time = min(machine_free_times[stage_name])
                machine_index = machine_free_times[stage_name].index(earliest_machine_free_time)
                start_time = max(earliest_possible_start, earliest_machine_free_time)
                
                wait_time = start_time - earliest_possible_start
                print(f"!!! НАРУШЕНИЕ: Для '{batch['id']}' между '{prev_stage_name}'->'{stage_name}'. "
                      f"Ожидание {wait_time} мин > Макс. {max_wait} мин. Нет свободных машин.")

            end_time = start_time + duration
            machine_free_times[stage_name][machine_index] = end_time
            previous_stage_end_time = end_time

            schedule.append({
                "Batch_ID": batch['id'], "Stage": stage_name,
                "Start_Time_Min": start_time, "End_Time_Min": end_time, "Duration_Min": duration
            })

    return schedule

# --- Подготовка партий и запуск симуляции ---
all_batches = []
for product, quantity in sorted(orders.items()):
    num_batches = math.ceil(quantity / BATCH_SIZE)
    for i in range(num_batches):
        batch_id = f"{product}_batch_{i+1}"
        tasks = []
        product_stages = tech_map_data.get(product, {})
        prev_stage = None
        for stage_name, duration in product_stages.items():
            tasks.append({
                "stage_name": stage_name, 
                "duration": 1 if duration <= 0 else duration,
                "prev_stage": prev_stage
            })
            prev_stage = stage_name
        all_batches.append({"id": batch_id, "tasks": tasks})

print("Запуск ИСПРАВЛЕННОЙ FIFO симуляции...")
fifo_schedule = run_fifo_simulation_with_constraints(all_batches, machines_available, MAX_WAIT_CONSTRAINTS)
print("Симуляция завершена.")

# --- Обработка и вывод результатов (без изменений) ---
if fifo_schedule:
    makespan_minutes = max(task['End_Time_Min'] for task in fifo_schedule)
    tdelta = datetime.timedelta(minutes=makespan_minutes)
    days, hours, minutes = tdelta.days, tdelta.seconds // 3600, (tdelta.seconds // 60) % 60
    makespan_formatted = f"{days} дн {hours:02}:{minutes:02}" if days > 0 else f"{hours:02}:{minutes:02}"
    print(f"\n--- Результат FIFO симуляции ---")
    print(f"Общее время производства (Makespan): {makespan_minutes:.0f} минут ({makespan_formatted})")
    
    try:
        with open(OUTPUT_CSV_FILE, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=["Batch_ID", "Stage", "Start_Time_Min", "End_Time_Min", "Duration_Min"])
            writer.writeheader(); writer.writerows(fifo_schedule)
        print(f"\nДетальное расписание FIFO сохранено в: '{OUTPUT_CSV_FILE}'")
    except Exception as e: print(f"Ошибка записи CSV: {e}")

    try:
        with open(OUTPUT_TXT_FILE, 'w', encoding='utf-8') as f:
            f.write("--- Сводка по FIFO Расписанию (с соблюдением ограничений) ---\n\n")
            f.write(f"Общее время производства (Makespan): {makespan_minutes:.0f} минут ({makespan_formatted})\n")
            f.write(f"Всего партий: {len(all_batches)}\n")
        print(f"Сводная информация сохранена в: '{OUTPUT_TXT_FILE}'")
    except Exception as e: print(f"Ошибка записи TXT: {e}")