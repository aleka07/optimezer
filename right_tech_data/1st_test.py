import math
import collections
import csv
import datetime
import json
import random
import os
from ortools.sat.python import cp_model

# =================================================================================
# ЧАСТЬ 1: ОПРЕДЕЛЕНИЕ ВХОДНЫХ ДАННЫХ
# Источники данных четко прокомментированы, как вы просили.
# =================================================================================

# --- 1. Технологическая карта (Источник: ваш JSON-файл) ---
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
            if not process: # Пропускаем продукты без тех. процесса
                continue
            
            tech_map_minutes[product_name] = {}
            for stage in process:
                stage_name = stage['name']
                # Конвертируем время из секунд в целые минуты (округляем вверх)
                duration_minutes = math.ceil(stage['time'] / 60.0)
                tech_map_minutes[product_name][stage_name] = duration_minutes
        
        print(f"Технологическая карта успешно загружена из '{filename}'. Найдено {len(tech_map_minutes)} продуктов с тех. процессом.")
        return tech_map_minutes
    except FileNotFoundError:
        print(f"ОШИБКА: Файл '{filename}' не найден. Убедитесь, что он находится в той же папке, что и скрипт.")
        return None
    except Exception as e:
        print(f"ОШИБКА: Не удалось прочитать или обработать файл '{filename}'. Ошибка: {e}")
        return None

tech_map_data = load_tech_map(TECH_MAP_FILE)
if not tech_map_data:
    exit()

# --- 2. Доступное оборудование (Источник: ваш старый код) ---
# Названия этапов приведены в соответствие с JSON-файлом
machines_available = {
    "Combination": 3,
    "Mixing": 3,
    "Molding": 2,
    "Proofing": 8,
    "Baking": 6,
    "Cooling": 150,  # Большое число имитирует практически неограниченные места для остывания
}

# --- 3. Параметры производства (Источник: ваш старый код) ---
BATCH_SIZE = 100
# Этапы, время которых зависит от размера неполной партии
PROPORTIONAL_TIME_STAGES = ["Combination", "Molding"]
# Последовательность этапов, взятая из JSON
STAGES = ["Combination", "Mixing", "Molding", "Proofing", "Baking", "Cooling"]

# --- 4. Ограничения модели (Источник: статья и ваш старый код) ---
# Это реализация уравнения (4) из статьи: Si,j+1,k ≤ Eijk + Δj,j+1
# Максимальное допустимое время ожидания между этапами в минутах
MAX_WAIT_CONSTRAINTS = {
    ("Combination", "Mixing"): 10, # <-- ДОБАВЬТЕ ЭТУ СТРОКУ (ожидание после подготовки смеси не более 15 минут)
    ("Mixing", "Molding"): 5, # Тесто после замеса не должно долго лежать
    ("Molding", "Proofing"): 5, # Сформованные изделия должны сразу идти на расстойку
    ("Proofing", "Baking"): 5, # Самое критичное: нельзя передержать расстойку
    ("Baking", "Cooling"): 5, # Выпеченные изделия должны сразу остывать
}


# --- 5. Генерация случайного заказа для демонстрации ---
def generate_random_order(tech_map, num_products=5, min_qty=50, max_qty=800):
    """Создает случайный словарь заказов на основе доступных продуктов."""
    available_products = list(tech_map.keys())
    if len(available_products) < num_products:
        num_products = len(available_products)
        
    selected_products = random.sample(available_products, num_products)
    order = {prod: random.randint(min_qty, max_qty) for prod in selected_products}
    print("\n--- Сгенерирован случайный заказ ---")
    for p, q in order.items():
        print(f"- {p}: {q} шт.")
    print("-------------------------------------\n")
    return order

# orders = generate_random_order(tech_map_data)
orders = {
    "Хлеб «Гречишный»": 318,
    "Багет с луком": 515,
    "Булочки": 244,
    "Хлеб «Формовой»": 556,
    "Лепешка с сыром и луком": 419
}
print("\n--- Используется зафиксированный заказ ---")
for p, q in orders.items():
    print(f"- {p}: {q} шт.")
print("-------------------------------------\n")



# --- Имена выходных файлов ---
script_dir = os.path.dirname(os.path.abspath(__file__))
OUTPUT_CSV_FILE = os.path.join(script_dir, 'production_schedule_optimized.csv')
OUTPUT_TXT_FILE = os.path.join(script_dir, 'production_summary_optimized.txt')

# =================================================================================
# ЧАСТЬ 2: ПОДГОТОВКА ДАННЫХ И МОДЕЛИРОВАНИЕ (ИСПРАВЛЕННАЯ ВЕРСИЯ)
# =================================================================================

# --- Формирование партий ---
all_batches = []
for product, quantity_ordered in orders.items():
    if product not in tech_map_data:
        print(f"Предупреждение: Продукт '{product}' из заказа отсутствует в тех. карте. Пропускается.")
        continue

    num_full_batches = quantity_ordered // BATCH_SIZE
    remaining_quantity = quantity_ordered % BATCH_SIZE
    total_batches_for_product = num_full_batches + (1 if remaining_quantity > 0 else 0)

    for i in range(total_batches_for_product):
        batch_id = f"{product}_batch_{i+1}"
        is_last_partial_batch = (i == total_batches_for_product - 1) and (remaining_quantity > 0)
        current_batch_size = remaining_quantity if is_last_partial_batch else BATCH_SIZE
        
        batch_tasks = []
        # Важно: теперь мы итерируемся по задачам из тех. карты продукта, а не по общему списку STAGES
        # Это гарантирует, что мы берем только нужные этапы в правильном порядке для каждого продукта.
        product_stages = tech_map_data.get(product, {})
        if not product_stages:
            continue

        for stage_name, base_duration in product_stages.items():
            
            # ИСПРАВЛЕНИЕ: Мы больше не пропускаем задачи с нулевой длительностью.
            # Вместо этого, мы дадим им минимальную длительность в 1 минуту, чтобы сохранить цепочку.
            
            current_duration = base_duration
            if is_last_partial_batch and stage_name in PROPORTIONAL_TIME_STAGES:
                current_duration = math.ceil(base_duration * (current_batch_size / BATCH_SIZE))
            
            # Если базовая или расчетная длительность равна 0, назначаем 1 минуту.
            # Это сохраняет последовательность операций.
            if current_duration <= 0:
                current_duration = 1
            
            batch_tasks.append({
                "stage_name": stage_name, "duration": current_duration,
            })
        
        if batch_tasks:
            all_batches.append({"id": batch_id, "tasks": batch_tasks})

if not all_batches:
    print("Нет партий для производства. Проверьте заказы и тех. карту.")
    exit()

print(f"Всего партий сгенерировано: {len(all_batches)}")
num_tasks_total = sum(len(b['tasks']) for b in all_batches)
print(f"Всего задач (операций): {num_tasks_total}")

# --- Расчет горизонта планирования (остается без изменений) ---
horizon = sum(task['duration'] for batch in all_batches for task in batch['tasks'])
min_machines = min(m for m in machines_available.values() if m > 0)
horizon = math.ceil(horizon / min_machines) * 2 + 1440 # Запас в 1 день
print(f"Расчетный горизонт планирования: {horizon} минут")

# =================================================================================
# ЧАСТЬ 3: СОЗДАНИЕ И РЕШЕНИЕ МОДЕЛИ CP-SAT
# =================================================================================
model = cp_model.CpModel()

# --- Создание переменных ---
# Словарь для хранения всех переменных задач для легкого доступа
# task_vars[batch_id][stage_name] -> (start_var, end_var, interval_var)
task_vars = collections.defaultdict(dict)

for batch in all_batches:
    for task in batch['tasks']:
        batch_id = batch['id']
        stage_name = task['stage_name']
        duration = task['duration']
        
        suffix = f'_{batch_id}_{stage_name}'
        start_var = model.NewIntVar(0, horizon, 'start' + suffix)
        end_var = model.NewIntVar(0, horizon, 'end' + suffix)
        interval_var = model.NewIntervalVar(start_var, duration, end_var, 'interval' + suffix)
        
        task_vars[batch_id][stage_name] = (start_var, end_var, interval_var)

# --- Определение ограничений ---

# a) Последовательность этапов внутри одной партии (Уравнение 3 из статьи)
# Si,j+1,k ≥ Eijk (минимальное время ожидания W=0)
for batch in all_batches:
    batch_id = batch['id']
    for i in range(len(STAGES) - 1):
        current_stage = STAGES[i]
        next_stage = STAGES[i+1]
        
        if current_stage in task_vars[batch_id] and next_stage in task_vars[batch_id]:
            current_end_var = task_vars[batch_id][current_stage][1]
            next_start_var = task_vars[batch_id][next_stage][0]
            model.Add(next_start_var >= current_end_var)

# b) Ограничения на ресурсы (No-Overlap) - соответствует Уравнениям (5) и (6) из статьи
for stage_name, machine_count in machines_available.items():
    intervals_for_stage = []
    for batch in all_batches:
        if stage_name in task_vars[batch['id']]:
            intervals_for_stage.append(task_vars[batch['id']][stage_name][2])
    
    if intervals_for_stage:
        model.AddCumulative(intervals_for_stage, [1] * len(intervals_for_stage), machine_count)

# c) Ограничения на максимальное время ожидания (Уравнение 4 из статьи)
# Si,j+1,k ≤ Eijk + Δj,j+1
for batch in all_batches:
    batch_id = batch['id']
    for (stage_before, stage_after), max_wait in MAX_WAIT_CONSTRAINTS.items():
        if stage_before in task_vars[batch_id] and stage_after in task_vars[batch_id]:
            end_before = task_vars[batch_id][stage_before][1]
            start_after = task_vars[batch_id][stage_after][0]
            model.Add(start_after <= end_before + max_wait)

# d) Определение Makespan (общее время выполнения) - Уравнение (7) из статьи
makespan = model.NewIntVar(0, horizon, 'makespan')
last_stage_tasks_ends = []
for batch in all_batches:
    # Последний этап для конкретной партии - это последний в ее списке задач
    last_stage_for_batch = batch['tasks'][-1]['stage_name']
    last_stage_tasks_ends.append(task_vars[batch['id']][last_stage_for_batch][1])

if last_stage_tasks_ends:
    model.AddMaxEquality(makespan, last_stage_tasks_ends)
else:
    model.Add(makespan == 0)

# --- Целевая функция: минимизировать Makespan (Уравнение 1 из статьи) ---
model.Minimize(makespan)

# --- Решение модели ---
solver = cp_model.CpSolver()
solver.parameters.log_search_progress = True

# ДОБАВЛЕНО ОГРАНИЧЕНИЕ: Решатель остановится через 60 секунд
# и вернет лучшее решение, найденное за это время.
solver.parameters.max_time_in_seconds = 60.0

print("\nЗапуск решателя с ограничением в 60 секунд...")
status = solver.Solve(model)


# =================================================================================
# ЧАСТЬ 4: ОБРАБОТКА И ВЫВОД РЕЗУЛЬТАТОВ
# =================================================================================
if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
    optimal_makespan_minutes = solver.ObjectiveValue()
    print("\n--- Оптимальное/Допустимое Расписание Найдено ---")
    print(f"Минимальное Время Производства (Makespan): {optimal_makespan_minutes:.0f} минут")
    
    tdelta = datetime.timedelta(minutes=optimal_makespan_minutes)
    days = tdelta.days
    hours, remainder = divmod(tdelta.seconds, 3600)
    minutes, _ = divmod(remainder, 60)
    makespan_formatted = f"{days} дн {hours:02}:{minutes:02}" if days > 0 else f"{hours:02}:{minutes:02}"
    print(f"Что составляет примерно: {makespan_formatted}")

    # --- Подготовка данных для вывода ---
    schedule_data = []
    stage_order_map = {name: i for i, name in enumerate(STAGES)}
    for batch in all_batches:
        for task in batch['tasks']:
            stage_name = task['stage_name']
            start_val = solver.Value(task_vars[batch['id']][stage_name][0])
            end_val = solver.Value(task_vars[batch['id']][stage_name][1])
            schedule_data.append({
                "Batch_ID": batch['id'],
                "Stage": stage_name,
                "Start_Time_Min": start_val,
                "End_Time_Min": end_val,
                "Duration_Min": end_val - start_val,
                "Stage_Order": stage_order_map.get(stage_name, 99)
            })
    
    schedule_data.sort(key=lambda x: (x['Start_Time_Min'], x['Batch_ID'], x['Stage_Order']))

    # --- Запись в CSV ---
    try:
        with open(OUTPUT_CSV_FILE, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=["Batch_ID", "Stage", "Start_Time_Min", "End_Time_Min", "Duration_Min"])
            writer.writeheader()
            writer.writerows([{k: v for k, v in row.items() if k != 'Stage_Order'} for row in schedule_data])
        print(f"\nДетальное расписание сохранено в CSV: '{OUTPUT_CSV_FILE}'")
    except Exception as e:
        print(f"\nОшибка записи CSV файла: {e}")

    # --- Запись в TXT ---
    try:
        with open(OUTPUT_TXT_FILE, 'w', encoding='utf-8') as f:
            f.write("--- Сводка по Оптимизированному Расписанию ---\n\n")
            f.write(f"Статус решения: {'Оптимальное' if status == cp_model.OPTIMAL else 'Допустимое'}\n")
            f.write(f"Общее время производства (Makespan): {optimal_makespan_minutes:.0f} минут ({makespan_formatted})\n")
            f.write(f"Всего партий в заказе: {len(all_batches)}\n")
            f.write(f"Всего операций в расписании: {len(schedule_data)}\n")
            f.write("\n--- Использованные параметры ---\n")
            f.write(f"Размер стандартной партии: {BATCH_SIZE} шт.\n")
            f.write(f"Источник тех. карты: {TECH_MAP_FILE}\n")
            f.write("\nОграничения на макс. время ожидания (мин):\n")
            for (s1, s2), t in MAX_WAIT_CONSTRAINTS.items():
                f.write(f"  - {s1} -> {s2}: {t} мин\n")
            f.write("\nДоступные ресурсы (машины):\n")
            for stage, count in machines_available.items():
                f.write(f"  - {stage}: {count}\n")
            f.write(f"\nФайл с детальным расписанием: {OUTPUT_CSV_FILE}\n")
        print(f"Сводная информация сохранена в TXT: '{OUTPUT_TXT_FILE}'")
    except Exception as e:
        print(f"\nОшибка записи TXT файла: {e}")

elif status == cp_model.INFEASIBLE:
    print("\n--- ЗАДАЧА НЕРАЗРЕШИМА (INFEASIBLE) ---")
    print("Решение не найдено. Возможные причины:")
    print(" - Слишком жесткие ограничения на время ожидания (MAX_WAIT_CONSTRAINTS).")
    print(" - Недостаточно машин для выполнения заказа в заданном горизонте.")
    print(" - Слишком короткий горизонт планирования (horizon).")
    print("Попробуйте ослабить ограничения или увеличить количество ресурсов.")
else:
    print(f"\n--- Решатель завершился со статусом: {solver.StatusName(status)} ---")