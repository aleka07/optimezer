import math
import collections
import csv
import datetime
from ortools.sat.python import cp_model

# 6 январа
# это версия уже без упаковки, также оан более реалистичные время делает

# 1. Define Input Data
tech_map_data = {
    # Этапы 1-3 (Подготовка смеси, 18-24 мин) -> Комбинирование: "0:21:00"
    # Этап 4 (Замес): Формовые "0:12:00", Остальные (10-11 мин) "0:10:30"
    # Этап 5 (Деление и формовка, 10-12 мин) -> Формовка: "0:11:00"
    # Этап 6 (Расстойка): Н/Д для "Остальных видов" -> "0:25:00" (среднее)
    # Выпекание: согласно таблице (16-17 -> "0:16:30", 18-19 -> "0:18:30")
    # Остывание/Упаковка: типовые значения

    # Убрана "Упаковка" из времен, так как мы ее опускаем
    "Мини формовой":        {"Комбинирование": "0:21:00", "Смешивание": "0:12:00", "Формовка": "0:11:00", "Расстойка": "0:25:00", "Выпекание": "0:45:00", "Остывание": "1:30:00"},
    "Формовой":             {"Комбинирование": "0:21:00", "Смешивание": "0:12:00", "Формовка": "0:11:00", "Расстойка": "0:25:00", "Выпекание": "0:45:00", "Остывание": "1:30:00"},
    "Домашний":             {"Комбинирование": "0:21:00", "Смешивание": "0:10:30", "Формовка": "0:11:00", "Расстойка": "0:25:00", "Выпекание": "0:17:00", "Остывание": "1:00:00"},
    "Семейный":             {"Комбинирование": "0:21:00", "Смешивание": "0:10:30", "Формовка": "0:11:00", "Расстойка": "0:25:00", "Выпекание": "0:17:00", "Остывание": "1:00:00"},
    "Славянский":           {"Комбинирование": "0:21:00", "Смешивание": "0:10:30", "Формовка": "0:11:00", "Расстойка": "0:25:00", "Выпекание": "0:17:00", "Остывание": "1:00:00"},
    "Жайлы":                {"Комбинирование": "0:21:00", "Смешивание": "0:10:30", "Формовка": "0:11:00", "Расстойка": "0:25:00", "Выпекание": "0:17:00", "Остывание": "1:00:00"},
    "Отрубной (общий)":     {"Комбинирование": "0:21:00", "Смешивание": "0:10:30", "Формовка": "0:11:00", "Расстойка": "0:25:00", "Выпекание": "0:17:00", "Остывание": "1:00:00"},
    "Любимый":              {"Комбинирование": "0:21:00", "Смешивание": "0:10:30", "Формовка": "0:11:00", "Расстойка": "0:25:00", "Выпекание": "0:17:00", "Остывание": "1:00:00"},
    "Датский":              {"Комбинирование": "0:21:00", "Смешивание": "0:10:30", "Формовка": "0:11:00", "Расстойка": "0:25:00", "Выпекание": "0:17:00", "Остывание": "1:00:00"},
    "Тартин (из таблицы)":  {"Комбинирование": "0:21:00", "Смешивание": "0:10:30", "Формовка": "0:11:00", "Расстойка": "0:25:00", "Выпекание": "0:17:00", "Остывание": "1:00:00"},
    "Зерновой Столичный":   {"Комбинирование": "0:21:00", "Смешивание": "0:10:30", "Формовка": "0:11:00", "Расстойка": "0:25:00", "Выпекание": "0:17:00", "Остывание": "1:00:00"},
    "Здоровье":             {"Комбинирование": "0:21:00", "Смешивание": "0:10:30", "Формовка": "0:11:00", "Расстойка": "0:25:00", "Выпекание": "0:18:00", "Остывание": "1:00:00"},
    "Бородинский":          {"Комбинирование": "0:21:00", "Смешивание": "0:10:30", "Формовка": "0:11:00", "Расстойка": "0:20:00", "Выпекание": "0:55:00", "Остывание": "2:00:00"},
    "Булочка для гамбургера большой с кунжутом": 
                            {"Комбинирование": "0:21:00", "Смешивание": "0:10:30", "Формовка": "0:11:00", "Расстойка": "0:25:00", "Выпекание": "0:22:00", "Остывание": "0:45:00"},
    "Булочка для хотдога штучно": 
                            {"Комбинирование": "0:21:00", "Смешивание": "0:10:30", "Формовка": "0:11:00", "Расстойка": "0:25:00", "Выпекание": "0:22:00", "Остывание": "0:45:00"},
    "Сэндвич":              {"Комбинирование": "0:21:00", "Смешивание": "0:10:30", "Формовка": "0:11:00", "Расстойка": "0:25:00", "Выпекание": "0:22:00", "Остывание": "0:45:00"},
    "Хлеб «Тартин бездрожжевой»":
                            {"Комбинирование": "0:21:00", "Смешивание": "0:10:30", "Формовка": "0:11:00", "Расстойка": "0:20:00", "Выпекание": "0:16:30", "Остывание": "1:00:00"},
    "Береке":               {"Комбинирование": "0:21:00", "Смешивание": "0:10:30", "Формовка": "0:11:00", "Расстойка": "0:20:00", "Выпекание": "0:16:30", "Остывание": "1:00:00"},
    "Баварский Деревенский Ржаной":
                            {"Комбинирование": "0:21:00", "Смешивание": "0:10:30", "Формовка": "0:11:00", "Расстойка": "0:30:00", "Выпекание": "0:18:30", "Остывание": "1:30:00"},
    "Плетенка":             {"Комбинирование": "0:21:00", "Смешивание": "0:10:30", "Формовка": "0:11:00", "Расстойка": "0:25:00", "Выпекание": "0:17:00", "Остывание": "1:00:00"},
    "Батон Верный":         {"Комбинирование": "0:21:00", "Смешивание": "0:10:30", "Формовка": "0:11:00", "Расстойка": "0:25:00", "Выпекание": "0:17:00", "Остывание": "1:00:00"},
    "Батон Нарезной":       {"Комбинирование": "0:21:00", "Смешивание": "0:10:30", "Формовка": "0:11:00", "Расстойка": "0:25:00", "Выпекание": "0:17:00", "Остывание": "1:00:00"},
    "Диета":                {"Комбинирование": "0:21:00", "Смешивание": "0:10:30", "Формовка": "0:11:00", "Расстойка": "0:37:30", "Выпекание": "0:35:00", "Остывание": "1:30:00"},
    "Багет отрубной":       {"Комбинирование": "0:21:00", "Смешивание": "0:10:30", "Формовка": "0:11:00", "Расстойка": "0:25:00", "Выпекание": "0:18:00", "Остывание": "0:45:00"},
    "Премиум":              {"Комбинирование": "0:21:00", "Смешивание": "0:10:30", "Формовка": "0:11:00", "Расстойка": "0:25:00", "Выпекание": "0:25:00", "Остывание": "1:15:00"},
    "Хлеб «Зерновой»":      {"Комбинирование": "0:21:00", "Смешивание": "0:10:30", "Формовка": "0:11:00", "Расстойка": "0:25:00", "Выпекание": "0:25:00", "Остывание": "1:15:00"},
    "Багет новый":          {"Комбинирование": "0:21:00", "Смешивание": "0:10:30", "Формовка": "0:11:00", "Расстойка": "0:25:00", "Выпекание": "0:25:00", "Остывание": "1:15:00"},
    "Чиабатта":             {"Комбинирование": "0:21:00", "Смешивание": "0:10:30", "Формовка": "0:11:00", "Расстойка": "0:25:00", "Выпекание": "0:25:00", "Остывание": "1:15:00"},
    "Багет луковый":        {"Комбинирование": "0:21:00", "Смешивание": "0:10:30", "Формовка": "0:11:00", "Расстойка": "0:20:00", "Выпекание": "0:18:00", "Остывание": "0:45:00"},
}

orders = {
    "Баварский Деревенский Ржаной": 25,     # Деревенский хлеб (9+5) + Баварский в упаковке (15)
    "Багет луковый": 51,
    "Багет новый": 0,                       # Нет реализации
    "Багет отрубной": 57,                   # Частичная реализация (в упаковке)
    "Премиум": 17,
    "Батон Верный": 33,                     # Частичная реализация (в упаковке)
    "Батон Нарезной": 338,                  # Нарезной (256) + упаковка (82)
    "Береке": 120,                          # Береке (77) + упаковка (43)
    "Бородинский": 145,                     # Только в упаковке (147)
    "Булочка для гамбургера большой/ с кунжутом": 82,  # Частичная реализация
    "Булочка для хотдога штучно": 134,
    "Датский": 38,                          # Базовый (5) + упаковка (33)
    "Диета": 282,                           # Диетический (215) + упаковка (67)
    "Домашний": 7,                          # В упаковке
    "Жайлы": 139,                           # Жайлы (108) + упаковка (31)
    "Здоровье": 15,                         # Базовый (5) + упаковка (10)
    "Любимый": 799,                         # Любимый (716) + упаковка (83)
    "Немецкий хлеб": 24,
    "Отрубной (общий)": 226,                # Базовый (132) + упаковка (94)
    "Плетенка": 118,                        # Все виды плетенок: (71+13+19+14+5 за минусом 5? Убрали нереализованные остатки)
    "Семейный": 262,                        # Семейный (155) + упаковка (107)
    "Славянский": 6,
    "Зерновой Столичный": 15,
    "Сэндвич": 1297,                        # Все виды сэндвичей (70+52+48+1070+7+10)
    "Хлеб «Тартин бездрожжевой»": 12,
    "Хлеб «Зерновой»": 150,                 # Зерновой (137) + упаковка (13)
    "Чиабатта": 21,
    "Формовой": 2353,                       # Формовой (2105) + упаковка (248)
    "Мини формовой": 327,                   # Мини формовой (244) + упаковка (83)
}






machines_available = {
    "Комбинирование": 2,
    "Смешивание": 3,
    "Формовка": 2,
    "Расстойка": 8,
    "Выпекание": 6,
    "Остывание": 25,
    # "Упаковка": 10, # Упаковку убираем из активных ресурсов, т.к. убираем этап
}

BATCH_SIZE = 100
STAGES = [ # Убрана "Упаковка"
    "Комбинирование", "Смешивание", "Формовка", "Расстойка",
    "Выпекание", "Остывание",
]

# --- Параметры ---
MAX_WAIT_COMBINING_MIXING_MIN = 1
MAX_WAIT_MIXING_FORMING_MIN = 1
MAX_WAIT_FORMING_PROOFING_MIN = 5 # Можно увеличить, если нет жестких ограничений
MAX_WAIT_PROOFING_BAKING_MIN = 5  # Можно увеличить

# Определяем критические этапы для ограничений по ожиданию.
# Так как "Упаковка" убрана, последний этап в цепочке - "Выпекание"
# Если бы были еще этапы, нужно было бы их корректно указать.
# Сейчас ограничения будут только до Выпекания.
CRITICAL_STAGE_BEFORE_0 = "Комбинирование"
CRITICAL_STAGE_AFTER_0 = "Смешивание"
CRITICAL_STAGE_BEFORE_1 = "Смешивание"
CRITICAL_STAGE_AFTER_1 = "Формовка"
CRITICAL_STAGE_BEFORE_2 = "Формовка"
CRITICAL_STAGE_AFTER_2 = "Расстойка"
CRITICAL_STAGE_BEFORE_3 = "Расстойка"
CRITICAL_STAGE_AFTER_3 = "Выпекание"


# --- Имя выходного файла ---
OUTPUT_CSV_FILE = 'production_schedule_v2.csv'
OUTPUT_TXT_FILE = 'production_summary_v2.txt'

# 2. Helper Function & Preprocessing
def time_str_to_minutes_int(time_str):
    try:
        parts = list(map(int, time_str.split(':')))
        if len(parts) == 3: h, m, s = parts; return round(h * 60 + m + s / 60.0)
        elif len(parts) == 2: m, s = parts; return round(m + s / 60.0)
        else: return 0
    except: return 0

tech_map_minutes_int = {}
for product, stages_data in tech_map_data.items():
    if product not in orders or orders[product] <= 0:
        continue
    tech_map_minutes_int[product] = {}
    for stage_name in STAGES: # Используем обновленный STAGES
        time_str = stages_data.get(stage_name, "0:00:00") # Упаковки здесь уже не будет
        duration = time_str_to_minutes_int(time_str)
        tech_map_minutes_int[product][stage_name] = duration

all_batches = []
proportional_time_stages = ["Комбинирование", "Формовка"]

for product, quantity_ordered in orders.items():
    if quantity_ordered <= 0: continue
    if product not in tech_map_data:
        print(f"Предупреждение: Продукт '{product}' из заказа отсутствует в технологической карте. Пропускается.")
        continue

    num_full_batches = quantity_ordered // BATCH_SIZE
    remaining_quantity = quantity_ordered % BATCH_SIZE
    
    total_batches_for_product = num_full_batches
    if remaining_quantity > 0:
        total_batches_for_product += 1

    for i in range(total_batches_for_product):
        batch_id = f"{product}_batch_{i+1}"
        is_last_partial_batch = (i == total_batches_for_product - 1) and (remaining_quantity > 0)
        current_batch_actual_size = BATCH_SIZE if not is_last_partial_batch else remaining_quantity
        
        batch_tasks = []
        for stage_index, stage_name in enumerate(STAGES): # Используем обновленный STAGES
            base_duration_for_100 = tech_map_minutes_int.get(product, {}).get(stage_name, 0)
            current_task_duration = base_duration_for_100

            if base_duration_for_100 > 0:
                if is_last_partial_batch: # Только для последней неполной партии
                    if stage_name in proportional_time_stages:
                        current_task_duration = math.ceil(base_duration_for_100 * (current_batch_actual_size / BATCH_SIZE))
                    # Для других этапов (Смешивание, Расстойка, Выпекание, Остывание)
                    # current_task_duration остается base_duration_for_100 (т.е. как для полной партии)
                
                # Гарантируем минимальную длительность 1, если базовая была > 0, а расчетная стала <=0
                if current_task_duration <= 0 and base_duration_for_100 > 0:
                    current_task_duration = 1
                
                if current_task_duration > 0:
                    batch_tasks.append({
                        "batch_id": batch_id, "stage_index": stage_index,
                        "stage_name": stage_name, "duration": current_task_duration,
                    })
        if batch_tasks:
            all_batches.append({"id": batch_id, "product": product, "tasks": batch_tasks})

if not all_batches:
    print("Нет партий для производства. Проверьте заказы и технологическую карту.")
    exit()

print(f"Всего партий сгенерировано: {len(all_batches)}")
num_tasks_total = sum(len(b['tasks']) for b in all_batches)
print(f"Всего задач (операций) с ненулевой длительностью: {num_tasks_total}")

# Расчет горизонта
horizon = 0
# Сумма всех реальных длительностей задач
for batch in all_batches:
    for task in batch['tasks']:
        horizon += task['duration']
# Добавим запас, основанный на количестве партий и средней длительности этапа (грубо)
# Это нужно, чтобы учесть параллелизм и возможное ожидание.
# Можно взять максимальную общую длительность одной партии * кол-во партий / мин_кол_машин + запас
# Но проще так: (сумма всех длительностей / мин_кол_машин_на_узком_месте) * некий_коэфф
min_machines = min(m_count for m_count in machines_available.values() if m_count > 0) # Минимальное кол-во машин на любом этапе
if min_machines > 0:
    horizon = math.ceil(horizon / min_machines) * 2 # Примерный запас для параллелизма и ожиданий
else: # Если где-то 0 машин, а задачи есть, будет неразрешимо, но горизонт нужен
    horizon = horizon * 2 
horizon += 1000 # Дополнительный буфер
print(f"Расчетный горизонт (макс. возможное время): {horizon} минут")


# 3. Model Creation
model = cp_model.CpModel()

# 4. Define Variables
task_vars = collections.defaultdict(dict)
task_lookup = {}

for i, batch in enumerate(all_batches):
    batch_id = batch['id']
    for task in batch['tasks']:
        stage_idx = task['stage_index']
        stage_name = task['stage_name']
        duration = task['duration']

        suffix = f'_{batch_id}_{stage_name}'
        start_var = model.NewIntVar(0, horizon, 'start' + suffix)
        end_var = model.NewIntVar(0, horizon, 'end' + suffix)
        interval_var = model.NewIntervalVar(start_var, duration, end_var, 'interval' + suffix)

        task_vars[i][stage_idx] = (start_var, end_var, interval_var)
        task_lookup[(batch_id, stage_name)] = (start_var, end_var, interval_var)

# 5. Define Constraints
# a) Sequence
for i, batch in enumerate(all_batches):
    sorted_tasks_for_batch = sorted(batch['tasks'], key=lambda t: t['stage_index'])
    for k in range(len(sorted_tasks_for_batch) - 1):
        current_task_info = sorted_tasks_for_batch[k]
        next_task_info = sorted_tasks_for_batch[k+1]
        curr_idx = current_task_info['stage_index']
        next_idx = next_task_info['stage_index']
        if curr_idx in task_vars[i] and next_idx in task_vars[i]:
            model.Add(task_vars[i][next_idx][0] >= task_vars[i][curr_idx][1])

# b) Resources
for stage_index, stage_name in enumerate(STAGES): # Используем обновленный STAGES
    machine_count = machines_available.get(stage_name)
    if machine_count is None or machine_count <= 0: # Упаковки уже нет в machines_available
        if any(stage_name == task['stage_name'] for batch in all_batches for task in batch['tasks']):
             print(f"Критическая ошибка: Для этапа '{stage_name}' не указано количество машин > 0, но задачи для него есть!")
        continue

    intervals_for_stage = []
    for i, batch in enumerate(all_batches):
         if stage_index in task_vars[i]:
             intervals_for_stage.append(task_vars[i][stage_index][2])

    if intervals_for_stage:
        demands = [1] * len(intervals_for_stage)
        model.AddCumulative(intervals_for_stage, demands, machine_count)

# c) Spoilage/Waiting Time Constraints
critical_constraints_defined = [
    (CRITICAL_STAGE_BEFORE_0, CRITICAL_STAGE_AFTER_0, MAX_WAIT_COMBINING_MIXING_MIN),
    (CRITICAL_STAGE_BEFORE_1, CRITICAL_STAGE_AFTER_1, MAX_WAIT_MIXING_FORMING_MIN),
    (CRITICAL_STAGE_BEFORE_2, CRITICAL_STAGE_AFTER_2, MAX_WAIT_FORMING_PROOFING_MIN),
    (CRITICAL_STAGE_BEFORE_3, CRITICAL_STAGE_AFTER_3, MAX_WAIT_PROOFING_BAKING_MIN),
]

for i, batch in enumerate(all_batches):
    batch_id = batch['id']
    for stage_before, stage_after, max_wait in critical_constraints_defined:
        # Проверяем, что оба этапа есть в STAGES (актуально, т.к. Упаковку убрали)
        if stage_before not in STAGES or stage_after not in STAGES:
            continue

        task_before_key = (batch_id, stage_before)
        task_after_key = (batch_id, stage_after)
        if task_before_key in task_lookup and task_after_key in task_lookup:
            model.Add(task_lookup[task_after_key][0] - task_lookup[task_before_key][1] <= max_wait)

# d) Makespan
makespan = model.NewIntVar(0, horizon, 'makespan')
last_stage_tasks_ends = []

for i, batch in enumerate(all_batches):
     if batch['tasks']:
        actual_last_stage_idx = batch['tasks'][-1]['stage_index'] # Последняя задача в списке задач партии
        if actual_last_stage_idx in task_vars[i]:
            last_stage_tasks_ends.append(task_vars[i][actual_last_stage_idx][1])

if last_stage_tasks_ends:
    model.AddMaxEquality(makespan, last_stage_tasks_ends)
else:
    model.Add(makespan == 0)

# 6. Define Objective
model.Minimize(makespan)

# 7. Solve Model
solver = cp_model.CpSolver()
solver.parameters.log_search_progress = True # Для отладки можно включить
# solver.parameters.max_time_in_seconds = 300.0 
print("\nЗапуск решателя...")
status = solver.Solve(model)
print("Решатель завершил работу.")

# 8. Process Results and Write to Files
if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
    optimal_makespan_minutes = solver.ObjectiveValue()
    print("\n--- Оптимальное/Допустимое Расписание Найдено ---")
    print(f"Минимальное Время Производства (Makespan): {optimal_makespan_minutes:.2f} минут")
    total_seconds_makespan = int(optimal_makespan_minutes * 60)
    tdelta = datetime.timedelta(seconds=total_seconds_makespan)
    days = tdelta.days
    hours, remainder = divmod(tdelta.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    makespan_formatted = ""
    if days > 0: makespan_formatted += f"{days} дн "
    makespan_formatted += f"{hours:02}:{minutes:02}:{seconds:02}"
    print(f"Что составляет примерно: {makespan_formatted}")

    schedule_data_for_output = []
    stage_order_map = {name: i for i, name in enumerate(STAGES)} 

    for i, batch in enumerate(all_batches):
        batch_id = batch['id']
        for task_info in batch['tasks']:
            stage_idx = task_info['stage_index']
            stage_name = task_info['stage_name']
            if stage_idx in task_vars[i]:
                start_val = solver.Value(task_vars[i][stage_idx][0])
                end_val = solver.Value(task_vars[i][stage_idx][1])
                duration_val = end_val - start_val
                schedule_data_for_output.append({
                    "Batch_ID": batch_id, "Stage": stage_name,
                    "Start_Time_Min": start_val, "End_Time_Min": end_val,
                    "Duration_Min": duration_val,
                    "Stage_Order": stage_order_map.get(stage_name, 999)
                })

    schedule_data_for_output.sort(key=lambda x: (x['Start_Time_Min'], x['Batch_ID'], x['Stage_Order']))

    csv_fieldnames = ["Batch_ID", "Stage", "Start_Time_Min", "End_Time_Min", "Duration_Min"]
    try:
        with open(OUTPUT_CSV_FILE, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_fieldnames)
            writer.writeheader()
            for row_data in schedule_data_for_output:
                row_to_write = {key: row_data[key] for key in csv_fieldnames}
                writer.writerow(row_to_write)
        print(f"\nРасписание успешно записано в CSV файл: '{OUTPUT_CSV_FILE}'")
    except Exception as e:
        print(f"\nОшибка записи CSV файла '{OUTPUT_CSV_FILE}': {e}")

    try:
        with open(OUTPUT_TXT_FILE, 'w', encoding='utf-8') as txtfile:
            txtfile.write("--- Сводка по Производственному Расписанию ---\n\n")
            txtfile.write(f"Статус решения: {'Оптимальное' if status == cp_model.OPTIMAL else 'Допустимое'}\n")
            txtfile.write(f"Общее время производства (Makespan): {optimal_makespan_minutes:.2f} минут\n")
            txtfile.write(f"Общее время производства (формат): {makespan_formatted}\n")
            txtfile.write(f"Всего партий: {len(all_batches)}\n")
            txtfile.write(f"Всего задач (операций) в расписании: {len(schedule_data_for_output)}\n")
            txtfile.write(f"\nПараметры модели (макс. время ожидания):\n")
            for sb, sa, mw in critical_constraints_defined:
                 if sb in STAGES and sa in STAGES: # Выводим только актуальные
                    txtfile.write(f"  - {sb} -> {sa}: {mw} мин\n")
            txtfile.write(f"\nРазмер партии (BATCH_SIZE): {BATCH_SIZE}\n")
            txtfile.write(f"\nЭтапы с пропорциональным временем для неполных партий: {', '.join(proportional_time_stages)}\n")
            txtfile.write(f"\nДоступные ресурсы (машины):\n")
            for stage_name_res, count in machines_available.items(): # machines_available теперь не содержит Упаковку
                txtfile.write(f"  - {stage_name_res}: {count}\n")
            txtfile.write(f"\nФайл с детальным расписанием: {OUTPUT_CSV_FILE}\n")
        print(f"Сводная информация успешно записана в TXT файл: '{OUTPUT_TXT_FILE}'")
    except Exception as e:
        print(f"\nОшибка записи TXT файла '{OUTPUT_TXT_FILE}': {e}")

elif status == cp_model.INFEASIBLE:
    print("\n--- Задача Неразрешима (INFEASIBLE) ---")
    print("Возможные причины:")
    print("  - Слишком жесткие ограничения по времени ожидания (MAX_WAIT_...).")
    print("  - Недостаточное количество машин (machines_available) для выполнения всех задач в срок, учитывая их длительности.")
    print("  - Слишком короткий горизонт планирования (horizon). Попробуйте увеличить его вручную.")
    print("  - Логическая ошибка в определении задач, их длительностей или ограничений.")
    print("Проверьте параметры и входные данные. Попробуйте ослабить ограничения.")
elif status == cp_model.MODEL_INVALID:
    print("\n--- Модель Некорректна (MODEL_INVALID) ---")
    print("Произошла ошибка при построении модели. Проверьте логику ограничений и определения переменных.")
    print("Сообщение от решателя:", solver.ResponseStats())
else:
    print(f"\n--- Решатель завершился со статусом: {status} ({cp_model.SolverEnumName(status)}) ---")
    print("Статус не является оптимальным или допустимым. Дополнительная информация:")
    print(solver.ResponseStats())