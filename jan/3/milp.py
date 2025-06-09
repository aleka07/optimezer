import math
import collections
import csv
import datetime
import itertools # For creating pairs of tasks
from pulp import LpProblem, LpVariable, lpSum, LpMinimize, LpStatus, PULP_CBC_CMD

# 1. Define Input Data (Copied from your script)
tech_map_data = {
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
    "Формовой": 2185, "Мини формовой": 378, "Бородинский": 548, "Домашний": 10,
    "Багет луковый": 57, "Багет новый": 302, "Багет отрубной": 60, "Премиум": 15,
    "Батон Верный": 48, "Батон Нарезной": 335, "Береке": 170, "Жайлы": 165,
    "Датский": 31, "Баварский Деревенский Ржаной": 10, "Диета": 339, "Здоровье": 17,
    "Семейный": 334, "Славянский": 6, "Немецкий хлеб": 7, "Зерновой Столичный": 13,
    "Булочка для гамбургера большой с кунжутом": 1530, "Булочка для хотдога штучно": 196,
    "Плетенка": 139, "Сэндвич": 1359, "Хлеб «Тартин бездрожжевой»": 14,
    "Хлеб «Зерновой»": 160, "Чиабатта": 16,
}
machines_available = {
    "Комбинирование": 2, "Смешивание": 3, "Формовка": 2,
    "Расстойка": 8, "Выпекание": 6, "Остывание": 25,
}
BATCH_SIZE = 100
STAGES = [
    "Комбинирование", "Смешивание", "Формовка", "Расстойка",
    "Выпекание", "Остывание",
]
proportional_time_stages = ["Комбинирование", "Формовка"]
MAX_WAIT_COMBINING_MIXING_MIN = 1
MAX_WAIT_MIXING_FORMING_MIN = 1
MAX_WAIT_FORMING_PROOFING_MIN = 5
MAX_WAIT_PROOFING_BAKING_MIN = 5
CRITICAL_STAGE_BEFORE_0 = "Комбинирование"; CRITICAL_STAGE_AFTER_0 = "Смешивание"
CRITICAL_STAGE_BEFORE_1 = "Смешивание"; CRITICAL_STAGE_AFTER_1 = "Формовка"
CRITICAL_STAGE_BEFORE_2 = "Формовка"; CRITICAL_STAGE_AFTER_2 = "Расстойка"
CRITICAL_STAGE_BEFORE_3 = "Расстойка"; CRITICAL_STAGE_AFTER_3 = "Выпекание"

OUTPUT_CSV_FILE = 'production_schedule_milp.csv'
OUTPUT_TXT_FILE = 'production_summary_milp.txt'
SOLVER_TIME_LIMIT_SECONDS = 300 # Ограничение времени для решателя

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
    if product not in orders or orders[product] <= 0: continue
    tech_map_minutes_int[product] = {
        stage_name: time_str_to_minutes_int(stages_data.get(stage_name, "0:0:0"))
        for stage_name in STAGES
    }

# --- Создание плоского списка всех задач (операций) ---
# Каждая задача будет иметь уникальный ID
all_task_operations = []
task_counter = 0
batch_details = [] # Сохраним детали для облегчения доступа

for product, quantity_ordered in orders.items():
    if quantity_ordered <= 0 or product not in tech_map_data: continue

    num_full_batches = quantity_ordered // BATCH_SIZE
    remaining_quantity = quantity_ordered % BATCH_SIZE
    total_batches_for_product = num_full_batches + (1 if remaining_quantity > 0 else 0)

    for i in range(total_batches_for_product):
        batch_id_str = f"{product}_batch_{i+1}"
        is_last_partial = (i == total_batches_for_product - 1) and (remaining_quantity > 0)
        current_batch_size = remaining_quantity if is_last_partial else BATCH_SIZE
        
        batch_ops = []
        for stage_idx, stage_name in enumerate(STAGES):
            base_duration = tech_map_minutes_int.get(product, {}).get(stage_name, 0)
            if base_duration <= 0: continue

            current_duration = base_duration
            if is_last_partial and stage_name in proportional_time_stages:
                current_duration = math.ceil(base_duration * (current_batch_size / BATCH_SIZE))
            
            if current_duration <= 0 and base_duration > 0: current_duration = 1
            
            if current_duration > 0:
                task_id = f"task_{task_counter}"
                task_op_data = {
                    "id": task_id, "batch_id_str": batch_id_str, "product": product,
                    "stage_name": stage_name, "stage_idx_in_batch": stage_idx, 
                    "duration": current_duration
                }
                all_task_operations.append(task_op_data)
                batch_ops.append(task_op_data)
                task_counter += 1
        if batch_ops:
            batch_details.append({"batch_id_str": batch_id_str, "ops": batch_ops})


if not all_task_operations:
    print("Нет задач для планирования.")
    exit()

print(f"Всего задач (операций) для MILP: {len(all_task_operations)}")

# Расчет горизонта (Big M)
horizon = sum(op['duration'] for op in all_task_operations) # Сумма всех длительностей
min_machines_overall = min(m_count for m_count in machines_available.values() if m_count > 0)
if min_machines_overall > 0:
    horizon = math.ceil(horizon / min_machines_overall) * 3 # Запас
else:
    horizon = horizon * 3
horizon += 1000 # Дополнительный буфер
BIG_M = horizon
print(f"Расчетный горизонт (BIG_M): {BIG_M} минут")


# 3. Model Creation (PuLP)
model = LpProblem("BakerySchedulingMILP", LpMinimize)

# 4. Define Variables
start_vars = {
    op['id']: LpVariable(f"start_{op['id']}", lowBound=0, upBound=BIG_M, cat='Continuous')
    for op in all_task_operations
}

# Переменные назначения задачи на экземпляр машины
# assign_vars[task_id][machine_instance_num_for_stage_type]
assign_vars = {}
for op in all_task_operations:
    op_id = op['id']
    stage = op['stage_name']
    num_machines_for_stage = machines_available.get(stage, 0)
    if num_machines_for_stage > 0:
        assign_vars[op_id] = {
            m_idx: LpVariable(f"assign_{op_id}_m{m_idx}", cat='Binary')
            for m_idx in range(num_machines_for_stage)
        }

# Переменные порядка для задач на одной машине
# order_vars[task1_id][task2_id] = 1 if task1 before task2, 0 if task2 before task1
order_vars = {}
tasks_by_stage = collections.defaultdict(list)
for op in all_task_operations:
    tasks_by_stage[op['stage_name']].append(op)

for stage_name, ops_on_stage in tasks_by_stage.items():
    if machines_available.get(stage_name, 0) > 0 and len(ops_on_stage) > 1:
        for op1, op2 in itertools.combinations(ops_on_stage, 2):
            # Ключ должен быть уникальным и не зависеть от порядка op1, op2 в combinations
            key = tuple(sorted((op1['id'], op2['id'])))
            if key not in order_vars: # Создаем только один раз для пары
                 order_vars[key] = LpVariable(f"order_{key[0]}_{key[1]}", cat='Binary')


makespan_var = LpVariable("makespan", lowBound=0, upBound=BIG_M, cat='Continuous')

# 5. Define Constraints

# a) Sequence within each batch
for batch_data in batch_details:
    ops_in_batch = sorted(batch_data['ops'], key=lambda x: x['stage_idx_in_batch'])
    for i in range(len(ops_in_batch) - 1):
        op_curr = ops_in_batch[i]
        op_next = ops_in_batch[i+1]
        model += start_vars[op_next['id']] >= start_vars[op_curr['id']] + op_curr['duration'], \
                 f"seq_{op_curr['id']}_{op_next['id']}"

# b) Assignment to machines: each task must be assigned to exactly one machine of its type
for op in all_task_operations:
    op_id = op['id']
    stage = op['stage_name']
    if machines_available.get(stage, 0) > 0:
        model += lpSum(assign_vars[op_id][m_idx] for m_idx in assign_vars[op_id]) == 1, \
                 f"assign_sum_{op_id}"

# c) No-Overlap on machines (Big-M)
# This is the most complex part and generates many constraints
for stage_name, ops_on_stage in tasks_by_stage.items():
    num_machines = machines_available.get(stage_name, 0)
    if num_machines > 0 and len(ops_on_stage) > 1:
        for op1, op2 in itertools.combinations(ops_on_stage, 2):
            op1_id, op2_id = op1['id'], op2['id']
            # Получаем переменную порядка для этой пары
            order_key = tuple(sorted((op1_id, op2_id)))
            # Если order_vars[order_key] = 1, то task_with_id=order_key[0] идет перед task_with_id=order_key[1]
            # Иначе task_with_id=order_key[1] идет перед task_with_id=order_key[0]
            
            # Определим, кто из op1, op2 соответствует order_key[0] и order_key[1]
            # Это нужно, чтобы правильно использовать order_var
            task_A_id, task_B_id = order_key[0], order_key[1]
            task_A_duration = next(o['duration'] for o in all_task_operations if o['id'] == task_A_id)
            task_B_duration = next(o['duration'] for o in all_task_operations if o['id'] == task_B_id)

            for m_idx in range(num_machines):
                # If op1 and op2 are on the same machine m_idx:
                # start_A + dur_A <= start_B + M * (1-order_AB) + M * (2 - assign_A_m - assign_B_m)
                # start_B + dur_B <= start_A + M * order_AB      + M * (2 - assign_A_m - assign_B_m)
                # где order_AB = 1 если A < B, 0 если B < A
                
                # Ограничение 1: A перед B (если order_vars[order_key] == 1)
                model += start_vars[task_A_id] + task_A_duration <= \
                         start_vars[task_B_id] + \
                         BIG_M * (1 - order_vars[order_key]) + \
                         BIG_M * (2 - assign_vars[task_A_id][m_idx] - assign_vars[task_B_id][m_idx]), \
                         f"no_overlap_{task_A_id}_{task_B_id}_m{m_idx}_A_before_B"

                # Ограничение 2: B перед A (если order_vars[order_key] == 0)
                model += start_vars[task_B_id] + task_B_duration <= \
                         start_vars[task_A_id] + \
                         BIG_M * order_vars[order_key] + \
                         BIG_M * (2 - assign_vars[task_A_id][m_idx] - assign_vars[task_B_id][m_idx]), \
                         f"no_overlap_{task_A_id}_{task_B_id}_m{m_idx}_B_before_A"

# d) Spoilage/Waiting Time Constraints
critical_constraints_params = [
    (CRITICAL_STAGE_BEFORE_0, CRITICAL_STAGE_AFTER_0, MAX_WAIT_COMBINING_MIXING_MIN),
    (CRITICAL_STAGE_BEFORE_1, CRITICAL_STAGE_AFTER_1, MAX_WAIT_MIXING_FORMING_MIN),
    (CRITICAL_STAGE_BEFORE_2, CRITICAL_STAGE_AFTER_2, MAX_WAIT_FORMING_PROOFING_MIN),
    (CRITICAL_STAGE_BEFORE_3, CRITICAL_STAGE_AFTER_3, MAX_WAIT_PROOFING_BAKING_MIN),
]
for batch_data in batch_details:
    ops_map = {op['stage_name']: op for op in batch_data['ops']}
    for stage_before, stage_after, max_wait in critical_constraints_params:
        if stage_before in ops_map and stage_after in ops_map:
            op_before = ops_map[stage_before]
            op_after = ops_map[stage_after]
            model += start_vars[op_after['id']] - (start_vars[op_before['id']] + op_before['duration']) <= max_wait, \
                     f"wait_{op_before['id']}_{op_after['id']}"

# e) Makespan definition
for batch_data in batch_details:
    if batch_data['ops']:
        last_op_in_batch = sorted(batch_data['ops'], key=lambda x: x['stage_idx_in_batch'])[-1]
        model += makespan_var >= start_vars[last_op_in_batch['id']] + last_op_in_batch['duration'], \
                 f"makespan_def_{last_op_in_batch['id']}"
if not any(batch_data['ops'] for batch_data in batch_details): # Если нет задач
    model += makespan_var == 0

# 6. Define Objective
model += makespan_var, "Minimize_Makespan"

# 7. Solve Model
print(f"\nЗапуск MILP решателя (ограничение по времени: {SOLVER_TIME_LIMIT_SECONDS} сек)...")
# model.writeLP("bakery_milp_model.lp") # Можно записать модель в файл для отладки
solver = PULP_CBC_CMD(msg=True, timeLimit=SOLVER_TIME_LIMIT_SECONDS)
model.solve(solver)
print("MILP Решатель завершил работу.")

status_code = model.status
# Получаем текстовое представление статуса из словаря LpStatus, если ключ существует
status_text = LpStatus.get(status_code, f"Unknown status code: {status_code}")

print(f"\nСтатус решения: {status_text} (PuLP код: {status_code})")

# Инициализируем переменные для результатов
optimal_makespan_minutes = None
schedule_data_for_output = [] # Важно инициализировать здесь
makespan_formatted = "N/A"

# Проверяем, было ли найдено хотя бы допустимое решение
# Оптимальное решение
if status_code == LpStatus['Optimal']: # LpStatus['Optimal'] == 1
    print("--- Оптимальное Решение Найдено ---")
    optimal_makespan_minutes = makespan_var.value()
# Допустимое решение (например, если решатель остановлен по времени, но решение есть)
elif status_code == LpStatus['Feasible'] and makespan_var.value() is not None: # LpStatus['Feasible'] == 0
    print("--- Допустимое Решение Найдено (возможно, не оптимальное) ---")
    optimal_makespan_minutes = makespan_var.value()
# Если решатель CBC был остановлен по времени и НЕ нашел решения,
# PuLP может вернуть status_code = 0 (LpStatus['Feasible']) или -1 (LpStatus['Not Solved']),
# но makespan_var.value() будет None.
elif makespan_var.value() is None : # Явная проверка, что значения решения нет
    if status_code == LpStatus['Infeasible']: # LpStatus['Infeasible'] == -2
        print("\n--- Задача Неразрешима (Infeasible) ---")
        print("Возможные причины: противоречивые ограничения, слишком короткий горизонт (BIG_M), ошибки в логике.")
    elif status_code == LpStatus['Unbounded']: # LpStatus['Unbounded'] == -3
        print("\n--- Задача Неограничена (Unbounded) ---")
        print("Целевая функция может стремиться к минус бесконечности. Проверьте определения makespan и ограничения.")
    else: # LpStatus['Not Solved'] == -1 или другой код, но решения нет
        print(f"\n--- Решение не найдено: {status_text} ---")
        print("Допустимое решение не было найдено решателем в отведенное время или по другой причине.")
        print("Это может быть связано с нехваткой времени, памяти, чрезмерной сложностью модели или другими проблемами решателя.")
        print("Рекомендуется: увеличить лимит времени, упростить модель, или использовать более мощный решатель/подход (например, CP-SAT).")
else: # Неожиданный случай: есть makespan.value(), но статус не Optimal/Feasible
      # Маловероятно, но для полноты
    print(f"\n--- Неожиданный статус с найденным решением: {status_text} ---")
    optimal_makespan_minutes = makespan_var.value()
    print(f"Решатель вернул статус {status_text}, но есть значение makespan: {optimal_makespan_minutes:.2f}. Проверьте детали.")


if optimal_makespan_minutes is not None:
    print(f"Минимальное/Найденное Время Производства (Makespan): {optimal_makespan_minutes:.2f} минут")
    total_seconds_makespan = int(optimal_makespan_minutes * 60)
    tdelta = datetime.timedelta(seconds=total_seconds_makespan)
    days, hours, remainder = tdelta.days, tdelta.seconds // 3600, tdelta.seconds % 3600
    minutes, seconds = remainder // 60, remainder % 60
    makespan_formatted = f"{days} дн " if days > 0 else ""
    makespan_formatted += f"{hours:02}:{minutes:02}:{seconds:02}"
    print(f"Что составляет примерно: {makespan_formatted}")

    stage_order_map = {name: i for i, name in enumerate(STAGES)}
    for op in all_task_operations:
        op_id = op['id']
        start_val = start_vars[op_id].value()
        if start_val is None: start_val = -1

        duration_val = op['duration']
        end_val = start_val + duration_val if start_val != -1 else -1

        assigned_machine = -1
        if op_id in assign_vars and assign_vars[op_id]: # Проверка, что assign_vars[op_id] существует и не пуст
            for m_idx, var in assign_vars[op_id].items():
                if var.value() is not None and var.value() > 0.5:
                    assigned_machine = m_idx
                    break
        
        schedule_data_for_output.append({
            "Batch_ID": op['batch_id_str'], "Stage": op['stage_name'],
            "Task_ID": op_id,
            "Start_Time_Min": start_val, "End_Time_Min": end_val,
            "Duration_Min": duration_val,
            "Assigned_Machine_Idx": assigned_machine,
            "Stage_Order": stage_order_map.get(op['stage_name'], 999)
        })
    schedule_data_for_output.sort(key=lambda x: (x['Start_Time_Min'], x['Batch_ID'], x['Stage_Order']))

# Запись в файлы, только если есть данные
if schedule_data_for_output:
    csv_fieldnames = ["Batch_ID", "Stage", "Task_ID", "Start_Time_Min", "End_Time_Min", "Duration_Min", "Assigned_Machine_Idx"]
    try:
        with open(OUTPUT_CSV_FILE, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_fieldnames)
            writer.writeheader()
            for row_data in schedule_data_for_output:
                row_to_write = {key: row_data[key] for key in csv_fieldnames} # Избегаем лишних ключей
                writer.writerow(row_to_write)
        print(f"\nРасписание успешно записано в CSV файл: '{OUTPUT_CSV_FILE}'")
    except Exception as e:
        print(f"\nОшибка записи CSV файла '{OUTPUT_CSV_FILE}': {e}")
else:
    print("\nНет данных для записи в CSV, так как решение не было найдено или не является допустимым.")

# Запись в TXT файл
try:
    with open(OUTPUT_TXT_FILE, 'w', encoding='utf-8') as txtfile:
        txtfile.write(f"--- Сводка по Производственному Расписанию (MILP - PuLP) ---\n\n")
        txtfile.write(f"Статус решения: {status_text} (PuLP код: {status_code})\n")
        if optimal_makespan_minutes is not None:
             txtfile.write(f"Общее время производства (Makespan): {optimal_makespan_minutes:.2f} минут\n")
             txtfile.write(f"Общее время производства (формат): {makespan_formatted}\n")
        else:
            txtfile.write(f"Общее время производства (Makespan): Не найдено\n")

        txtfile.write(f"Всего задач (операций): {len(all_task_operations)}\n")
        txtfile.write(f"Всего партий (уникальных batch_id_str): {len(batch_details)}\n")
        txtfile.write(f"Время решения ограничено: {SOLVER_TIME_LIMIT_SECONDS} сек.\n")
        if makespan_var.value() is None and status_code != LpStatus['Infeasible'] and status_code != LpStatus['Unbounded']:
            txtfile.write("ПРЕДУПРЕЖДЕНИЕ: Допустимое решение не было найдено в отведенное время.\n")
        # ... (остальная информация для TXT файла)
        critical_constraints_defined_for_summary = [
            (CRITICAL_STAGE_BEFORE_0, CRITICAL_STAGE_AFTER_0, MAX_WAIT_COMBINING_MIXING_MIN),
            (CRITICAL_STAGE_BEFORE_1, CRITICAL_STAGE_AFTER_1, MAX_WAIT_MIXING_FORMING_MIN),
            (CRITICAL_STAGE_BEFORE_2, CRITICAL_STAGE_AFTER_2, MAX_WAIT_FORMING_PROOFING_MIN),
            (CRITICAL_STAGE_BEFORE_3, CRITICAL_STAGE_AFTER_3, MAX_WAIT_PROOFING_BAKING_MIN),
        ]
        txtfile.write(f"\nПараметры модели (макс. время ожидания):\n")
        for sb, sa, mw in critical_constraints_defined_for_summary:
             if sb in STAGES and sa in STAGES:
                txtfile.write(f"  - {sb} -> {sa}: {mw} мин\n")
        txtfile.write(f"\nРазмер партии (BATCH_SIZE): {BATCH_SIZE}\n")
        txtfile.write(f"Этапы с пропорциональным временем для неполных партий: {', '.join(proportional_time_stages)}\n")
        txtfile.write(f"\nДоступные ресурсы (машины):\n")
        for stage_name_res, count in machines_available.items():
            txtfile.write(f"  - {stage_name_res}: {count}\n")
        if schedule_data_for_output:
            txtfile.write(f"\nФайл с детальным расписанием: {OUTPUT_CSV_FILE}\n")
        else:
            txtfile.write(f"\nДетальное расписание не было сгенерировано.\n")
    print(f"Сводная информация успешно записана в TXT файл: '{OUTPUT_TXT_FILE}'")
except Exception as e:
    print(f"\nОшибка записи TXT файла '{OUTPUT_TXT_FILE}': {e}")