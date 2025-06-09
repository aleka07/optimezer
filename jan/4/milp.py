import math
import csv
import datetime
from collections import defaultdict
import pulp
import os

# 1. Исходные данные
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
    "Немецкий хлеб":        {"Комбинирование": "0:21:00", "Смешивание": "0:10:30", "Формовка": "0:11:00", "Расстойка": "0:25:00", "Выпекание": "0:25:00", "Остывание": "1:15:00"},
    "Багет луковый":        {"Комбинирование": "0:21:00", "Смешивание": "0:10:30", "Формовка": "0:11:00", "Расстойка": "0:20:00", "Выпекание": "0:18:00", "Остывание": "0:45:00"},
}

orders = {
    "Формовой": 1910,                   # Формовой хлеб 600гр (1676) + упаковка (234)
    "Мини формовой": 306,                # Формовой мини хлеб 300гр (200) + упаковка (106)
    "Бородинский": 488,                  # Бородинский хлеб 300гр (292) + упаковка (196)
    "Домашний": 17,                      # Домашний хлеб 600гр в упаковке
    "Багет луковый": 33,                 # Багет Луковый 300гр в упаковке
    "Багет новый": 219,                  # Багет Новый 300гр в упаковке
    "Багет отрубной": 49,   # Багет Отрубной 300гр в упаковке
    "Премиум": 20,                       # Багет Премиум 350гр в упаковке
    "Батон Верный": 54,                  # Батон Верный 400гр в упаковке
    "Батон Нарезной": 336,               # Батон Нарезной 400гр (253) + упаковка (83)
    "Береке": 109,                       # Береке хлеб 420гр (53) + упаковка (56)
    "Жайлы": 131,                        # Жайлы хлеб 600гр (85) + упаковка (46)
    "Диета": 210,                        # Диетический хлеб (136 без упаковки + 74 в упаковке)
    "Здоровье": 30,                      # Здоровье хлеб (5 без упаковки + 25 в упаковке)
    "Любимый": 459,                      # Любимый хлеб 500гр (391) + упаковка (68)
    "Немецкий хлеб": 15,                 # Немецкий хлеб 250гр в упаковке
    "Отрубной (общий)": 161,             # Отрубной хлеб (97 без упаковки + 64 в упаковке)
    "Плетенка": 94,                      # Плетенка (все виды: 41+19+11+23)
    "Семейный": 212,                     # Семейный хлеб 600гр (128) + упаковка (84)
    "Славянский": 6,                     # Славянский хлеб 600гр в упаковке
    "Зерновой Столичный": 16,            # Столичный хлеб 450гр в упаковке
    "Сэндвич": 1866,                     # Все виды сэндвичей (70+36+30+41+1674+15)
    "Хлеб «Тартин бездрожжевой»": 18,    # Тартин
    "Хлеб «Зерновой»": 113,              # Зерновой хлеб (90 без упаковки + 23 в упаковке)
    "Чиабатта": 18,                      # Чиабатта
    "Булочка для гамбургера большой с кунжутом": 160  # Булочка для гамбургера
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

# Параметры
BATCH_SIZE = 100
STAGES = ["Комбинирование", "Смешивание", "Формовка", "Расстойка", "Выпекание", "Остывание"] 
        #   "Упаковка"]

# Ограничения времени ожидания (в минутах)
MAX_WAIT_COMBINING_MIXING = 1
MAX_WAIT_MIXING_FORMING = 1
MAX_WAIT_FORMING_PROOFING = 5
MAX_WAIT_PROOFING_BAKING = 5


# --- Имя выходного файла ---
script_dir = os.path.dirname(os.path.abspath(__file__))
OUTPUT_CSV_FILE = os.path.join(script_dir, 'milp_production_schedule.csv')
OUTPUT_TXT_FILE = os.path.join(script_dir, 'milp_production_schedule.txt')

# 2. Вспомогательные функции
def time_str_to_minutes(time_str):
    """Конвертирует строку времени в минуты"""
    try:
        parts = list(map(int, time_str.split(':')))
        if len(parts) == 3:
            h, m, s = parts
            return round(h * 60 + m + s / 60.0)
        elif len(parts) == 2:
            m, s = parts
            return round(m + s / 60.0)
        else:
            return 0
    except:
        return 0

# 3. Предварительная обработка данных
print("Обработка исходных данных...")

# Конвертация времени в минуты
tech_map_minutes = {}
for product, stages_data in tech_map_data.items():
    tech_map_minutes[product] = {}
    for stage_name in STAGES:
        time_str = stages_data.get(stage_name, "0:00:00")
        duration = time_str_to_minutes(time_str)
        tech_map_minutes[product][stage_name] = duration

# Создание партий
all_batches = []
batch_counter = 0
for product, quantity_ordered in orders.items():
    if quantity_ordered <= 0:
        continue
    num_batches = math.ceil(quantity_ordered / BATCH_SIZE)
    for i in range(num_batches):
        batch_id = f"{product}_batch_{i+1}"
        batch_tasks = []
        for stage_index, stage_name in enumerate(STAGES):
            duration = tech_map_minutes[product][stage_name]
            if duration > 0:
                batch_tasks.append({
                    "batch_id": batch_id,
                    "stage_index": stage_index,
                    "stage_name": stage_name,
                    "duration": duration,
                    "product": product
                })
        if batch_tasks:
            all_batches.append({
                "id": batch_id,
                "product": product,
                "tasks": batch_tasks,
                "batch_num": batch_counter
            })
            batch_counter += 1

if not all_batches:
    print("Нет партий для производства.")
    exit()

print(f"Всего партий сгенерировано: {len(all_batches)}")
num_tasks_total = sum(len(b['tasks']) for b in all_batches)
print(f"Всего задач (операций): {num_tasks_total}")

# Расчет горизонта планирования
total_duration = sum(task['duration'] for batch in all_batches for task in batch['tasks'])
horizon = total_duration + 1000  # Добавляем буфер
print(f"Горизонт планирования: {horizon} минут")

# 4. Создание MILP модели
print("\nСоздание MILP модели...")
model = pulp.LpProblem("Production_Scheduling", pulp.LpMinimize)

# 5. Переменные решения
# Время начала каждой задачи
start_times = {}
for batch in all_batches:
    for task in batch['tasks']:
        var_name = f"start_{batch['id']}_{task['stage_name']}"
        start_times[(batch['batch_num'], task['stage_index'])] = pulp.LpVariable(
            var_name, lowBound=0, upBound=horizon, cat='Integer'
        )

# Переменная для makespan
makespan = pulp.LpVariable("makespan", lowBound=0, upBound=horizon, cat='Integer')

# Бинарные переменные для упорядочивания задач на одном ресурсе
precedence_vars = {}
for stage_name in STAGES:
    # Найти все задачи для данного этапа
    stage_tasks = []
    for batch in all_batches:
        for task in batch['tasks']:
            if task['stage_name'] == stage_name:
                stage_tasks.append((batch['batch_num'], task['stage_index'], task['duration']))
    
    # Создать переменные упорядочивания для каждой пары задач
    for i in range(len(stage_tasks)):
        for j in range(i+1, len(stage_tasks)):
            batch_i, stage_i, duration_i = stage_tasks[i]
            batch_j, stage_j, duration_j = stage_tasks[j]
            
            var_name = f"prec_{batch_i}_{stage_i}_{batch_j}_{stage_j}"
            precedence_vars[(batch_i, stage_i, batch_j, stage_j)] = pulp.LpVariable(
                var_name, cat='Binary'
            )

print(f"Создано переменных времени начала: {len(start_times)}")
print(f"Создано переменных упорядочивания: {len(precedence_vars)}")

# 6. Ограничения
print("Добавление ограничений...")

# 6.1 Ограничения последовательности операций внутри партии
sequence_constraints = 0
for batch in all_batches:
    sorted_tasks = sorted(batch['tasks'], key=lambda t: t['stage_index'])
    for k in range(len(sorted_tasks) - 1):
        curr_task = sorted_tasks[k]
        next_task = sorted_tasks[k+1]
        
        curr_start = start_times[(batch['batch_num'], curr_task['stage_index'])]
        next_start = start_times[(batch['batch_num'], next_task['stage_index'])]
        
        # Следующая операция должна начаться не раньше, чем закончится текущая
        model += next_start >= curr_start + curr_task['duration']
        sequence_constraints += 1

print(f"Добавлено ограничений последовательности: {sequence_constraints}")

# 6.2 Ограничения ресурсов (машин)
resource_constraints = 0
big_M = horizon  # Большое число для моделирования дизъюнкций

for stage_name in STAGES:
    machine_count = machines_available.get(stage_name, 0)
    if machine_count <= 0:
        continue
    
    # Найти все задачи для данного этапа
    stage_tasks = []
    for batch in all_batches:
        for task in batch['tasks']:
            if task['stage_name'] == stage_name:
                stage_tasks.append((batch['batch_num'], task['stage_index'], task['duration']))
    
    if len(stage_tasks) <= machine_count:
        continue  # Достаточно машин для всех задач
    
    # Для каждой пары задач добавить ограничения неперекрывания
    for i in range(len(stage_tasks)):
        for j in range(i+1, len(stage_tasks)):
            batch_i, stage_i, duration_i = stage_tasks[i]
            batch_j, stage_j, duration_j = stage_tasks[j]
            
            start_i = start_times[(batch_i, stage_i)]
            start_j = start_times[(batch_j, stage_j)]
            prec_var = precedence_vars[(batch_i, stage_i, batch_j, stage_j)]
            
            # Если prec_var = 1, то задача i выполняется раньше задачи j
            model += start_j >= start_i + duration_i - big_M * (1 - prec_var)
            # Если prec_var = 0, то задача j выполняется раньше задачи i
            model += start_i >= start_j + duration_j - big_M * prec_var
            
            resource_constraints += 2

print(f"Добавлено ограничений ресурсов: {resource_constraints}")

# 6.3 Ограничения времени ожидания
waiting_constraints = 0
for batch in all_batches:
    batch_num = batch['batch_num']
    
    # Словарь для быстрого поиска задач по этапам
    task_by_stage = {}
    for task in batch['tasks']:
        task_by_stage[task['stage_name']] = task
    
    # Комбинирование -> Смешивание
    if "Комбинирование" in task_by_stage and "Смешивание" in task_by_stage:
        comb_task = task_by_stage["Комбинирование"]
        mix_task = task_by_stage["Смешивание"]
        
        comb_start = start_times[(batch_num, comb_task['stage_index'])]
        mix_start = start_times[(batch_num, mix_task['stage_index'])]
        
        # Время ожидания = Начало смешивания - (Начало комбинирования + Длительность комбинирования)
        model += mix_start - (comb_start + comb_task['duration']) <= MAX_WAIT_COMBINING_MIXING
        waiting_constraints += 1
    
    # Смешивание -> Формовка
    if "Смешивание" in task_by_stage and "Формовка" in task_by_stage:
        mix_task = task_by_stage["Смешивание"]
        form_task = task_by_stage["Формовка"]
        
        mix_start = start_times[(batch_num, mix_task['stage_index'])]
        form_start = start_times[(batch_num, form_task['stage_index'])]
        
        model += form_start - (mix_start + mix_task['duration']) <= MAX_WAIT_MIXING_FORMING
        waiting_constraints += 1
    
    # Формовка -> Расстойка
    if "Формовка" in task_by_stage and "Расстойка" in task_by_stage:
        form_task = task_by_stage["Формовка"]
        proof_task = task_by_stage["Расстойка"]
        
        form_start = start_times[(batch_num, form_task['stage_index'])]
        proof_start = start_times[(batch_num, proof_task['stage_index'])]
        
        model += proof_start - (form_start + form_task['duration']) <= MAX_WAIT_FORMING_PROOFING
        waiting_constraints += 1
    
    # Расстойка -> Выпекание
    if "Расстойка" in task_by_stage and "Выпекание" in task_by_stage:
        proof_task = task_by_stage["Расстойка"]
        bake_task = task_by_stage["Выпекание"]
        
        proof_start = start_times[(batch_num, proof_task['stage_index'])]
        bake_start = start_times[(batch_num, bake_task['stage_index'])]
        
        model += bake_start - (proof_start + proof_task['duration']) <= MAX_WAIT_PROOFING_BAKING
        waiting_constraints += 1

print(f"Добавлено ограничений времени ожидания: {waiting_constraints}")

# 6.4 Ограничения makespan
makespan_constraints = 0
for batch in all_batches:
    if batch['tasks']:
        # Найти последний этап в партии
        last_task = max(batch['tasks'], key=lambda t: t['stage_index'])
        last_start = start_times[(batch['batch_num'], last_task['stage_index'])]
        
        # Makespan должен быть не меньше времени окончания последней задачи
        model += makespan >= last_start + last_task['duration']
        makespan_constraints += 1

print(f"Добавлено ограничений makespan: {makespan_constraints}")

# 7. Целевая функция
model += makespan

print(f"\nВсего ограничений в модели: {len(model.constraints)}")
print(f"Всего переменных в модели: {len(model.variables())}")

# 8. Решение модели
print("\nЗапуск решателя MILP...")
model.solve(pulp.PULP_CBC_CMD(msg=1, timeLimit=300))  # 5 минут лимит

# 9. Обработка результатов
status = pulp.LpStatus[model.status]
print(f"\nСтатус решения: {status}")

if model.status == pulp.LpStatusOptimal or model.status == pulp.LpStatusFeasible:
    optimal_makespan = pulp.value(makespan)
    print(f"Оптимальное время производства: {optimal_makespan:.2f} минут")
    
    # Форматирование времени
    total_seconds = int(optimal_makespan * 60)
    time_delta = datetime.timedelta(seconds=total_seconds)
    print(f"Время производства: {time_delta}")
    
    # Сбор расписания
    schedule_data = []
    stage_order = {name: i for i, name in enumerate(STAGES)}
    
    for batch in all_batches:
        for task in batch['tasks']:
            start_time = pulp.value(start_times[(batch['batch_num'], task['stage_index'])])
            end_time = start_time + task['duration']
            
            schedule_data.append({
                "Batch_ID": batch['id'],
                "Product": batch['product'],
                "Stage": task['stage_name'],
                "Start_Time_Min": start_time,
                "End_Time_Min": end_time,
                "Duration_Min": task['duration'],
                "Stage_Order": stage_order.get(task['stage_name'], 999)
            })
    
    # Сортировка по времени начала
    schedule_data.sort(key=lambda x: (x['Start_Time_Min'], x['Batch_ID'], x['Stage_Order']))
    
    # 10. Запись результатов в файлы
    # CSV файл
    try:
        csv_fieldnames = ["Batch_ID", "Product", "Stage", "Start_Time_Min", "End_Time_Min", "Duration_Min"]
        with open(OUTPUT_CSV_FILE, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_fieldnames)
            writer.writeheader()
            for row_data in schedule_data:
                row_to_write = {key: row_data[key] for key in csv_fieldnames}
                writer.writerow(row_to_write)
        print(f"\nРасписание записано в CSV файл: '{OUTPUT_CSV_FILE}'")
    except Exception as e:
        print(f"Ошибка записи CSV файла: {e}")
    
    # TXT файл с сводкой
    try:
        with open(OUTPUT_TXT_FILE, 'w', encoding='utf-8') as txtfile:
            txtfile.write("--- Сводка по Производственному Расписанию (MILP) ---\n\n")
            txtfile.write(f"Статус решения: {status}\n")
            txtfile.write(f"Общее время производства: {optimal_makespan:.2f} минут\n")
            txtfile.write(f"Время производства (формат): {time_delta}\n")
            txtfile.write(f"Всего партий: {len(all_batches)}\n")
            txtfile.write(f"Всего задач: {len(schedule_data)}\n")
            txtfile.write(f"\nПараметры ограничений времени ожидания:\n")
            txtfile.write(f"  - Комбинирование -> Смешивание: {MAX_WAIT_COMBINING_MIXING} мин\n")
            txtfile.write(f"  - Смешивание -> Формовка: {MAX_WAIT_MIXING_FORMING} мин\n")
            txtfile.write(f"  - Формовка -> Расстойка: {MAX_WAIT_FORMING_PROOFING} мин\n")
            txtfile.write(f"  - Расстойка -> Выпекание: {MAX_WAIT_PROOFING_BAKING} мин\n")
            txtfile.write(f"\nДоступные ресурсы:\n")
            for stage, count in machines_available.items():
                txtfile.write(f"  - {stage}: {count} машин(ы)\n")
            txtfile.write(f"\nДетальное расписание: {OUTPUT_CSV_FILE}\n")
        print(f"Сводка записана в TXT файл: '{OUTPUT_TXT_FILE}'")
    except Exception as e:
        print(f"Ошибка записи TXT файла: {e}")
    
    # Показать первые несколько записей расписания
    print(f"\nПервые 10 записей расписания:")
    print(f"{'Партия':<30} {'Этап':<15} {'Начало':<8} {'Конец':<8} {'Длит.':<6}")
    print("-" * 70)
    for i, row in enumerate(schedule_data[:10]):
        print(f"{row['Batch_ID']:<30} {row['Stage']:<15} {row['Start_Time_Min']:<8.0f} {row['End_Time_Min']:<8.0f} {row['Duration_Min']:<6.0f}")
    
    if len(schedule_data) > 10:
        print(f"... и еще {len(schedule_data) - 10} записей")

else:
    print("Не удалось найти оптимальное решение")
    if model.status == pulp.LpStatusInfeasible:
        print("Задача неразрешима - возможно, слишком жесткие ограничения времени ожидания")
    elif model.status == pulp.LpStatusUnbounded:
        print("Задача неограничена")
    else:
        print("Неизвестная ошибка решателя")

print("\nВыполнение завершено!")