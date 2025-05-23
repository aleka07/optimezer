import math
import collections
import csv
import datetime
from ortools.sat.python import cp_model



# 1. Define Input Data
tech_map_data = {
    # Этапы 1-3 (Подготовка смеси, 18-24 мин) -> Комбинирование: "0:21:00"
    # Этап 4 (Замес): Формовые "0:12:00", Остальные (10-11 мин) "0:10:30"
    # Этап 5 (Деление и формовка, 10-12 мин) -> Формовка: "0:11:00"
    # Этап 6 (Расстойка): Н/Д для "Остальных видов" -> "0:25:00" (среднее)
    # Выпекание: согласно таблице (16-17 -> "0:16:30", 18-19 -> "0:18:30")
    # Остывание/Упаковка: типовые значения

    "Мини формовой":        {"Комбинирование": "0:21:00", "Смешивание": "0:12:00", "Формовка": "0:11:00", "Расстойка": "0:25:00", "Выпекание": "0:45:00", "Остывание": "1:30:00", "Упаковка": "0:10:00"},
    "Формовой":             {"Комбинирование": "0:21:00", "Смешивание": "0:12:00", "Формовка": "0:11:00", "Расстойка": "0:25:00", "Выпекание": "0:45:00", "Остывание": "1:30:00", "Упаковка": "0:10:00"},
    "Домашний":             {"Комбинирование": "0:21:00", "Смешивание": "0:10:30", "Формовка": "0:11:00", "Расстойка": "0:25:00", "Выпекание": "0:17:00", "Остывание": "1:00:00", "Упаковка": "0:10:00"},
    "Семейный":             {"Комбинирование": "0:21:00", "Смешивание": "0:10:30", "Формовка": "0:11:00", "Расстойка": "0:25:00", "Выпекание": "0:17:00", "Остывание": "1:00:00", "Упаковка": "0:10:00"},
    "Славянский":           {"Комбинирование": "0:21:00", "Смешивание": "0:10:30", "Формовка": "0:11:00", "Расстойка": "0:25:00", "Выпекание": "0:17:00", "Остывание": "1:00:00", "Упаковка": "0:10:00"},
    "Жайлы":                {"Комбинирование": "0:21:00", "Смешивание": "0:10:30", "Формовка": "0:11:00", "Расстойка": "0:25:00", "Выпекание": "0:17:00", "Остывание": "1:00:00", "Упаковка": "0:10:00"},
    "Отрубной (общий)":     {"Комбинирование": "0:21:00", "Смешивание": "0:10:30", "Формовка": "0:11:00", "Расстойка": "0:25:00", "Выпекание": "0:17:00", "Остывание": "1:00:00", "Упаковка": "0:10:00"},
    "Любимый":              {"Комбинирование": "0:21:00", "Смешивание": "0:10:30", "Формовка": "0:11:00", "Расстойка": "0:25:00", "Выпекание": "0:17:00", "Остывание": "1:00:00", "Упаковка": "0:10:00"},
    "Датский":              {"Комбинирование": "0:21:00", "Смешивание": "0:10:30", "Формовка": "0:11:00", "Расстойка": "0:25:00", "Выпекание": "0:17:00", "Остывание": "1:00:00", "Упаковка": "0:10:00"},
    "Тартин (из таблицы)":  {"Комбинирование": "0:21:00", "Смешивание": "0:10:30", "Формовка": "0:11:00", "Расстойка": "0:25:00", "Выпекание": "0:17:00", "Остывание": "1:00:00", "Упаковка": "0:10:00"},
    "Зерновой Столичный":   {"Комбинирование": "0:21:00", "Смешивание": "0:10:30", "Формовка": "0:11:00", "Расстойка": "0:25:00", "Выпекание": "0:17:00", "Остывание": "1:00:00", "Упаковка": "0:10:00"},
    "Здоровье":             {"Комбинирование": "0:21:00", "Смешивание": "0:10:30", "Формовка": "0:11:00", "Расстойка": "0:25:00", "Выпекание": "0:18:00", "Остывание": "1:00:00", "Упаковка": "0:10:00"},
    "Бородинский":          {"Комбинирование": "0:21:00", "Смешивание": "0:10:30", "Формовка": "0:11:00", "Расстойка": "0:20:00", "Выпекание": "0:55:00", "Остывание": "2:00:00", "Упаковка": "0:15:00"},
    "Булочка для хотдога/бургера (из таблицы)": {"Комбинирование": "0:21:00", "Смешивание": "0:10:30", "Формовка": "0:11:00", "Расстойка": "0:25:00", "Выпекание": "0:22:00", "Остывание": "0:45:00", "Упаковка": "0:15:00"},
    "Береке":               {"Комбинирование": "0:21:00", "Смешивание": "0:10:30", "Формовка": "0:11:00", "Расстойка": "0:20:00", "Выпекание": "0:16:30", "Остывание": "1:00:00", "Упаковка": "0:10:00"},
    "Баварский Деревенский Ржаной": {"Комбинирование": "0:21:00", "Смешивание": "0:10:30", "Формовка": "0:11:00", "Расстойка": "0:30:00", "Выпекание": "0:18:30", "Остывание": "1:30:00", "Упаковка": "0:10:00"},
    "Плетенка":             {"Комбинирование": "0:21:00", "Смешивание": "0:10:30", "Формовка": "0:11:00", "Расстойка": "0:25:00", "Выпекание": "0:17:00", "Остывание": "1:00:00", "Упаковка": "0:10:00"},
    "Батон Верный":         {"Комбинирование": "0:21:00", "Смешивание": "0:10:30", "Формовка": "0:11:00", "Расстойка": "0:25:00", "Выпекание": "0:17:00", "Остывание": "1:00:00", "Упаковка": "0:10:00"},
    "Батон Нарезной":       {"Комбинирование": "0:21:00", "Смешивание": "0:10:30", "Формовка": "0:11:00", "Расстойка": "0:25:00", "Выпекание": "0:17:00", "Остывание": "1:00:00", "Упаковка": "0:10:00"},
    "Диета":                {"Комбинирование": "0:21:00", "Смешивание": "0:10:30", "Формовка": "0:11:00", "Расстойка": "0:37:30", "Выпекание": "0:35:00", "Остывание": "1:30:00", "Упаковка": "0:10:00"},
    "Багет отрубной (из таблицы)": {"Комбинирование": "0:21:00", "Смешивание": "0:10:30", "Формовка": "0:11:00", "Расстойка": "0:25:00", "Выпекание": "0:18:00", "Остывание": "0:45:00", "Упаковка": "0:10:00"},
    "Премиум":              {"Комбинирование": "0:21:00", "Смешивание": "0:10:30", "Формовка": "0:11:00", "Расстойка": "0:25:00", "Выпекание": "0:25:00", "Остывание": "1:15:00", "Упаковка": "0:10:00"},
    "Багет новый":          {"Комбинирование": "0:21:00", "Смешивание": "0:10:30", "Формовка": "0:11:00", "Расстойка": "0:25:00", "Выпекание": "0:25:00", "Остывание": "1:15:00", "Упаковка": "0:10:00"},
    "Багет луковый":        {"Комбинирование": "0:21:00", "Смешивание": "0:10:30", "Формовка": "0:11:00", "Расстойка": "0:20:00", "Выпекание": "0:18:00", "Остывание": "0:45:00", "Упаковка": "0:10:00"},

    # Оригинальные данные из вашего первого скрипта для сравнения и использования
    "Чиабатта":                      {"Комбинирование": "0:03:39", "Смешивание": "0:15:00", "Формовка": "0:20:00", "Расстойка": "0:30:00", "Выпекание": "0:25:00", "Остывание": "1:00:00", "Упаковка": "0:15:00"},
    "Булочка (ориг.)":               {"Комбинирование": "0:03:39", "Смешивание": "0:11:00", "Формовка": "0:21:30", "Расстойка": "0:45:00", "Выпекание": "0:22:00", "Остывание": "1:15:00", "Упаковка": "0:15:00"},
    "Булочка для хот-дог/гамбургера (ориг.)": {"Комбинирование": "0:03:39", "Смешивание": "0:12:00", "Формовка": "0:25:00", "Расстойка": "0:00:00", "Выпекание": "0:40:00", "Остывание": "2:45:00", "Упаковка": "0:15:00"}, # Здесь расстойка была 0
    "Бриошь":                        {"Комбинирование": "0:03:39", "Смешивание": "0:07:00", "Формовка": "0:25:00", "Расстойка": "0:20:00", "Выпекание": "0:03:00", "Остывание": "0:20:00", "Упаковка": "0:15:00"},
    "Булочка для гамбургера большой/ с кунжутом": {"Комбинирование": "0:03:39", "Смешивание": "0:10:00", "Формовка": "0:30:00", "Расстойка": "1:00:00", "Выпекание": "0:35:00", "Остывание": "2:30:00", "Упаковка": "0:15:00"},
    "Немецкий хлеб":                 {"Комбинирование": "0:03:39", "Смешивание": "0:12:00", "Формовка": "0:30:00", "Расстойка": "1:10:00", "Выпекание": "0:40:00", "Остывание": "2:40:00", "Упаковка": "0:15:00"},
    "Хлеб «Зерновой»":               {"Комбинирование": "0:03:39", "Смешивание": "0:18:00", "Формовка": "0:35:00", "Расстойка": "1:00:00", "Выпекание": "0:18:00", "Остывание": "1:00:00", "Упаковка": "0:15:00"},
    "Хот-дог/Гамбургер солодовый с семечками": {"Комбинирование": "0:03:39", "Смешивание": "0:09:00", "Формовка": "0:40:00", "Расстойка": "0:40:00", "Выпекание": "0:20:00", "Остывание": "0:45:00", "Упаковка": "0:15:00"},
    "Сэндвич":                       {"Комбинирование": "0:03:39", "Смешивание": "0:08:00", "Формовка": "1:00:00", "Расстойка": "0:50:00", "Выпекание": "0:25:00", "Остывание": "2:00:00", "Упаковка": "0:15:00"},
    "Хлеб «Тартин бездрожжевой»":     {"Комбинирование": "0:03:39", "Смешивание": "0:08:00", "Формовка": "3:00:00", "Расстойка": "0:00:00", "Выпекание": "0:18:00", "Остывание": "0:40:00", "Упаковка": "0:15:00"}, # Здесь расстойка была 0
}

orders = { # Используем имя orders_one_day из вашего примера, но переименовал в orders для совместимости с кодом ниже
    "Формовой":             400,
    "Бородинский":          200,
    "Домашний":             300,
    "Багет луковый":        300,
    "Чиабатта (ориг.)":     200,
    "Хлеб «Тартин бездрожжевой» (ориг.)": 100,
    "Булочка (ориг.)":      300,
}


machines_available = {
    "Комбинирование": 1,
    "Смешивание": 3,
    "Формовка": 2,
    "Расстойка": 8, # Может стать узким местом, т.к. у многих теперь есть расстойка
    "Выпекание": 6, # Ограниченное количество печей для более реалистичного сценария
    "Остывание": 25, # Достаточно места для остывания
    "Упаковка": 10,
}

# orders_one_day ДЛЯ ВСТАВКИ В time_minimizer.py или отдельный скрипт
# (Замените существующий словарь orders)

orders_one_day = {
    # Продукты из вашей таблицы (с расстойкой 25 мин для "Н/Д", если применимо)
    "Формовой":             400,  # 4 партии
    "Бородинский":          200,  # 2 партии
    "Домашний":             300,  # 3 партии (с расстойкой 25 мин)
    "Багет луковый":        300,  # 3 партии
    
    # Продукты из вашего "оригинального" списка для разнообразия
    "Чиабатта (ориг.)":     200,  # 2 партии
    "Хлеб «Тартин бездрожжевой» (ориг.)": 100, # 1 партия (длинная формовка, 0 расстойка)
    "Булочка (ориг.)":      300,  # 3 партии
}
# Итого: 4+2+3+3+2+1+3 = 18 партий.

BATCH_SIZE = 100 # Размер партии
STAGES = [
    "Комбинирование", "Смешивание", "Формовка", "Расстойка",
    "Выпекание", "Остывание", "Упаковка",
]

# --- Параметры ---
MAX_WAIT_COMBINING_MIXING_MIN = 15
MAX_WAIT_MIXING_FORMING_MIN = 20
MAX_WAIT_FORMING_PROOFING_MIN = 30
MAX_WAIT_PROOFING_BAKING_MIN = 60

CRITICAL_STAGE_BEFORE_0 = "Комбинирование"
CRITICAL_STAGE_AFTER_0 = "Смешивание"
CRITICAL_STAGE_BEFORE_1 = "Смешивание"
CRITICAL_STAGE_AFTER_1 = "Формовка"
CRITICAL_STAGE_BEFORE_2 = "Формовка"
CRITICAL_STAGE_AFTER_2 = "Расстойка"
CRITICAL_STAGE_BEFORE_3 = "Расстойка"
CRITICAL_STAGE_AFTER_3 = "Выпекание"

# --- Имя выходного файла ---
OUTPUT_CSV_FILE = 'production_schedule.csv'
OUTPUT_TXT_FILE = 'production_summary.txt'

# 2. Helper Function & Preprocessing
def time_str_to_minutes_int(time_str):
    try:
        parts = list(map(int, time_str.split(':')))
        if len(parts) == 3: h, m, s = parts; return round(h * 60 + m + s / 60.0)
        elif len(parts) == 2: m, s = parts; return round(m + s / 60.0)
        else: return 0
    except: return 0

tech_map_minutes_int = {}
total_duration_estimate = 0
for product, stages_data in tech_map_data.items():
    if product not in orders or orders[product] <= 0: # Оптимизация: не обрабатывать продукты, которых нет в заказе
        continue
    tech_map_minutes_int[product] = {}
    prod_duration = 0
    for stage_name in STAGES:
        time_str = stages_data.get(stage_name, "0:00:00")
        duration = time_str_to_minutes_int(time_str)
        tech_map_minutes_int[product][stage_name] = duration
        prod_duration += duration
    total_duration_estimate += prod_duration * math.ceil(orders.get(product, 0) / BATCH_SIZE) # Учитываем количество партий

all_batches = []
for product, quantity_ordered in orders.items():
    if quantity_ordered <= 0: continue
    if product not in tech_map_data:
        print(f"Предупреждение: Продукт '{product}' из заказа отсутствует в технологической карте. Пропускается.")
        continue
    num_batches = math.ceil(quantity_ordered / BATCH_SIZE)
    for i in range(num_batches):
        batch_id = f"{product}_batch_{i+1}"
        batch_tasks = []
        for stage_index, stage_name in enumerate(STAGES):
            # Используем предварительно рассчитанные минуты
            duration = tech_map_minutes_int.get(product, {}).get(stage_name, 0)
            if duration > 0:
                batch_tasks.append({
                    "batch_id": batch_id, "stage_index": stage_index,
                    "stage_name": stage_name, "duration": duration, })
        if batch_tasks: all_batches.append({"id": batch_id, "product": product, "tasks": batch_tasks})

if not all_batches:
    print("Нет партий для производства. Проверьте заказы и технологическую карту.")
    exit()

print(f"Всего партий сгенерировано: {len(all_batches)}")
num_tasks_total = sum(len(b['tasks']) for b in all_batches)
print(f"Всего задач (операций) с ненулевой длительностью: {num_tasks_total}")

# Уточненный горизонт
max_single_batch_duration = 0
if all_batches:
    max_single_batch_duration = max(sum(task['duration'] for task in batch['tasks']) for batch in all_batches)

# Горизонт: сумма всех длительностей всех задач + запас, или более сложная оценка
# Простая оценка: сумма всех длительностей / мин_количество_машин_на_узком_месте + запас
# Более надежно: Сумма длительностей всех задач (если бы они шли одна за другой) + некий буфер
# total_task_durations = sum(t['duration'] for batch in all_batches for t in batch['tasks'])
# horizon = total_task_durations + max_single_batch_duration * len(all_batches) # Очень грубая верхняя граница
horizon = sum(tech_map_minutes_int[batch['product']][task['stage_name']] for batch in all_batches for task in batch['tasks'])
# Добавим запас, например, среднюю длительность всех этапов на количество партий
avg_stage_duration_total = 0
num_stages_total = 0
for prod_stages in tech_map_minutes_int.values():
    for stage_dur in prod_stages.values():
        if stage_dur > 0:
            avg_stage_duration_total += stage_dur
            num_stages_total +=1
avg_stage_duration = (avg_stage_duration_total / num_stages_total) if num_stages_total > 0 else 30 # fallback
horizon += avg_stage_duration * len(all_batches) * 2 # Запас
horizon = math.ceil(horizon)

print(f"Расчетный горизонт (макс. возможное время): {horizon} минут")


# 3. Model Creation
model = cp_model.CpModel()

# 4. Define Variables
task_vars = collections.defaultdict(dict)
task_lookup = {} # Для быстрого доступа к переменным по (batch_id, stage_name)

for i, batch in enumerate(all_batches):
    batch_id = batch['id']
    for task in batch['tasks']:
        stage_idx = task['stage_index']
        stage_name = task['stage_name']
        duration = task['duration'] # Уже в минутах

        suffix = f'_{batch_id}_{stage_name}' # Уникальный суффикс
        start_var = model.NewIntVar(0, horizon, 'start' + suffix)
        end_var = model.NewIntVar(0, horizon, 'end' + suffix)
        interval_var = model.NewIntervalVar(start_var, duration, end_var, 'interval' + suffix)

        task_vars[i][stage_idx] = (start_var, end_var, interval_var)
        task_lookup[(batch_id, stage_name)] = (start_var, end_var, interval_var)


# 5. Define Constraints
# a) Sequence (задачи внутри одной партии идут последовательно)
for i, batch in enumerate(all_batches):
    # Задачи уже отсортированы по stage_index при создании all_batches, если STAGES корректен
    # Но для надежности можно отсортировать еще раз или убедиться, что tasks отсортированы
    sorted_tasks_for_batch = sorted(batch['tasks'], key=lambda t: t['stage_index'])
    for k in range(len(sorted_tasks_for_batch) - 1):
        current_task_info = sorted_tasks_for_batch[k]
        next_task_info = sorted_tasks_for_batch[k+1]

        curr_idx = current_task_info['stage_index']
        next_idx = next_task_info['stage_index']

        # Убедимся, что обе задачи существуют в task_vars (из-за возможного пропуска нулевых этапов)
        if curr_idx in task_vars[i] and next_idx in task_vars[i]:
            model.Add(task_vars[i][next_idx][0] >= task_vars[i][curr_idx][1])


# b) Resources (ограничение по количеству машин на каждом этапе)
for stage_index, stage_name in enumerate(STAGES):
    machine_count = machines_available.get(stage_name)
    if machine_count is None or machine_count <= 0:
        # Если для этапа не указаны машины, или их 0, задачи этого этапа не могут быть выполнены, если они есть
        # Это должно привести к неразрешимости, если такие задачи существуют
        # Либо можно считать, что ресурс не ограничен (не рекомендуется для основных этапов)
        # print(f"Предупреждение: Для этапа '{stage_name}' не указано количество машин или оно равно 0.")
        continue

    intervals_for_stage = []
    for i, batch in enumerate(all_batches):
         # Проверяем, есть ли у данной партии задача на данном этапе
         if stage_index in task_vars[i]: # task_vars[i] - это словарь {stage_idx: (start, end, interval)}
             intervals_for_stage.append(task_vars[i][stage_index][2]) # Берем interval_var

    if intervals_for_stage:
        # Для каждого интервала нужна его "нагрузка" на ресурс, здесь она равна 1
        demands = [1] * len(intervals_for_stage)
        model.AddCumulative(intervals_for_stage, demands, machine_count)


# c) Spoilage/Waiting Time Constraints
for i, batch in enumerate(all_batches):
    batch_id = batch['id']

    # Ограничение 0: Комбинирование -> Смешивание
    task0_key = (batch_id, CRITICAL_STAGE_BEFORE_0)
    task1_key = (batch_id, CRITICAL_STAGE_AFTER_0)
    if task0_key in task_lookup and task1_key in task_lookup:
        model.Add(task_lookup[task1_key][0] - task_lookup[task0_key][1] <= MAX_WAIT_COMBINING_MIXING_MIN)

    # Ограничение 1: Смешивание -> Формовка
    task1_key_new = (batch_id, CRITICAL_STAGE_BEFORE_1)
    task2_key_new = (batch_id, CRITICAL_STAGE_AFTER_1)
    if task1_key_new in task_lookup and task2_key_new in task_lookup:
        model.Add(task_lookup[task2_key_new][0] - task_lookup[task1_key_new][1] <= MAX_WAIT_MIXING_FORMING_MIN)

    # Ограничение 2: Формовка -> Расстойка
    task2_key_old1 = (batch_id, CRITICAL_STAGE_BEFORE_2)
    task3_key_old1 = (batch_id, CRITICAL_STAGE_AFTER_2)
    if task2_key_old1 in task_lookup and task3_key_old1 in task_lookup:
        # Проверяем, что расстойка вообще есть для этого продукта (длительность > 0)
        # Это уже учтено тем, что task_lookup[key] существует только для задач с duration > 0
        model.Add(task_lookup[task3_key_old1][0] - task_lookup[task2_key_old1][1] <= MAX_WAIT_FORMING_PROOFING_MIN)

    # Ограничение 3: Расстойка -> Выпекание
    task3_key_old2 = (batch_id, CRITICAL_STAGE_BEFORE_3)
    task4_key_old2 = (batch_id, CRITICAL_STAGE_AFTER_3)
    if task3_key_old2 in task_lookup and task4_key_old2 in task_lookup:
        # Аналогично, проверяем, что расстойка и выпекание существуют
        model.Add(task_lookup[task4_key_old2][0] - task_lookup[task3_key_old2][1] <= MAX_WAIT_PROOFING_BAKING_MIN)


# d) Makespan (общее время выполнения всех заказов)
makespan = model.NewIntVar(0, horizon, 'makespan')
last_stage_tasks_ends = []

for i, batch in enumerate(all_batches):
     if batch['tasks']: # Убедимся, что у партии вообще есть задачи
        # Находим индекс последнего реального этапа для данной партии (с ненулевой длительностью)
        actual_last_stage_idx = batch['tasks'][-1]['stage_index']
        if actual_last_stage_idx in task_vars[i]: # Проверяем, что для этого этапа были созданы переменные
            last_stage_tasks_ends.append(task_vars[i][actual_last_stage_idx][1]) # Берем end_var

if last_stage_tasks_ends:
    model.AddMaxEquality(makespan, last_stage_tasks_ends)
else:
    # Если нет задач вообще (маловероятно после проверки all_batches), makespan = 0
    model.Add(makespan == 0)


# 6. Define Objective
model.Minimize(makespan)

# 7. Solve Model
solver = cp_model.CpSolver()
# Установим лимит по времени, если нужно, например, 5 минут
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
    # Форматирование timedelta для вывода дней, часов, минут, секунд
    days = tdelta.days
    hours, remainder = divmod(tdelta.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    makespan_formatted = ""
    if days > 0: makespan_formatted += f"{days} дн "
    makespan_formatted += f"{hours:02}:{minutes:02}:{seconds:02}"
    print(f"Что составляет примерно: {makespan_formatted}")

    schedule_data_for_output = []
    stage_order_map = {name: i for i, name in enumerate(STAGES)} # Для сортировки этапов в CSV

    for i, batch in enumerate(all_batches):
        batch_id = batch['id']
        for task_info in batch['tasks']: # Итерируем по информации о задачах из all_batches
            stage_idx = task_info['stage_index']
            stage_name = task_info['stage_name']

            # Проверяем, были ли созданы переменные для этой задачи (они создаются только для duration > 0)
            if stage_idx in task_vars[i]:
                start_val = solver.Value(task_vars[i][stage_idx][0])
                end_val = solver.Value(task_vars[i][stage_idx][1])
                # Длительность должна совпадать с исходной, но можно пересчитать для проверки
                duration_val = end_val - start_val # task_info['duration']

                schedule_data_for_output.append({
                    "Batch_ID": batch_id,
                    "Stage": stage_name,
                    "Start_Time_Min": start_val,
                    "End_Time_Min": end_val,
                    "Duration_Min": duration_val,
                    "Stage_Order": stage_order_map.get(stage_name, 999) # Для сортировки
                })

    # Сортировка для CSV: сначала по времени начала, потом по ID партии, потом по порядку этапа
    schedule_data_for_output.sort(key=lambda x: (x['Start_Time_Min'], x['Batch_ID'], x['Stage_Order']))

    # Запись в CSV
    csv_fieldnames = ["Batch_ID", "Stage", "Start_Time_Min", "End_Time_Min", "Duration_Min"]
    try:
        with open(OUTPUT_CSV_FILE, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_fieldnames)
            writer.writeheader()
            for row_data in schedule_data_for_output:
                # Убираем 'Stage_Order' перед записью, если он не нужен в CSV
                row_to_write = {key: row_data[key] for key in csv_fieldnames}
                writer.writerow(row_to_write)
        print(f"\nРасписание успешно записано в CSV файл: '{OUTPUT_CSV_FILE}'")
    except Exception as e:
        print(f"\nОшибка записи CSV файла '{OUTPUT_CSV_FILE}': {e}")

    # Запись в TXT
    try:
        with open(OUTPUT_TXT_FILE, 'w', encoding='utf-8') as txtfile:
            txtfile.write("--- Сводка по Производственному Расписанию ---\n\n")
            txtfile.write(f"Статус решения: {'Оптимальное' if status == cp_model.OPTIMAL else 'Допустимое'}\n")
            txtfile.write(f"Общее время производства (Makespan): {optimal_makespan_minutes:.2f} минут\n")
            txtfile.write(f"Общее время производства (формат): {makespan_formatted}\n")
            txtfile.write(f"Всего партий: {len(all_batches)}\n")
            txtfile.write(f"Всего задач (операций) в расписании: {len(schedule_data_for_output)}\n") # Это количество записей в CSV
            txtfile.write(f"\nПараметры модели (макс. время ожидания):\n")
            txtfile.write(f"  - {CRITICAL_STAGE_BEFORE_0} -> {CRITICAL_STAGE_AFTER_0}: {MAX_WAIT_COMBINING_MIXING_MIN} мин\n")
            txtfile.write(f"  - {CRITICAL_STAGE_BEFORE_1} -> {CRITICAL_STAGE_AFTER_1}: {MAX_WAIT_MIXING_FORMING_MIN} мин\n")
            txtfile.write(f"  - {CRITICAL_STAGE_BEFORE_2} -> {CRITICAL_STAGE_AFTER_2}: {MAX_WAIT_FORMING_PROOFING_MIN} мин\n")
            txtfile.write(f"  - {CRITICAL_STAGE_BEFORE_3} -> {CRITICAL_STAGE_AFTER_3}: {MAX_WAIT_PROOFING_BAKING_MIN} мин\n")
            txtfile.write(f"\nРазмер партии (BATCH_SIZE): {BATCH_SIZE}\n")
            txtfile.write(f"\nДоступные ресурсы (машины):\n")
            for stage_name_res, count in machines_available.items():
                txtfile.write(f"  - {stage_name_res}: {count}\n")
            txtfile.write(f"\nФайл с детальным расписанием: {OUTPUT_CSV_FILE}\n")
        print(f"Сводная информация успешно записана в TXT файл: '{OUTPUT_TXT_FILE}'")
    except Exception as e:
        print(f"\nОшибка записи TXT файла '{OUTPUT_TXT_FILE}': {e}")

elif status == cp_model.INFEASIBLE:
    print("\n--- Задача Неразрешима (INFEASIBLE) ---")
    print("Возможные причины:")
    print("  - Слишком жесткие ограничения по времени ожидания (MAX_WAIT_...).")
    print("  - Недостаточное количество машин (machines_available) для выполнения всех задач в срок.")
    print("  - Слишком большой объем заказов (orders) для текущих ресурсов и ограничений.")
    print("  - Ошибки в технологической карте (tech_map_data), например, слишком длинные этапы.")
    print("  - Слишком короткий горизонт планирования (horizon) - маловероятно с текущим расчетом, но возможно.")
    print("Проверьте параметры и входные данные.")
elif status == cp_model.MODEL_INVALID:
    print("\n--- Модель Некорректна (MODEL_INVALID) ---")
    print("Произошла ошибка при построении модели. Проверьте логику ограничений и определения переменных.")
    print("Сообщение от решателя:", solver.ResponseStats()) # Может дать больше информации
else:
    print(f"\n--- Решатель завершился со статусом: {status} ---")
    print("Статус не является оптимальным или допустимым. Дополнительная информация:")
    print(solver.ResponseStats())