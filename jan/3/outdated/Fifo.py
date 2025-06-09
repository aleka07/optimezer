import math
import collections
import csv
import datetime
import heapq # For efficiently managing machine free times

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
    "Комбинирование": 2,
    "Смешивание": 3,
    "Формовка": 2,
    "Расстойка": 8,
    "Выпекание": 6,
    "Остывание": 25,
}

BATCH_SIZE = 100
STAGES = [
    "Комбинирование", "Смешивание", "Формовка", "Расстойка",
    "Выпекание", "Остывание",
]
proportional_time_stages = ["Комбинирование", "Формовка"]

OUTPUT_CSV_FILE = 'production_schedule_fifo.csv'
OUTPUT_TXT_FILE = 'production_summary_fifo.txt'

# 2. Helper Function & Preprocessing (Copied and adapted)
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
    for stage_name in STAGES:
        time_str = stages_data.get(stage_name, "0:00:00")
        duration = time_str_to_minutes_int(time_str)
        tech_map_minutes_int[product][stage_name] = duration

all_batches_tasks = [] # Will store a flat list of all tasks with batch info

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
        
        # Store tasks for this batch, to be processed sequentially
        current_batch_task_list = []
        for stage_index, stage_name in enumerate(STAGES):
            base_duration_for_100 = tech_map_minutes_int.get(product, {}).get(stage_name, 0)
            current_task_duration = base_duration_for_100

            if base_duration_for_100 > 0:
                if is_last_partial_batch:
                    if stage_name in proportional_time_stages:
                        current_task_duration = math.ceil(base_duration_for_100 * (current_batch_actual_size / BATCH_SIZE))
                
                if current_task_duration <= 0 and base_duration_for_100 > 0:
                    current_task_duration = 1
                
                if current_task_duration > 0:
                    current_batch_task_list.append({
                        "batch_id": batch_id, 
                        "product": product, # Store product for reference if needed
                        "stage_index": stage_index,
                        "stage_name": stage_name, 
                        "duration": current_task_duration,
                    })
        if current_batch_task_list:
            all_batches_tasks.append({"id": batch_id, "tasks": current_batch_task_list})


if not all_batches_tasks:
    print("Нет партий для производства. Проверьте заказы и технологическую карту.")
    exit()

print(f"Всего партий сгенерировано: {len(all_batches_tasks)}")
num_tasks_total = sum(len(b['tasks']) for b in all_batches_tasks)
print(f"Всего задач (операций) с ненулевой длительностью: {num_tasks_total}")


# 3. FIFO Scheduling Logic
scheduled_tasks = []
machine_finish_times = {} # Key: stage_name, Value: list of finish times for each machine of this type (using min-heap)
for stage, count in machines_available.items():
    machine_finish_times[stage] = [0] * count # All machines start at time 0
    heapq.heapify(machine_finish_times[stage]) # Convert to min-heap

# Keep track of when the last task for a particular batch finishes
batch_previous_task_end_time = collections.defaultdict(int) 

# Process batches in the order they were generated (FIFO for batches)
for batch_info in all_batches_tasks:
    batch_id = batch_info['id']
    
    for task in batch_info['tasks']:
        stage_name = task['stage_name']
        duration = task['duration']
        
        if machines_available.get(stage_name, 0) == 0:
            print(f"Предупреждение: Нет доступных машин для этапа '{stage_name}'. Задача для партии '{batch_id}' пропускается.")
            continue

        # Earliest time this task can start due to previous task in the same batch
        min_start_time_due_to_sequence = batch_previous_task_end_time[batch_id]
        
        # Earliest time a machine for this stage is available
        # heapq.heappop gives the smallest (earliest) finish time
        earliest_machine_available_time = heapq.heappop(machine_finish_times[stage_name])
        
        # Actual start time is the later of the two
        start_time = max(min_start_time_due_to_sequence, earliest_machine_available_time)
        end_time = start_time + duration
        
        # Add this task to the schedule
        scheduled_tasks.append({
            "Batch_ID": batch_id,
            "Stage": stage_name,
            "Start_Time_Min": start_time,
            "End_Time_Min": end_time,
            "Duration_Min": duration,
            "Stage_Order": task['stage_index'] # For sorting output if needed
        })
        
        # Update the finish time for the machine that was just used
        heapq.heappush(machine_finish_times[stage_name], end_time)
        
        # Update the end time for the current batch's last completed task
        batch_previous_task_end_time[batch_id] = end_time

# 4. Process Results and Write to Files
if scheduled_tasks:
    final_makespan_minutes = 0
    for task in scheduled_tasks:
        if task["End_Time_Min"] > final_makespan_minutes:
            final_makespan_minutes = task["End_Time_Min"]
            
    print("\n--- FIFO Расписание Сгенерировано ---")
    print(f"Общее Время Производства (Makespan): {final_makespan_minutes:.2f} минут")
    
    total_seconds_makespan = int(final_makespan_minutes * 60)
    tdelta = datetime.timedelta(seconds=total_seconds_makespan)
    days = tdelta.days
    hours, remainder = divmod(tdelta.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    makespan_formatted = ""
    if days > 0: makespan_formatted += f"{days} дн "
    makespan_formatted += f"{hours:02}:{minutes:02}:{seconds:02}"
    print(f"Что составляет примерно: {makespan_formatted}")

    # Sort for output consistency
    scheduled_tasks.sort(key=lambda x: (x['Start_Time_Min'], x['Batch_ID'], x['Stage_Order']))

    csv_fieldnames = ["Batch_ID", "Stage", "Start_Time_Min", "End_Time_Min", "Duration_Min"]
    try:
        with open(OUTPUT_CSV_FILE, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_fieldnames)
            writer.writeheader()
            for row_data in scheduled_tasks:
                row_to_write = {key: row_data[key] for key in csv_fieldnames}
                writer.writerow(row_to_write)
        print(f"\nРасписание успешно записано в CSV файл: '{OUTPUT_CSV_FILE}'")
    except Exception as e:
        print(f"\nОшибка записи CSV файла '{OUTPUT_CSV_FILE}': {e}")

    try:
        with open(OUTPUT_TXT_FILE, 'w', encoding='utf-8') as txtfile:
            txtfile.write("--- Сводка по Производственному Расписанию (FIFO) ---\n\n")
            txtfile.write(f"Метод планирования: FIFO (First-In, First-Out)\n")
            txtfile.write(f"Общее время производства (Makespan): {final_makespan_minutes:.2f} минут\n")
            txtfile.write(f"Общее время производства (формат): {makespan_formatted}\n")
            txtfile.write(f"Всего партий: {len(all_batches_tasks)}\n")
            txtfile.write(f"Всего задач (операций) в расписании: {len(scheduled_tasks)}\n")
            txtfile.write(f"\nРазмер партии (BATCH_SIZE): {BATCH_SIZE}\n")
            txtfile.write(f"Этапы с пропорциональным временем для неполных партий: {', '.join(proportional_time_stages)}\n")
            txtfile.write(f"\nДоступные ресурсы (машины):\n")
            for stage_name_res, count in machines_available.items():
                txtfile.write(f"  - {stage_name_res}: {count}\n")
            txtfile.write(f"\nФайл с детальным расписанием: {OUTPUT_CSV_FILE}\n")
        print(f"Сводная информация успешно записана в TXT файл: '{OUTPUT_TXT_FILE}'")
    except Exception as e:
        print(f"\nОшибка записи TXT файла '{OUTPUT_TXT_FILE}': {e}")
else:
    print("Не удалось сгенерировать расписание. Проверьте входные данные.")