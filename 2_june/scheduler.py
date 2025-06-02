# scheduler.py
import math
import collections
import csv
import datetime
from ortools.sat.python import cp_model
import config # Импортируем наш конфигурационный файл

# 2. Helper Function & Preprocessing
def time_str_to_minutes_int(time_str):
    try:
        parts = list(map(int, time_str.split(':')))
        if len(parts) == 3: h, m, s = parts; return round(h * 60 + m + s / 60.0)
        elif len(parts) == 2: m, s = parts; return round(m + s / 60.0)
        else: return 0
    except: return 0

def run_scheduler():
    print("--- Начало процесса планирования ---")

    # --- Используем данные из config.py ---
    tech_map_data = config.tech_map_data
    orders = config.orders_simple
    machines_available = config.machines_available
    BATCH_SIZE = config.BATCH_SIZE
    STAGES = config.STAGES
    OUTPUT_CSV_FILE = config.OUTPUT_CSV_FILE
    OUTPUT_TXT_FILE = config.OUTPUT_TXT_FILE

    tech_map_minutes_int = {}
    for product, stages_data in tech_map_data.items():
        if product not in orders or orders[product] <= 0:
            continue
        tech_map_minutes_int[product] = {}
        for stage_name in STAGES:
            time_str = stages_data.get(stage_name, "0:00:00")
            duration = time_str_to_minutes_int(time_str)
            tech_map_minutes_int[product][stage_name] = duration

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
                duration = tech_map_minutes_int.get(product, {}).get(stage_name, 0)
                if duration > 0:
                    batch_tasks.append({
                        "batch_id": batch_id, "stage_index": stage_index,
                        "stage_name": stage_name, "duration": duration, })
            if batch_tasks: all_batches.append({"id": batch_id, "product": product, "tasks": batch_tasks})

    if not all_batches:
        print("Нет партий для производства. Проверьте заказы и технологическую карту.")
        return # Выход, если нет партий

    print(f"Всего партий сгенерировано: {len(all_batches)}")
    num_tasks_total = sum(len(b['tasks']) for b in all_batches)
    print(f"Всего задач (операций) с ненулевой длительностью: {num_tasks_total}")

    horizon = sum(tech_map_minutes_int[batch['product']][task['stage_name']] for batch in all_batches for task in batch['tasks'])
    avg_stage_duration_total = 0
    num_stages_total = 0
    for prod_stages in tech_map_minutes_int.values():
        for stage_dur in prod_stages.values():
            if stage_dur > 0:
                avg_stage_duration_total += stage_dur
                num_stages_total +=1
    avg_stage_duration = (avg_stage_duration_total / num_stages_total) if num_stages_total > 0 else 30
    horizon += avg_stage_duration * len(all_batches) * 2 # Запас
    horizon = math.ceil(horizon)
    if horizon == 0 and num_tasks_total > 0 : # Если все длительности нулевые, но задачи есть, горизонт должен быть > 0
        horizon = 1 # Минимальный горизонт
    elif num_tasks_total == 0: # Если нет задач, горизонт может быть 0
         horizon = 0


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
    for stage_index, stage_name in enumerate(STAGES):
        machine_count = machines_available.get(stage_name)
        if machine_count is None or machine_count <= 0:
            continue
        intervals_for_stage = []
        for i, batch in enumerate(all_batches):
             if stage_index in task_vars[i]:
                 intervals_for_stage.append(task_vars[i][stage_index][2])
        if intervals_for_stage:
            demands = [1] * len(intervals_for_stage)
            model.AddCumulative(intervals_for_stage, demands, machine_count)

    # c) Spoilage/Waiting Time Constraints (using constants from config)
    for i, batch in enumerate(all_batches):
        batch_id = batch['id']
        task0_key = (batch_id, config.CRITICAL_STAGE_BEFORE_0)
        task1_key = (batch_id, config.CRITICAL_STAGE_AFTER_0)
        if task0_key in task_lookup and task1_key in task_lookup:
            model.Add(task_lookup[task1_key][0] - task_lookup[task0_key][1] <= config.MAX_WAIT_COMBINING_MIXING_MIN)

        task1_key_new = (batch_id, config.CRITICAL_STAGE_BEFORE_1)
        task2_key_new = (batch_id, config.CRITICAL_STAGE_AFTER_1)
        if task1_key_new in task_lookup and task2_key_new in task_lookup:
            model.Add(task_lookup[task2_key_new][0] - task_lookup[task1_key_new][1] <= config.MAX_WAIT_MIXING_FORMING_MIN)

        task2_key_old1 = (batch_id, config.CRITICAL_STAGE_BEFORE_2)
        task3_key_old1 = (batch_id, config.CRITICAL_STAGE_AFTER_2)
        if task2_key_old1 in task_lookup and task3_key_old1 in task_lookup:
            model.Add(task_lookup[task3_key_old1][0] - task_lookup[task2_key_old1][1] <= config.MAX_WAIT_FORMING_PROOFING_MIN)

        task3_key_old2 = (batch_id, config.CRITICAL_STAGE_BEFORE_3)
        task4_key_old2 = (batch_id, config.CRITICAL_STAGE_AFTER_3)
        if task3_key_old2 in task_lookup and task4_key_old2 in task_lookup:
            model.Add(task_lookup[task4_key_old2][0] - task_lookup[task3_key_old2][1] <= config.MAX_WAIT_PROOFING_BAKING_MIN)

    # d) Makespan
    makespan = model.NewIntVar(0, horizon, 'makespan')
    last_stage_tasks_ends = []
    if all_batches and any(b['tasks'] for b in all_batches): # Ensure there are tasks
        for i, batch in enumerate(all_batches):
             if batch['tasks']:
                actual_last_stage_idx = batch['tasks'][-1]['stage_index']
                if actual_last_stage_idx in task_vars[i]:
                    last_stage_tasks_ends.append(task_vars[i][actual_last_stage_idx][1])
    
    if last_stage_tasks_ends:
        model.AddMaxEquality(makespan, last_stage_tasks_ends)
    else: # No tasks to schedule or all tasks have zero duration and no valid end time
        model.Add(makespan == 0)


    # 6. Define Objective
    model.Minimize(makespan)

    # 7. Solve Model
    solver = cp_model.CpSolver()
    if config.SOLVER_TIME_LIMIT_SECONDS is not None:
        solver.parameters.max_time_in_seconds = config.SOLVER_TIME_LIMIT_SECONDS
    
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
                txtfile.write(f"  - {config.CRITICAL_STAGE_BEFORE_0} -> {config.CRITICAL_STAGE_AFTER_0}: {config.MAX_WAIT_COMBINING_MIXING_MIN} мин\n")
                txtfile.write(f"  - {config.CRITICAL_STAGE_BEFORE_1} -> {config.CRITICAL_STAGE_AFTER_1}: {config.MAX_WAIT_MIXING_FORMING_MIN} мин\n")
                txtfile.write(f"  - {config.CRITICAL_STAGE_BEFORE_2} -> {config.CRITICAL_STAGE_AFTER_2}: {config.MAX_WAIT_FORMING_PROOFING_MIN} мин\n")
                txtfile.write(f"  - {config.CRITICAL_STAGE_BEFORE_3} -> {config.CRITICAL_STAGE_AFTER_3}: {config.MAX_WAIT_PROOFING_BAKING_MIN} мин\n")
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
        print("  - Слишком короткий горизонт планирования (horizon).")
        print("Проверьте параметры и входные данные в 'config.py'.")
    elif status == cp_model.MODEL_INVALID:
        print("\n--- Модель Некорректна (MODEL_INVALID) ---")
        print("Произошла ошибка при построении модели. Проверьте логику ограничений и определения переменных.")
        print("Сообщение от решателя:", solver.ResponseStats())
    else:
        print(f"\n--- Решатель завершился со статусом: {solver.StatusName(status)} ({status}) ---")
        print("Статус не является оптимальным или допустимым. Дополнительная информация:")
        print(solver.ResponseStats())
    
    print("--- Процесс планирования завершен ---")


if __name__ == "__main__":
    run_scheduler()