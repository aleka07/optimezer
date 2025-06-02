# scheduler_pulp_multi_machine.py
import math
import collections
import csv
import datetime
from pulp import LpProblem, LpMinimize, LpVariable, lpSum, LpStatus, PULP_CBC_CMD
import itertools # For combinations
import config # Import your existing config file

def time_str_to_minutes_int(time_str):
    try:
        parts = list(map(int, time_str.split(':')))
        if len(parts) == 3: h, m, s = parts; return round(h * 60 + m + s / 60.0)
        elif len(parts) == 2: m, s = parts; return round(m + s / 60.0)
        else: return 0
    except: return 0

def run_pulp_multi_machine_scheduler():
    print("--- Начало процесса планирования (PuLP MILP - Multi-Machine) ---")

    tech_map_data = config.tech_map_data
    orders = config.orders_simple
    machines_available = config.machines_available
    BATCH_SIZE = config.BATCH_SIZE
    STAGES = config.STAGES
    OUTPUT_CSV_FILE = config.OUTPUT_CSV_FILE.replace(".csv", "_pulp_mm.csv")
    OUTPUT_TXT_FILE = config.OUTPUT_TXT_FILE.replace(".txt", "_pulp_mm.txt")

    tech_map_minutes_int = {}
    for product, stages_data in tech_map_data.items():
        if product not in orders or orders[product] <= 0: continue
        tech_map_minutes_int[product] = {}
        for stage_name in STAGES:
            time_str = stages_data.get(stage_name, "0:00:00")
            duration = time_str_to_minutes_int(time_str)
            tech_map_minutes_int[product][stage_name] = duration

    all_tasks_details = []
    task_counter = 0
    for product_name, quantity_ordered in orders.items():
        if quantity_ordered <= 0: continue
        if product_name not in tech_map_data:
            print(f"Предупреждение: Продукт '{product_name}' из заказа отсутствует. Пропускается.")
            continue
        num_batches = math.ceil(quantity_ordered / BATCH_SIZE)
        for batch_num in range(num_batches):
            batch_id = f"{product_name}_batch_{batch_num+1}"
            current_batch_tasks = []
            for stage_idx, stage_name in enumerate(STAGES):
                duration = tech_map_minutes_int.get(product_name, {}).get(stage_name, 0)
                if duration > 0:
                    task_id = f"task_{task_counter}"
                    current_batch_tasks.append({
                        "id": task_id, "batch_id": batch_id, "product": product_name,
                        "stage_name": stage_name, "stage_idx": stage_idx, "duration": duration
                    })
                    task_counter += 1
            current_batch_tasks.sort(key=lambda t: t['stage_idx'])
            all_tasks_details.extend(current_batch_tasks)

    if not all_tasks_details:
        print("Нет задач для производства.")
        return

    print(f"Всего задач (операций) сгенерировано: {len(all_tasks_details)}")
    horizon = sum(t['duration'] for t in all_tasks_details)
    if horizon == 0 and all_tasks_details: horizon = 1
    if not all_tasks_details: horizon = 0
    print(f"Расчетный горизонт (Big-M): {horizon} минут")

    model = LpProblem("BakeryScheduling_PuLP_MM", LpMinimize)
    start_vars = {
        task['id']: LpVariable(f"start_{task['id']}", 0, horizon, 'Continuous')
        for task in all_tasks_details
    }
    makespan_var = LpVariable("makespan", 0, cat='Continuous')
    model += makespan_var, "Minimize_Makespan"

    task_map = {task['id']: task for task in all_tasks_details}
    tasks_by_batch_id = collections.defaultdict(list)
    for task in all_tasks_details:
        tasks_by_batch_id[task['batch_id']].append(task)

    for batch_id, tasks_in_batch in tasks_by_batch_id.items():
        for i in range(len(tasks_in_batch) - 1):
            curr = tasks_in_batch[i]
            next_t = tasks_in_batch[i+1]
            model += start_vars[next_t['id']] >= start_vars[curr['id']] + curr['duration'], \
                     f"seq_{curr['id']}_{next_t['id']}"
        
        def get_task_by_stage(b_tasks, s_name):
            return next((t for t in b_tasks if t['stage_name'] == s_name), None)

        critical_pairs = [
            (config.CRITICAL_STAGE_BEFORE_0, config.CRITICAL_STAGE_AFTER_0, config.MAX_WAIT_COMBINING_MIXING_MIN, "spoil0"),
            (config.CRITICAL_STAGE_BEFORE_1, config.CRITICAL_STAGE_AFTER_1, config.MAX_WAIT_MIXING_FORMING_MIN, "spoil1"),
            (config.CRITICAL_STAGE_BEFORE_2, config.CRITICAL_STAGE_AFTER_2, config.MAX_WAIT_FORMING_PROOFING_MIN, "spoil2"),
            (config.CRITICAL_STAGE_BEFORE_3, config.CRITICAL_STAGE_AFTER_3, config.MAX_WAIT_PROOFING_BAKING_MIN, "spoil3"),
        ]
        for cs_before, cs_after, max_wait, c_name in critical_pairs:
            task_b = get_task_by_stage(tasks_in_batch, cs_before)
            task_a = get_task_by_stage(tasks_in_batch, cs_after)
            if task_b and task_a:
                model += start_vars[task_a['id']] - (start_vars[task_b['id']] + task_b['duration']) <= max_wait, \
                         f"{c_name}_{batch_id}"

    for task in all_tasks_details:
        model += makespan_var >= start_vars[task['id']] + task_map[task['id']]['duration'], \
                 f"makespan_def_{task['id']}"

    BIG_M = horizon + 1 
    tasks_by_stage = collections.defaultdict(list)
    for task in all_tasks_details:
        tasks_by_stage[task['stage_name']].append(task['id'])

    # y_ij_precedes = 1 if task i finishes before task j starts (on same machine type)
    # z_ij_overlap = 1 if task i and task j overlap
    # For a more direct cumulative-like constraint, we can use a slightly different formulation for multi-machine
    # For every pair of tasks (ti, tj) on the same stage:
    #   y_ij = 1 if ti precedes tj, y_ji = 1 if tj precedes ti. y_ij + y_ji <= 1 (or could be =1 if they must be ordered)
    #   start_i + duration_i <= start_j + M * (1 - y_ij)
    #   start_j + duration_j <= start_i + M * (1 - y_ji)
    # This sets up ordering for all pairs.
    
    # Then, for any C+1 tasks on a stage with C machines, at least one of these y_ab + y_ba must be 1.
    # This means at least one pair is ordered.
    
    # Binary variables y_ij = 1 if task i precedes task j (on the same stage)
    # These are only needed for tasks on the *same* stage.
    y_precedes = {} # (task1_id, task2_id) -> LpVariable, where task1_id < task2_id lexicographically for uniqueness

    for stage_name, task_ids_on_stage in tasks_by_stage.items():
        if len(task_ids_on_stage) > 1:
            # Sort task_ids to make pairs unique (t1_id < t2_id)
            sorted_task_ids = sorted(task_ids_on_stage)
            for i in range(len(sorted_task_ids)):
                for j in range(i + 1, len(sorted_task_ids)):
                    t1_id = sorted_task_ids[i]
                    t2_id = sorted_task_ids[j]
                    # y_precedes[(t1_id, t2_id)] = 1 if t1 before t2
                    # y_precedes[(t2_id, t1_id)] will be used for t2 before t1 implicitly (1 - y_precedes[(t1_id, t2_id)])
                    # No, let's be explicit:
                    y_precedes[(t1_id, t2_id)] = LpVariable(f"y_{t1_id}_prec_{t2_id}", cat='Binary') # t1 before t2
                    y_precedes[(t2_id, t1_id)] = LpVariable(f"y_{t2_id}_prec_{t1_id}", cat='Binary') # t2 before t1
                    
                    # Ensure only one precedence order or they don't overlap
                    model += y_precedes[(t1_id, t2_id)] + y_precedes[(t2_id, t1_id)] <= 1, \
                             f"prec_order_excl_{t1_id}_{t2_id}"

                    # Link precedence to start times
                    t1_duration = task_map[t1_id]['duration']
                    t2_duration = task_map[t2_id]['duration']

                    model += start_vars[t1_id] + t1_duration <= start_vars[t2_id] + BIG_M * (1 - y_precedes[(t1_id, t2_id)]), \
                             f"disj_{t1_id}_before_{t2_id}"
                    model += start_vars[t2_id] + t2_duration <= start_vars[t1_id] + BIG_M * (1 - y_precedes[(t2_id, t1_id)]), \
                             f"disj_{t2_id}_before_{t1_id}"
    
    # Now, for stages with C machines, for any C+1 tasks, at least one pair must be ordered
    for stage_name, task_ids_on_stage in tasks_by_stage.items():
        C = machines_available.get(stage_name, 1) # Default to 1 if not specified
        if C <= 0: C = 1 # Treat 0 or negative as 1 machine
        
        if len(task_ids_on_stage) > C:
            print(f"Добавление ограничений на кол-во машин ({C}) для этапа: {stage_name} ({len(task_ids_on_stage)} задач)")
            
            # For every combination of C + 1 tasks on this stage
            for combo_indices in itertools.combinations(range(len(task_ids_on_stage)), C + 1):
                tasks_in_combo = [task_ids_on_stage[i] for i in combo_indices]
                
                # For this combo, at least one pair must be ordered
                # Sum of (y_ij + y_ji) for all pairs (i,j) in tasks_in_combo must be >= 1
                ordered_pairs_sum = []
                for i in range(len(tasks_in_combo)):
                    for j in range(i + 1, len(tasks_in_combo)):
                        t1_id_combo = tasks_in_combo[i]
                        t2_id_combo = tasks_in_combo[j]
                        
                        # Ensure canonical order for y_precedes keys
                        key1 = (t1_id_combo, t2_id_combo) if t1_id_combo < t2_id_combo else (t2_id_combo, t1_id_combo)
                        key2 = (t2_id_combo, t1_id_combo) if t1_id_combo < t2_id_combo else (t1_id_combo, t2_id_combo)
                        
                        # We need to refer to the correct y_precedes variables
                        # y_t1_prec_t2 and y_t2_prec_t1
                        if (t1_id_combo, t2_id_combo) in y_precedes:
                             ordered_pairs_sum.append(y_precedes[(t1_id_combo, t2_id_combo)])
                        if (t2_id_combo, t1_id_combo) in y_precedes:
                             ordered_pairs_sum.append(y_precedes[(t2_id_combo, t1_id_combo)])
                
                if ordered_pairs_sum: # If any pairs were formed
                    model += lpSum(ordered_pairs_sum) >= 1, \
                             f"capacity_{stage_name}_combo_{'_'.join(sorted(tasks_in_combo))}"
                else:
                    # This case should not happen if C+1 >= 2
                    pass


    print("\nЗапуск решателя PuLP (CBC) для Multi-Machine...")
    solver_options = {"msg": 1} # Show solver output
    if config.SOLVER_TIME_LIMIT_SECONDS is not None:
        solver_options["timeLimit"] = config.SOLVER_TIME_LIMIT_SECONDS
    
    solver = PULP_CBC_CMD(**solver_options)
    # model.writeLP("bakery_mm.lp") # DEBUG: write model to file
    # print("Модель записана в bakery_mm.lp")
    model.solve(solver)
    print("Решатель PuLP (Multi-Machine) завершил работу.")

    status_text = LpStatus[model.status]
    print(f"Статус решения: {status_text} ({model.status})")

    if model.status == 1: # Optimal
        optimal_makespan_minutes = makespan_var.value()
        print(f"\n--- Оптимальное Расписание Найдено (PuLP MM) ---")
        print(f"Makespan: {optimal_makespan_minutes:.2f} минут")
        # (Rest of the output processing is similar to the previous PuLP script)
        # ... (Copied and adapted from previous PuLP script)
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

        for task_detail in all_tasks_details:
            task_id = task_detail['id']
            start_val = start_vars[task_id].value()
            duration_val = task_detail['duration']
            end_val = start_val + duration_val
            
            schedule_data_for_output.append({
                "Batch_ID": task_detail['batch_id'], "Stage": task_detail['stage_name'],
                "Start_Time_Min": round(start_val, 2), "End_Time_Min": round(end_val, 2),
                "Duration_Min": round(duration_val, 2),
                "Stage_Order": stage_order_map.get(task_detail['stage_name'], 999)
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
        except Exception as e: print(f"\nОшибка записи CSV файла '{OUTPUT_CSV_FILE}': {e}")
        try:
            with open(OUTPUT_TXT_FILE, 'w', encoding='utf-8') as txtfile:
                txtfile.write(f"--- Сводка (PuLP MILP - Multi-Machine) ---\n\n")
                txtfile.write(f"Статус: {status_text}\n")
                txtfile.write(f"Makespan: {optimal_makespan_minutes:.2f} мин ({makespan_formatted})\n")
                txtfile.write(f"Всего задач: {len(all_tasks_details)}\n")
                txtfile.write(f"CSV: {OUTPUT_CSV_FILE}\n")
            print(f"Сводная информация успешно записана в TXT файл: '{OUTPUT_TXT_FILE}'")
        except Exception as e: print(f"\nОшибка записи TXT файла '{OUTPUT_TXT_FILE}': {e}")

    # ... (Other status handling from previous PuLP script)
    elif model.status == 0: print("\n--- Решение не было начато (PuLP MM) ---")
    elif model.status == -1:
        print("\n--- Задача Неразрешима (Infeasible - PuLP MM) ---")
        # model.writeLP("bakery_mm_infeasible.lp")
        # print("Модель записана в bakery_mm_infeasible.lp для анализа.")
    elif model.status == -2: print("\n--- Задача Неограничена (Unbounded - PuLP MM) ---")
    elif model.status == -3: print("\n--- Статус решения Неопределен (PuLP MM) ---")
    else: print(f"\n--- Решатель PuLP MM завершился со статусом: {status_text} ---")


if __name__ == "__main__":
    run_pulp_multi_machine_scheduler()