# scheduler_heuristic.py
import math
import collections
import csv
import datetime
import heapq # For a priority queue (min-heap) if we want more advanced ready list management
import config

def time_str_to_minutes_int(time_str):
    try:
        parts = list(map(int, time_str.split(':')))
        if len(parts) == 3: h, m, s = parts; return round(h * 60 + m + s / 60.0)
        elif len(parts) == 2: m, s = parts; return round(m + s / 60.0)
        else: return 0
    except: return 0

# --- Constants for task status ---
UNPROCESSED = 0
READY = 1
RUNNING = 2
COMPLETED = 3

def run_heuristic_scheduler():
    print("--- Начало процесса планирования (Heuristic Dispatching Rule) ---")

    tech_map_data = config.tech_map_data
    orders = config.orders_simple
    machines_available = config.machines_available
    BATCH_SIZE = config.BATCH_SIZE
    STAGES = config.STAGES
    OUTPUT_CSV_FILE = config.OUTPUT_CSV_FILE.replace(".csv", "_heuristic.csv")
    OUTPUT_TXT_FILE = config.OUTPUT_TXT_FILE.replace(".txt", "_heuristic.txt")

    tech_map_minutes_int = {}
    for product, stages_data in tech_map_data.items():
        if product not in orders or orders[product] <= 0: continue
        tech_map_minutes_int[product] = {}
        for stage_name in STAGES:
            time_str = stages_data.get(stage_name, "0:00:00")
            duration = time_str_to_minutes_int(time_str)
            tech_map_minutes_int[product][stage_name] = duration

    # --- Preprocessing: Create a flat list of all tasks ---
    # Each task dict will also store its 'status', 'start_time', 'end_time'
    all_tasks_list = [] # List of dicts
    task_counter = 0
    task_id_to_idx_map = {} # For quick lookup of task's index in all_tasks_list

    # Store predecessors for each task (within the same batch)
    # {task_id: predecessor_task_id_or_None}
    task_predecessors = {}
    # Store successors for each task (within the same batch)
    task_successors = {}


    for product_name, quantity_ordered in orders.items():
        if quantity_ordered <= 0: continue
        if product_name not in tech_map_data:
            print(f"Предупреждение: Продукт '{product_name}' из заказа отсутствует. Пропускается.")
            continue
        num_batches = math.ceil(quantity_ordered / BATCH_SIZE)
        for batch_num in range(num_batches):
            batch_id = f"{product_name}_batch_{batch_num+1}"
            
            tasks_in_this_batch_temp = []
            for stage_idx_order, stage_name in enumerate(STAGES): # Use original STAGES order
                duration = tech_map_minutes_int.get(product_name, {}).get(stage_name, 0)
                if duration > 0:
                    task_id = f"task_{task_counter}"
                    tasks_in_this_batch_temp.append({
                        "id": task_id, "batch_id": batch_id, "product": product_name,
                        "stage_name": stage_name, 
                        "stage_idx_in_product_flow": stage_idx_order, # Relative order in this product's flow
                        "duration": duration,
                        "status": UNPROCESSED, "start_time": -1, "end_time": -1
                    })
                    task_counter += 1
            
            # Sort by stage_idx_in_product_flow to establish precedence within the batch
            tasks_in_this_batch_temp.sort(key=lambda t: t['stage_idx_in_product_flow'])

            # Assign predecessors and successors
            for i in range(len(tasks_in_this_batch_temp)):
                current_task_dict = tasks_in_this_batch_temp[i]
                all_tasks_list.append(current_task_dict)
                task_id_to_idx_map[current_task_dict['id']] = len(all_tasks_list) - 1

                if i > 0:
                    predecessor_task_dict = tasks_in_this_batch_temp[i-1]
                    task_predecessors[current_task_dict['id']] = predecessor_task_dict['id']
                    task_successors[predecessor_task_dict['id']] = current_task_dict['id']
                else:
                    task_predecessors[current_task_dict['id']] = None # First task in batch
                
                if i == len(tasks_in_this_batch_temp) -1:
                    task_successors[current_task_dict['id']] = None # Last task in batch

    if not all_tasks_list:
        print("Нет задач для производства.")
        return

    print(f"Всего задач (операций) сгенерировано: {len(all_tasks_list)}")

    # --- Heuristic Scheduling Logic ---
    currentTime = 0
    completed_task_count = 0

    # machine_free_time[stage_name] = list of times when each machine of that stage becomes free
    machine_free_time = collections.defaultdict(list)
    for stage_name, count in machines_available.items():
        if count > 0:
            machine_free_time[stage_name] = [0] * count # All machines start free at time 0
        else: # No machines for this stage
            machine_free_time[stage_name] = []


    # Spoilage mapping: { (before_stage, after_stage): max_wait_time }
    spoilage_rules = {
        (config.CRITICAL_STAGE_BEFORE_0, config.CRITICAL_STAGE_AFTER_0): config.MAX_WAIT_COMBINING_MIXING_MIN,
        (config.CRITICAL_STAGE_BEFORE_1, config.CRITICAL_STAGE_AFTER_1): config.MAX_WAIT_MIXING_FORMING_MIN,
        (config.CRITICAL_STAGE_BEFORE_2, config.CRITICAL_STAGE_AFTER_2): config.MAX_WAIT_FORMING_PROOFING_MIN,
        (config.CRITICAL_STAGE_BEFORE_3, config.CRITICAL_STAGE_AFTER_3): config.MAX_WAIT_PROOFING_BAKING_MIN,
    }

    # Main simulation loop
    # We can use a list of active events (task completions) to jump time
    # event_queue = [(finish_time, task_id, machine_stage, machine_idx)] heapq
    event_queue = [] 

    # Initial population of ready tasks
    for task_idx, task_data in enumerate(all_tasks_list):
        if task_predecessors[task_data['id']] is None and task_data['status'] == UNPROCESSED:
            task_data['status'] = READY

    scheduled_this_iteration = True # Loop control
    while completed_task_count < len(all_tasks_list) :
        if not scheduled_this_iteration and not event_queue : #Stuck
             print("АЛГОРИТМ ЗАСТРЯЛ: Нет запланированных задач и нет событий в очереди. Проверьте логику.")
             # This might happen if no tasks can be made ready due to strict spoilage or resource deadlocks
             # For now, break and output what we have.
             break

        # --- Try to schedule new tasks ---
        # Collect all currently READY tasks
        ready_to_schedule_candidates = []
        for task_idx, task_data in enumerate(all_tasks_list):
            if task_data['status'] == READY:
                # Check if machines are available for this task's stage
                stage_machines = machine_free_time.get(task_data['stage_name'], [])
                if not stage_machines: # No machines for this stage type
                    # print(f"Warning: No machines for stage {task_data['stage_name']}, task {task_data['id']} cannot be scheduled.")
                    continue # This task can't run if its stage has no machines

                earliest_machine_available_time = min(stage_machines)
                
                # Determine earliest possible start time (EPST)
                # 1. Must be >= currentTime (or machine free time)
                # 2. Must respect spoilage from its direct critical predecessor
                
                possible_start_time = max(currentTime, earliest_machine_available_time)
                
                # Spoilage check
                predecessor_task_id = task_predecessors[task_data['id']]
                max_allowed_start_if_spoilage = float('inf')

                if predecessor_task_id: # If it has a general predecessor
                    pred_task_data = all_tasks_list[task_id_to_idx_map[predecessor_task_id]]
                    # Ensure predecessor is completed (should be by READY logic, but double check)
                    if pred_task_data['status'] != COMPLETED:
                        continue # Predecessor not done, so this task is not truly ready by precedence.

                    possible_start_time = max(possible_start_time, pred_task_data['end_time'])

                    # Check if current task is an "after_critical" stage for a spoilage rule
                    for (before_crit, after_crit), max_wait in spoilage_rules.items():
                        if pred_task_data['stage_name'] == before_crit and \
                           task_data['stage_name'] == after_crit:
                            max_allowed_start_if_spoilage = pred_task_data['end_time'] + max_wait
                            break # Found the relevant spoilage rule
                
                if possible_start_time > max_allowed_start_if_spoilage:
                    # This task cannot be scheduled now due to spoilage from its predecessor
                    # It might become schedulable if currentTime advances due to other tasks.
                    # Or, this indicates a problem if currentTime is already past this window.
                    # For a simple heuristic, we might just skip it for now.
                    # print(f"Task {task_data['id']} skipped due to spoilage: PST {possible_start_time} > MAX_AST {max_allowed_start_if_spoilage}")
                    continue

                ready_to_schedule_candidates.append({
                    "task_idx": task_idx,
                    "task_id": task_data['id'],
                    "epst": possible_start_time, # Earliest Possible Start Time
                    "duration": task_data['duration'],
                    "stage_name": task_data['stage_name']
                })
        
        # --- Dispatching Rule: Schedule one task if possible ---
        # Priority: 1. Min EPST, 2. Min Duration (SPT as tie-breaker)
        scheduled_this_iteration = False
        if ready_to_schedule_candidates:
            ready_to_schedule_candidates.sort(key=lambda x: (x['epst'], x['duration']))
            
            best_task_to_schedule_info = ready_to_schedule_candidates[0]
            task_idx_to_schedule = best_task_to_schedule_info['task_idx']
            task_to_schedule_data = all_tasks_list[task_idx_to_schedule]

            # Find the specific machine instance
            target_stage = task_to_schedule_data['stage_name']
            chosen_machine_idx = -1
            min_free_time_for_machine = float('inf')

            for m_idx, free_t in enumerate(machine_free_time[target_stage]):
                if free_t <= best_task_to_schedule_info['epst'] and free_t < min_free_time_for_machine: # Machine is free by EPST
                    min_free_time_for_machine = free_t
                    chosen_machine_idx = m_idx
            
            # If multiple machines are free at EPST, pick the one with the lowest index (arbitrary tie-break)
            if chosen_machine_idx == -1: # Should find one if epst was calculated based on min(stage_machines)
                 # This means the earliest_machine_available_time was > currentTime, so EPST is that future time
                 # Find the machine that becomes free earliest at or after EPST
                candidate_machines = []
                for m_idx, free_t in enumerate(machine_free_time[target_stage]):
                    candidate_machines.append((free_t, m_idx))
                candidate_machines.sort()
                chosen_machine_idx = candidate_machines[0][1]


            actual_start_time = max(best_task_to_schedule_info['epst'], machine_free_time[target_stage][chosen_machine_idx])
            
            # Final spoilage check against actual_start_time
            predecessor_task_id = task_predecessors[task_to_schedule_data['id']]
            can_schedule = True
            if predecessor_task_id:
                pred_task_data = all_tasks_list[task_id_to_idx_map[predecessor_task_id]]
                for (before_crit, after_crit), max_wait in spoilage_rules.items():
                    if pred_task_data['stage_name'] == before_crit and \
                        task_to_schedule_data['stage_name'] == after_crit:
                        if actual_start_time > pred_task_data['end_time'] + max_wait:
                            #print(f"Task {task_to_schedule_data['id']} ultimately failed spoilage on AST {actual_start_time}")
                            can_schedule = False 
                            # This task is problematic. For this iteration, we might just not schedule it
                            # and hope other tasks advancing time resolves it.
                            # A more complex heuristic might try to delay the predecessor or re-evaluate.
                            # We remove it from candidates for this iteration
                            ready_to_schedule_candidates.pop(0) # Remove the one we tried
                            if not ready_to_schedule_candidates: # If it was the only one
                                pass # Will proceed to event queue or advance time
                            else: # Try the next candidate in the already sorted list
                                # This part makes the loop a bit more complex, for now, let's assume if it fails here, we move on
                                # For simplicity, if the best one fails final check, we skip scheduling in this "immediate" phase
                                # and rely on time advancing via event_queue.
                                pass # Fall through to event processing or time advance
                            break 
                if not can_schedule:
                    pass # Fall through to event processing or time advance

            if can_schedule:
                task_to_schedule_data['status'] = RUNNING
                task_to_schedule_data['start_time'] = actual_start_time
                task_to_schedule_data['end_time'] = actual_start_time + task_to_schedule_data['duration']
                
                machine_free_time[target_stage][chosen_machine_idx] = task_to_schedule_data['end_time']
                
                heapq.heappush(event_queue, (task_to_schedule_data['end_time'], task_to_schedule_data['id']))
                scheduled_this_iteration = True
                # print(f"Scheduled: {task_to_schedule_data['id']} on {target_stage}[{chosen_machine_idx}] at {actual_start_time} to {task_to_schedule_data['end_time']}")


        # --- Advance Time and Process Events ---
        if not scheduled_this_iteration and not event_queue: # No new tasks scheduled, and no pending completions
            if completed_task_count < len(all_tasks_list):
                print(f"Warning: Deadlock or unable to schedule remaining {len(all_tasks_list) - completed_task_count} tasks. CurrentTime: {currentTime}. Outputting partial schedule.")
            break # Exit loop

        if not scheduled_this_iteration and event_queue: # No new tasks could be immediately scheduled, advance to next event
            next_event_time, finished_task_id = heapq.heappop(event_queue)
            currentTime = max(currentTime, next_event_time) # Advance time
            
            finished_task_idx = task_id_to_idx_map[finished_task_id]
            all_tasks_list[finished_task_idx]['status'] = COMPLETED
            completed_task_count += 1
            # print(f"Completed: {finished_task_id} at {currentTime}. Total done: {completed_task_count}")

            # Make successor ready if it exists
            successor_task_id = task_successors.get(finished_task_id)
            if successor_task_id:
                successor_idx = task_id_to_idx_map[successor_task_id]
                if all_tasks_list[successor_idx]['status'] == UNPROCESSED: # And its direct batch predecessor is now done
                    all_tasks_list[successor_idx]['status'] = READY
                    # print(f"Made ready: {successor_task_id}")
            
            scheduled_this_iteration = True # Set to true to re-evaluate scheduling immediately at new currentTime

        elif not event_queue and completed_task_count < len(all_tasks_list):
            # This should ideally not be reached if the deadlock check above is correct
            # but as a fallback, if nothing scheduled and no events, and tasks remain...
            print(f"Warning: No events in queue but tasks remain. Possible issue. currentTime: {currentTime}")
            # Try to advance time to the earliest possible start of any "stuck" ready task if any,
            # or min machine free time. This is a bit of a hack for simple heuristic.
            min_next_possible_time = float('inf')
            has_ready_tasks_stuck = False
            for task_data_rt in all_tasks_list:
                if task_data_rt['status'] == READY:
                    has_ready_tasks_stuck = True
                    # Simplistic: just find the earliest a machine for its stage becomes free
                    stage_machines_rt = machine_free_time.get(task_data_rt['stage_name'], [])
                    if stage_machines_rt:
                        min_next_possible_time = min(min_next_possible_time, min(stage_machines_rt))
            
            if has_ready_tasks_stuck and min_next_possible_time != float('inf') and min_next_possible_time > currentTime :
                # print(f"Heuristic advancing time to {min_next_possible_time} to try unstick tasks.")
                currentTime = min_next_possible_time
                scheduled_this_iteration = True # Force re-evaluation
            elif not has_ready_tasks_stuck : # No ready tasks, something is wrong with making tasks ready
                 print("Error: No tasks are ready, but not all tasks completed. Predecessor logic might be flawed.")
                 break
            else: # No way to advance time based on machine freeing
                break # Truly stuck


    # --- Output Results ---
    final_makespan = 0
    for task_data in all_tasks_list:
        if task_data['status'] == COMPLETED:
            final_makespan = max(final_makespan, task_data['end_time'])
        elif task_data['start_time'] == -1 and task_data['duration'] > 0 : # Task was not scheduled
             print(f"Warning: Task {task_data['id']} ({task_data['product']} - {task_data['stage_name']}) was NOT scheduled.")


    print("\n--- Эвристическое Расписание (Приближенное) ---")
    print(f"Общее Время Производства (Makespan): {final_makespan:.2f} минут")
    total_seconds_makespan = int(final_makespan * 60)
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

    for task_data in all_tasks_list:
        if task_data['start_time'] != -1 : # If it was scheduled
            schedule_data_for_output.append({
                "Batch_ID": task_data['batch_id'],
                "Stage": task_data['stage_name'],
                "Start_Time_Min": round(task_data['start_time'], 2),
                "End_Time_Min": round(task_data['end_time'], 2),
                "Duration_Min": round(task_data['duration'], 2),
                "Stage_Order": stage_order_map.get(task_data['stage_name'], 999)
            })
    schedule_data_for_output.sort(key=lambda x: (x['Start_Time_Min'], x['Batch_ID'], x['Stage_Order']))

    csv_fieldnames = ["Batch_ID", "Stage", "Start_Time_Min", "End_Time_Min", "Duration_Min"]
    try:
        with open(OUTPUT_CSV_FILE, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_fieldnames)
            writer.writeheader()
            for row_data in schedule_data_for_output:
                writer.writerow(row_data)
        print(f"\nРасписание успешно записано в CSV файл: '{OUTPUT_CSV_FILE}'")
    except Exception as e: print(f"\nОшибка записи CSV файла '{OUTPUT_CSV_FILE}': {e}")
    
    try:
        with open(OUTPUT_TXT_FILE, 'w', encoding='utf-8') as txtfile:
            txtfile.write(f"--- Сводка (Эвристический алгоритм) ---\n\n")
            txtfile.write(f"Общее время производства (Makespan): {final_makespan:.2f} минут ({makespan_formatted})\n")
            txtfile.write(f"Всего задач: {len(all_tasks_list)}, Запланировано: {len(schedule_data_for_output)}\n")
            txtfile.write(f"Использованная эвристика: Earliest Possible Start Time (EPST) + Shortest Processing Time (SPT) tie-breaker.\n")
            txtfile.write(f"CSV: {OUTPUT_CSV_FILE}\n")
        print(f"Сводная информация успешно записана в TXT файл: '{OUTPUT_TXT_FILE}'")
    except Exception as e: print(f"\nОшибка записи TXT файла '{OUTPUT_TXT_FILE}': {e}")


if __name__ == "__main__":
    run_heuristic_scheduler()