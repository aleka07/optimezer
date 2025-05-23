# --- START OF FILE fifo_simulator.py ---
import collections
import heapq # Для более эффективного управления машинами (не используется в текущей простой версии)

def simulate_fifo(all_batches_fifo_ordered, machines_available_fifo, stages_fifo_list, 
                  critical_pairs_with_limits=None): # Добавлен аргумент для лимитов ожидания
    """
    Симулирует производственный процесс по принципу FIFO.

    Args:
        all_batches_fifo_ordered (list): Список партий, УЖЕ ОТСОРТИРОВАННЫХ в порядке FIFO.
                                         Каждый элемент - словарь {"id": batch_id, "product": product, "tasks": batch_tasks}
                                         batch_tasks - список словарей с "stage_name", "duration", "stage_index".
        machines_available_fifo (dict): Словарь с количеством доступных машин для каждого этапа.
        stages_fifo_list (list): Список всех этапов в порядке их выполнения.
        critical_pairs_with_limits (dict, optional): Словарь для проверки времен ожидания.
            Формат: {("STAGE_BEFORE", "STAGE_AFTER"): limit_minutes, ...}
            Например: {("Смешивание", "Формовка"): 20, ...}

    Returns:
        tuple: (schedule_fifo, makespan_fifo, waiting_times_log_fifo, waiting_times_violations_fifo)
               schedule_fifo (list): Список выполненных задач с временами начала и конца.
               makespan_fifo (int): Общее время выполнения.
               waiting_times_log_fifo (list): Лог всех фактических времен ожидания между критическими парами.
               waiting_times_violations_fifo (list): Список нарушений времен ожидания.
    """
    print("\n--- Запуск FIFO Симуляции ---")

    machine_free_time = {} # Словарь: {stage_name: [время_освобождения_машины1, ..._машиныN]}
    for stage_name, count in machines_available_fifo.items():
        machine_free_time[stage_name] = [0] * count

    schedule_fifo = []
    batch_stage_end_times = collections.defaultdict(dict) # {batch_id: {stage_name: end_time}}

    waiting_times_log_fifo = []
    waiting_times_violations_fifo = []

    processed_batches_count = 0
    total_batches = len(all_batches_fifo_ordered)

    for batch_info in all_batches_fifo_ordered:
        batch_id = batch_info['id']
        # product_name = batch_info['product'] # Не используется напрямую в логике, но есть в batch_info
        
        # Задачи должны быть отсортированы по stage_index при создании all_batches
        # Для FIFO это важно, т.к. мы проходим их последовательно для каждой партии
        tasks_for_current_batch = sorted(batch_info['tasks'], key=lambda t: t['stage_index'])
        
        current_batch_previous_stage_end_time = 0

        for task_info in tasks_for_current_batch:
            stage_name = task_info['stage_name']
            duration = task_info['duration']
            # stage_index = task_info['stage_index'] # Для информации

            if duration == 0:
                # Если этап нулевой, его "окончание" совпадает с окончанием предыдущего реального этапа
                batch_stage_end_times[batch_id][stage_name] = current_batch_previous_stage_end_time
                continue

            num_machines_on_stage = machines_available_fifo.get(stage_name, 1)
            if stage_name not in machine_free_time: # На всякий случай, хотя инициализировали выше
                 machine_free_time[stage_name] = [0] * num_machines_on_stage
            
            # Задача не может начаться раньше, чем закончится предыдущий этап этой же партии
            earliest_start_due_to_precedence = current_batch_previous_stage_end_time
            
            # Найти машину, которая освободится раньше всех *и* не раньше, чем earliest_start_due_to_precedence
            best_finish_time_for_task = float('inf')
            chosen_machine_index = -1
            actual_start_time_for_task = -1

            # Ищем машину, которая позволит завершить задачу раньше всего
            for m_idx in range(num_machines_on_stage):
                # Время, когда задача МОГЛА БЫ начаться на этой машине
                potential_start_on_this_machine = max(machine_free_time[stage_name][m_idx], earliest_start_due_to_precedence)
                potential_finish_on_this_machine = potential_start_on_this_machine + duration

                if potential_finish_on_this_machine < best_finish_time_for_task:
                    best_finish_time_for_task = potential_finish_on_this_machine
                    actual_start_time_for_task = potential_start_on_this_machine
                    chosen_machine_index = m_idx
            
            task_start_time = actual_start_time_for_task
            task_end_time = best_finish_time_for_task
            
            # Обновляем время освобождения выбранной машины
            machine_free_time[stage_name][chosen_machine_index] = task_end_time
            
            schedule_fifo.append({
                "Batch_ID": batch_id,
                "Stage": stage_name,
                "Start_Time_Min": task_start_time,
                "End_Time_Min": task_end_time,
                "Duration_Min": duration,
                "Machine_ID": chosen_machine_index 
            })

            current_batch_previous_stage_end_time = task_end_time
            batch_stage_end_times[batch_id][stage_name] = task_end_time

            # Проверка времен ожидания для FIFO
            if critical_pairs_with_limits:
                for (prev_stage_critical, next_stage_critical), limit in critical_pairs_with_limits.items():
                    if stage_name == next_stage_critical: # Если текущий этап - "последующий" в критической паре
                        if prev_stage_critical in batch_stage_end_times[batch_id]:
                            end_of_prev_critical = batch_stage_end_times[batch_id][prev_stage_critical]
                            actual_wait_time = task_start_time - end_of_prev_critical
                            
                            waiting_times_log_fifo.append({
                                "batch_id": batch_id, 
                                "from_stage": prev_stage_critical, 
                                "to_stage": next_stage_critical,
                                "actual_wait": actual_wait_time,
                                "limit": limit
                            })
                            
                            if actual_wait_time > limit:
                                waiting_times_violations_fifo.append({
                                    "batch_id": batch_id, 
                                    "from_stage": prev_stage_critical, 
                                    "to_stage": next_stage_critical,
                                    "actual_wait": actual_wait_time,
                                    "limit": limit,
                                    "violation_amount": actual_wait_time - limit
                                })
                        # else: (предыдущий критический этап мог быть нулевым или не существовать для этой партии - не логируем)
                        break # Проверили одну пару, выходим из внутреннего цикла по парам

        processed_batches_count += 1
        if processed_batches_count % (max(1, total_batches // 10)) == 0 or processed_batches_count == total_batches:
            print(f"FIFO: Обработано партий: {processed_batches_count}/{total_batches}")

    makespan_fifo = 0
    if schedule_fifo:
        makespan_fifo = max(task['End_Time_Min'] for task in schedule_fifo) if schedule_fifo else 0

    print(f"--- FIFO Симуляция Завершена ---")
    print(f"FIFO Makespan: {makespan_fifo} минут")
    if waiting_times_violations_fifo:
       print(f"FIFO: Обнаружено нарушений времени ожидания: {len(waiting_times_violations_fifo)}")
       # for viol in waiting_times_violations_fifo[:min(3, len(waiting_times_violations_fifo))]: # Показать первые несколько
       #     print(f"  - Batch: {viol['batch_id']}, {viol['from_stage']}->{viol['to_stage']}, Wait: {viol['actual_wait']}, Limit: {viol['limit']}")
    else:
        print("FIFO: Нарушений времени ожидания не обнаружено (согласно заданным лимитам).")
           
    return schedule_fifo, makespan_fifo, waiting_times_log_fifo, waiting_times_violations_fifo

# --- END OF FILE fifo_simulator.py ---