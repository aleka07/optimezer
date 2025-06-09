import math
import random
import csv
import datetime
import collections
from deap import base, creator, tools, algorithms
import numpy as np
import os
import multiprocessing # <<< ДОБАВЛЕН ИМПОРТ
import sys

# 1. Исходные данные (без изменений)
tech_map_data = {
    # ... (данные остаются без изменений) ...
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
    # ... (данные остаются без изменений) ...
    "Формовой": 1910, "Мини формовой": 306, "Бородинский": 488, "Домашний": 17,
    "Багет луковый": 33, "Багет новый": 219, "Багет отрубной": 49, "Премиум": 20,
    "Батон Верный": 54, "Батон Нарезной": 336, "Береке": 109, "Жайлы": 131,
    "Диета": 210, "Здоровье": 30, "Любимый": 459, "Немецкий хлеб": 15,
    "Отрубной (общий)": 161, "Плетенка": 94, "Семейный": 212, "Славянский": 6,
    "Зерновой Столичный": 16, "Сэндвич": 1866, "Хлеб «Тартин бездрожжевой»": 18,
    "Хлеб «Зерновой»": 113, "Чиабатта": 18, "Булочка для гамбургера большой с кунжутом": 160
}

machines_available = { "Комбинирование": 2, "Смешивание": 3, "Формовка": 2, "Расстойка": 8, "Выпекание": 6, "Остывание": 25 }

# Параметры
BATCH_SIZE = 100
STAGES = ["Комбинирование", "Смешивание", "Формовка", "Расстойка", "Выпекание", "Остывание"]
MAX_WAIT_COMBINING_MIXING_MIN = 1; MAX_WAIT_MIXING_FORMING_MIN = 1; MAX_WAIT_FORMING_PROOFING_MIN = 5; MAX_WAIT_PROOFING_BAKING_MIN = 5

# <<< ИЗМЕНЕНЫ ИМЕНА ФАЙЛОВ И ПАРАМЕТРЫ GA >>>
script_dir = os.path.dirname(os.path.abspath(__file__))
OUTPUT_CSV_FILE = os.path.join(script_dir, 'ga_parallel_schedule.csv')
OUTPUT_TXT_FILE = os.path.join(script_dir, 'ga_parallel_summary.txt')
GA_POPULATION_SIZE = 100
GA_GENERATIONS = 1000 # Как вы и просили
GA_CROSSOVER_PROB = 0.8
GA_MUTATION_PROB = 0.2
GA_TOURNAMENT_SIZE = 5

# --- ГЛОБАЛЬНЫЕ ДАННЫЕ ДЛЯ ПРОЦЕССОВ ---
# Эти данные будут доступны "только для чтения" в каждом дочернем процессе
all_batches = []

# --- КЛАССЫ И ФУНКЦИИ (без изменений в логике) ---
def time_str_to_minutes_int(time_str):
    try:
        parts = list(map(int, time_str.split(':')))
        if len(parts) == 3: h, m, s = parts; return round(h * 60 + m + s / 60.0)
        elif len(parts) == 2: m, s = parts; return round(m + s / 60.0)
        return 0
    except: return 0

class ProductionScheduler:
    def __init__(self, batches):
        self.batches = batches
        self.num_batches = len(batches)
        self.stage_names = STAGES
    def evaluate_schedule(self, batch_sequence):
        machine_end_times = {stage: [0] * machines_available[stage] for stage in self.stage_names}
        batch_end_times = {}
        for batch_idx in batch_sequence:
            batch = self.batches[batch_idx]
            batch_id = batch['id']
            last_stage_end_time = 0
            for task in batch['tasks']:
                stage_name = task['stage_name']
                duration = task['duration']
                dependency_ready_time = last_stage_end_time
                available_machine_idx = np.argmin(machine_end_times[stage_name])
                machine_ready_time = machine_end_times[stage_name][available_machine_idx]
                start_time = max(dependency_ready_time, machine_ready_time)
                end_time = start_time + duration
                machine_end_times[stage_name][available_machine_idx] = end_time
                batch_end_times[(batch_id, stage_name)] = (start_time, end_time)
                last_stage_end_time = end_time
        makespan = max(max(times) for times in machine_end_times.values() if times)
        penalty = 0
        for batch in self.batches:
            for i in range(len(batch['tasks']) - 1):
                curr_task, next_task = batch['tasks'][i], batch['tasks'][i+1]
                curr_end = batch_end_times.get((batch['id'], curr_task['stage_name']), (0,0))[1]
                next_start = batch_end_times.get((batch['id'], next_task['stage_name']), (0,0))[0]
                wait_time = next_start - curr_end
                max_wait = 0
                if curr_task['stage_name'] == "Комбинирование" and next_task['stage_name'] == "Смешивание": max_wait = MAX_WAIT_COMBINING_MIXING_MIN
                elif curr_task['stage_name'] == "Смешивание" and next_task['stage_name'] == "Формовка": max_wait = MAX_WAIT_MIXING_FORMING_MIN
                elif curr_task['stage_name'] == "Формовка" and next_task['stage_name'] == "Расстойка": max_wait = MAX_WAIT_FORMING_PROOFING_MIN
                elif curr_task['stage_name'] == "Расстойка" and next_task['stage_name'] == "Выпекание": max_wait = MAX_WAIT_PROOFING_BAKING_MIN
                if max_wait > 0 and wait_time > max_wait: penalty += (wait_time - max_wait) * 10
        return (makespan + penalty,)
    def create_individual(self):
        return random.sample(range(self.num_batches), self.num_batches)

# <<< НОВАЯ ФУНКЦИЯ: ОДИН ПОЛНЫЙ ЗАПУСК GA ДЛЯ ОДНОГО ПРОЦЕССА >>>
def run_single_ga_instance(worker_id):
    """
    Эта функция будет выполняться в отдельном процессе.
    Она настраивает свой собственный экземпляр DEAP и запускает ГА.
    """
    print(f"Процесс {worker_id}: Запуск эволюции...")
    
    # Важно: каждый процесс должен создать свои собственные типы и toolbox.
    # Их нельзя передавать из главного процесса.
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)
    
    scheduler = ProductionScheduler(all_batches)
    
    toolbox = base.Toolbox()
    toolbox.register("individual", tools.initIterate, creator.Individual, scheduler.create_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", scheduler.evaluate_schedule)
    toolbox.register("mate", tools.cxOrdered)
    toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.05)
    toolbox.register("select", tools.selTournament, tournsize=GA_TOURNAMENT_SIZE)
    
    population = toolbox.population(n=GA_POPULATION_SIZE)
    hall_of_fame = tools.HallOfFame(1)
    
    # Запускаем алгоритм без вывода статистики в консоль для каждого процесса (чтобы не было мешанины)
    algorithms.eaSimple(population, toolbox,
                        cxpb=GA_CROSSOVER_PROB,
                        mutpb=GA_MUTATION_PROB,
                        ngen=GA_GENERATIONS,
                        stats=None, # Отключаем логирование
                        halloffame=hall_of_fame,
                        verbose=False) # Отключаем вывод

    best_ind = hall_of_fame[0]
    best_fitness = best_ind.fitness.values[0]
    
    print(f"Процесс {worker_id}: Эволюция завершена. Лучший фитнес: {best_fitness:.2f}")
    
    # Возвращаем лучший индивид, его фитнес и ID процесса
    return (best_ind, best_fitness, worker_id)

# <<< ОСНОВНОЙ БЛОК ПРОГРАММЫ >>>
if __name__ == '__main__':
    # 1. Подготовка данных (выполняется один раз в главном процессе)
    tech_map_minutes_int = {}
    for product, stages_data in tech_map_data.items():
        tech_map_minutes_int[product] = {}
        for stage_name in STAGES:
            time_str = stages_data.get(stage_name, "0:00:00")
            tech_map_minutes_int[product][stage_name] = time_str_to_minutes_int(time_str)

    for product, quantity_ordered in orders.items():
        if quantity_ordered <= 0: continue
        if product not in tech_map_minutes_int:
            print(f"Предупреждение: Продукт '{product}' из заказа отсутствует в 'tech_map_data'. Пропускается.")
            continue
        num_batches = math.ceil(quantity_ordered / BATCH_SIZE)
        for i in range(num_batches):
            batch_id = f"{product}_batch_{i+1}"
            batch_tasks = []
            for stage_index, stage_name in enumerate(STAGES):
                duration = tech_map_minutes_int.get(product, {}).get(stage_name, 0)
                if duration > 0:
                    batch_tasks.append({"stage_index": stage_index, "stage_name": stage_name, "duration": duration, "product": product})
            if batch_tasks:
                all_batches.append({"id": batch_id, "product": product, "tasks": batch_tasks})

    print(f"Всего партий сгенерировано для обработки: {len(all_batches)}")

    # <<< НАЧАЛО ИСПРАВЛЕНИЯ: ПРОВЕРКА КОЛИЧЕСТВА ПАРТИЙ >>>
    def create_detailed_schedule(batch_sequence):
        machine_end_times = {stage: [0] * machines_available[stage] for stage in STAGES}
        detailed_schedule = []
        for batch_idx in batch_sequence:
            batch = all_batches[batch_idx]
            last_stage_end_time = 0
            for task in batch['tasks']:
                stage_name = task['stage_name']; duration = task['duration']
                dependency_ready_time = last_stage_end_time
                available_machine_idx = np.argmin(machine_end_times[stage_name])
                machine_ready_time = machine_end_times[stage_name][available_machine_idx]
                start_time = max(dependency_ready_time, machine_ready_time)
                end_time = start_time + duration
                machine_end_times[stage_name][available_machine_idx] = end_time
                last_stage_end_time = end_time
                detailed_schedule.append({"Batch_ID": batch['id'], "Stage": stage_name, "Start_Time_Min": start_time, "End_Time_Min": end_time, "Duration_Min": duration})
        return detailed_schedule

    if len(all_batches) < 2:
        print("\nОшибка: Сгенерировано меньше двух партий для обработки.")
        print("Генетический алгоритм не может работать (оператор скрещивания требует минимум 2 элемента).")
        print("Возможные причины:")
        print("  - В заказах (orders) слишком мало позиций.")
        print("  - Большинство заказов для продуктов, отсутствующих в тех. карте (tech_map_data).")
        print("\nПрограмма будет завершена.")
        if len(all_batches) == 1:
            print("Создание простого последовательного расписания для единственной партии...")
            schedule_data_for_output = create_detailed_schedule([0])
            actual_makespan = max(task['End_Time_Min'] for task in schedule_data_for_output)
            tdelta = datetime.timedelta(seconds=int(actual_makespan * 60))
            print(f"Итоговый makespan: {actual_makespan:.2f} минут ({tdelta})")
            schedule_data_for_output.sort(key=lambda x: x['Start_Time_Min'])
            try:
                csv_fieldnames = ["Batch_ID", "Stage", "Start_Time_Min", "End_Time_Min", "Duration_Min"]
                with open(OUTPUT_CSV_FILE, 'w', newline='', encoding='utf-8') as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=csv_fieldnames)
                    writer.writeheader(); writer.writerows(schedule_data_for_output)
                print(f"Расписание для одной партии записано в CSV файл: '{OUTPUT_CSV_FILE}'")
            except Exception as e:
                print(f"\nОшибка записи CSV файла: {e}")
        sys.exit(0)
    # <<< КОНЕЦ ИСПРАВЛЕНИЯ >>>

    # 2. Параллельный запуск ГА
    num_processes = multiprocessing.cpu_count()
    print(f"\n--- Запуск {num_processes} параллельных экземпляров ГА ---")
    print(f"Каждый экземпляр будет работать {GA_GENERATIONS} поколений.")
    
    with multiprocessing.Pool(processes=num_processes) as pool:
        # pool.map запускает функцию run_single_ga_instance для каждого элемента в range(num_processes)
        # и собирает результаты в список
        results = pool.map(run_single_ga_instance, range(num_processes))
        
    print("\n--- Все процессы завершили работу. Поиск лучшего решения... ---")

    # 3. Выбор лучшего результата из всех процессов
    best_overall_result = min(results, key=lambda x: x[1]) # x[1] - это фитнес
    
    best_individual = best_overall_result[0]
    best_fitness = best_overall_result[1]
    winning_worker_id = best_overall_result[2]

    print(f"Победитель: Процесс {winning_worker_id} с итоговым фитнесом {best_fitness:.2f}")
    
    # 4. Создание и сохранение итогового расписания
    schedule_data_for_output = create_detailed_schedule(best_individual)
    actual_makespan = max(task['End_Time_Min'] for task in schedule_data_for_output)
    
    print(f"\nИтоговый makespan: {actual_makespan:.2f} минут")
    tdelta = datetime.timedelta(seconds=int(actual_makespan * 60))
    print(f"Время производства: {tdelta}")

    # 5. Запись в файлы (без изменений)
    schedule_data_for_output.sort(key=lambda x: x['Start_Time_Min'])
    try:
        csv_fieldnames = ["Batch_ID", "Stage", "Start_Time_Min", "End_Time_Min", "Duration_Min"]
        with open(OUTPUT_CSV_FILE, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_fieldnames)
            writer.writeheader(); writer.writerows(schedule_data_for_output)
        print(f"\nРасписание записано в CSV файл: '{OUTPUT_CSV_FILE}'")
    except Exception as e:
        print(f"\nОшибка записи CSV файла: {e}")

    try:
        with open(OUTPUT_TXT_FILE, 'w', encoding='utf-8') as txtfile:
            txtfile.write("--- Сводка по Производственному Расписанию (Параллельный ГА) ---\n\n")
            txtfile.write(f"Количество параллельных запусков: {num_processes}\n")
            txtfile.write(f"Поколений в каждом запуске: {GA_GENERATIONS}\n")
            txtfile.write(f"Победитель: Процесс {winning_worker_id}\n\n")
            txtfile.write(f"Время производства (Makespan): {actual_makespan:.2f} минут\n")
            txtfile.write(f"Время производства (формат): {tdelta}\n")
            txtfile.write(f"Итоговый фитнес (с учетом штрафов): {best_fitness:.2f}\n")
        print(f"Сводная информация записана в TXT файл: '{OUTPUT_TXT_FILE}'")
    except Exception as e:
        print(f"\nОшибка записи TXT файла: {e}")