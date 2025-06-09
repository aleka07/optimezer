import math
import random
import csv
import datetime
import collections
from deap import base, creator, tools, algorithms
import numpy as np
import os

# 1. Исходные данные (без изменений)
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
    "Немецкий хлеб":        {"Комбинирование": "0:21:00", "Смешивание": "0:10:30", "Формовка": "0:11:00", "Расстойка": "0:25:00", "Выпекание": "0:25:00", "Остывание": "1:15:00"},
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

# Параметры
BATCH_SIZE = 100
STAGES = ["Комбинирование", "Смешивание", "Формовка", "Расстойка", "Выпекание", "Остывание"]

# Ограничения времени ожидания
MAX_WAIT_COMBINING_MIXING_MIN = 1
MAX_WAIT_MIXING_FORMING_MIN = 1
MAX_WAIT_FORMING_PROOFING_MIN = 5
MAX_WAIT_PROOFING_BAKING_MIN = 5

# Выходные файлы
script_dir = os.path.dirname(os.path.abspath(__file__))
OUTPUT_CSV_FILE = os.path.join(script_dir, 'ga_production_schedule_v2.csv')
OUTPUT_TXT_FILE = os.path.join(script_dir, 'ga_production_summary_v2.txt')

# Параметры генетического алгоритма
GA_POPULATION_SIZE = 100
GA_GENERATIONS = 200
GA_CROSSOVER_PROB = 0.8
GA_MUTATION_PROB = 0.2
GA_TOURNAMENT_SIZE = 5

# 2. Helper Functions
def time_str_to_minutes_int(time_str):
    try:
        parts = list(map(int, time_str.split(':')))
        if len(parts) == 3: h, m, s = parts; return round(h * 60 + m + s / 60.0)
        elif len(parts) == 2: m, s = parts; return round(m + s / 60.0)
        return 0
    except: return 0

# 3. Data Preprocessing
tech_map_minutes_int = {}
for product, stages_data in tech_map_data.items():
    tech_map_minutes_int[product] = {}
    for stage_name in STAGES:
        time_str = stages_data.get(stage_name, "0:00:00")
        tech_map_minutes_int[product][stage_name] = time_str_to_minutes_int(time_str)

all_batches = []
for product, quantity_ordered in orders.items():
    if quantity_ordered <= 0: continue
    num_batches = math.ceil(quantity_ordered / BATCH_SIZE)
    for i in range(num_batches):
        batch_id = f"{product}_batch_{i+1}"
        batch_tasks = []
        for stage_index, stage_name in enumerate(STAGES):
            duration = tech_map_minutes_int[product][stage_name]
            if duration > 0:
                batch_tasks.append({
                    "stage_index": stage_index, "stage_name": stage_name,
                    "duration": duration, "product": product
                })
        if batch_tasks:
            all_batches.append({"id": batch_id, "product": product, "tasks": batch_tasks})

print(f"Всего партий сгенерировано: {len(all_batches)}")

# 4. Genetic Algorithm Implementation (ПОЛНОСТЬЮ ПЕРЕРАБОТАНО)
class ProductionScheduler:
    def __init__(self, batches):
        self.batches = batches
        self.num_batches = len(batches)
        self.stage_names = STAGES
        print(f"ГА будет работать с {self.num_batches} партиями.")

    def evaluate_schedule(self, batch_sequence):
        """
        Оценивает последовательность ПАРТИЙ.
        Индивид (batch_sequence) - это перестановка индексов партий, например [2, 0, 1].
        """
        # Расписание освобождения каждой машины на каждом этапе
        machine_end_times = {
            stage: [0] * machines_available[stage] for stage in self.stage_names
        }
        # Время окончания каждого этапа для каждой партии
        batch_end_times = {} # (batch_id, stage_name) -> end_time

        # Обрабатываем партии в порядке, заданном индивидом
        for batch_idx in batch_sequence:
            batch = self.batches[batch_idx]
            batch_id = batch['id']
            last_stage_end_time = 0

            # Планируем все задачи для этой партии последовательно
            for task in batch['tasks']:
                stage_name = task['stage_name']
                duration = task['duration']

                # 1. Зависимость от предыдущего этапа той же партии
                dependency_ready_time = last_stage_end_time

                # 2. Доступность машины на текущем этапе
                # Находим машину, которая освободится раньше всех
                available_machine_idx = np.argmin(machine_end_times[stage_name])
                machine_ready_time = machine_end_times[stage_name][available_machine_idx]

                # Задача может начаться не раньше, чем выполнен предыдущий этап И освободилась машина
                start_time = max(dependency_ready_time, machine_ready_time)
                end_time = start_time + duration

                # Обновляем данные
                machine_end_times[stage_name][available_machine_idx] = end_time
                batch_end_times[(batch_id, stage_name)] = (start_time, end_time)
                last_stage_end_time = end_time

        # Вычисляем makespan (общее время)
        makespan = 0
        for stage_machines in machine_end_times.values():
            if stage_machines:
                makespan = max(makespan, max(stage_machines))

        # Вычисляем штрафы за нарушение времени ожидания
        penalty = 0
        for batch in self.batches:
            batch_id = batch['id']
            for i in range(len(batch['tasks']) - 1):
                curr_task = batch['tasks'][i]
                next_task = batch['tasks'][i+1]
                
                curr_stage = curr_task['stage_name']
                next_stage = next_task['stage_name']

                curr_end_time = batch_end_times.get((batch_id, curr_stage), (0,0))[1]
                next_start_time = batch_end_times.get((batch_id, next_stage), (0,0))[0]
                
                wait_time = next_start_time - curr_end_time
                
                max_wait = 0
                if curr_stage == "Комбинирование" and next_stage == "Смешивание": max_wait = MAX_WAIT_COMBINING_MIXING_MIN
                elif curr_stage == "Смешивание" and next_stage == "Формовка": max_wait = MAX_WAIT_MIXING_FORMING_MIN
                elif curr_stage == "Формовка" and next_stage == "Расстойка": max_wait = MAX_WAIT_FORMING_PROOFING_MIN
                elif curr_stage == "Расстойка" and next_stage == "Выпекание": max_wait = MAX_WAIT_PROOFING_BAKING_MIN
                
                if max_wait > 0 and wait_time > max_wait:
                    penalty += (wait_time - max_wait) * 10  # Штраф за каждую минуту простоя

        return (makespan + penalty,)

    def create_individual(self):
        """Создает случайную последовательность ИНДЕКСОВ ПАРТИЙ."""
        return random.sample(range(self.num_batches), self.num_batches)

# 5. Setup DEAP
scheduler = ProductionScheduler(all_batches)

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
# <<< КЛЮЧЕВОЕ ИЗМЕНЕНИЕ: Индивид теперь создается на основе количества партий >>>
toolbox.register("individual", tools.initIterate, creator.Individual, scheduler.create_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", scheduler.evaluate_schedule)
# Операторы скрещивания и мутации для перестановок подходят идеально, их менять не нужно
toolbox.register("mate", tools.cxOrdered)
toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=GA_TOURNAMENT_SIZE)

# 6. Run Genetic Algorithm (без изменений в логике, но теперь работает с правильными данными)
def run_ga():
    print("\n--- Запуск генетического алгоритма (новая версия) ---")
    population = toolbox.population(n=GA_POPULATION_SIZE)
    hall_of_fame = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    stats.register("max", np.max)

    algorithms.eaSimple(population, toolbox,
                        cxpb=GA_CROSSOVER_PROB,
                        mutpb=GA_MUTATION_PROB,
                        ngen=GA_GENERATIONS,
                        stats=stats,
                        halloffame=hall_of_fame,
                        verbose=True)

    return hall_of_fame[0]

# 7. Execute GA and Process Results
best_individual = run_ga()

print(f"\n--- Лучшее найденное решение ---")
best_fitness = best_individual.fitness.values[0]
print(f"Фитнес (makespan + штрафы): {best_fitness:.2f} минут")

# <<< ИЗМЕНЕНО: Функция для создания детального расписания на основе ПОСЛЕДОВАТЕЛЬНОСТИ ПАРТИЙ >>>
def create_detailed_schedule(batch_sequence):
    machine_end_times = {stage: [0] * machines_available[stage] for stage in STAGES}
    detailed_schedule = []
    
    for batch_idx in batch_sequence:
        batch = all_batches[batch_idx]
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
            last_stage_end_time = end_time
            
            detailed_schedule.append({
                "Batch_ID": batch_id, "Stage": stage_name,
                "Start_Time_Min": start_time, "End_Time_Min": end_time,
                "Duration_Min": duration
            })
            
    return detailed_schedule

schedule_data_for_output = create_detailed_schedule(best_individual)
actual_makespan = max(task['End_Time_Min'] for task in schedule_data_for_output)

print(f"Фактический makespan: {actual_makespan:.2f} минут")
total_seconds = int(actual_makespan * 60)
tdelta = datetime.timedelta(seconds=total_seconds)
days = tdelta.days
hours, remainder = divmod(tdelta.seconds, 3600)
minutes, seconds = divmod(remainder, 60)
makespan_formatted = ""
if days > 0: makespan_formatted += f"{days} дн "
makespan_formatted += f"{hours:02}:{minutes:02}:{seconds:02}"
print(f"Время производства: {makespan_formatted}")


# 8. Export Results
schedule_data_for_output.sort(key=lambda x: x['Start_Time_Min'])

try:
    csv_fieldnames = ["Batch_ID", "Stage", "Start_Time_Min", "End_Time_Min", "Duration_Min"]
    with open(OUTPUT_CSV_FILE, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_fieldnames)
        writer.writeheader()
        writer.writerows(schedule_data_for_output)
    print(f"\nРасписание записано в CSV файл: '{OUTPUT_CSV_FILE}'")
except Exception as e:
    print(f"\nОшибка записи CSV файла: {e}")

try:
    with open(OUTPUT_TXT_FILE, 'w', encoding='utf-8') as txtfile:
        txtfile.write("--- Сводка по Производственному Расписанию (Генетический Алгоритм v2) ---\n\n")
        txtfile.write(f"Подход: Планирование последовательности партий (не отдельных задач)\n")
        txtfile.write(f"Время производства (Makespan): {actual_makespan:.2f} минут\n")
        txtfile.write(f"Время производства (формат): {makespan_formatted}\n")
        txtfile.write(f"Итоговый фитнес (с учетом штрафов): {best_fitness:.2f}\n")
        txtfile.write(f"Всего партий: {len(all_batches)}\n")
        txtfile.write(f"Всего задач: {len(schedule_data_for_output)}\n\n")
        txtfile.write(f"Параметры GA:\n")
        txtfile.write(f"  - Размер популяции: {GA_POPULATION_SIZE}\n")
        txtfile.write(f"  - Количество поколений: {GA_GENERATIONS}\n\n")
        txtfile.write(f"Доступные ресурсы (машины):\n")
        for stage, count in machines_available.items():
            txtfile.write(f"  - {stage}: {count}\n")
        txtfile.write(f"\nФайл с детальным расписанием: {OUTPUT_CSV_FILE}\n")
    print(f"Сводная информация записана в TXT файл: '{OUTPUT_TXT_FILE}'")
except Exception as e:
    print(f"\nОшибка записи TXT файла: {e}")