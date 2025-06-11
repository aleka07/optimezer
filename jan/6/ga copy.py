import math
import random
import csv
import datetime
import collections
from deap import base, creator, tools, algorithms
import numpy as np
import os

# 1. ИСХОДНЫЕ ДАННЫЕ (Синхронизированы с time_min.py)
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
    "Формовой": 2047,             # Формовой хлеб 600гр (2270) + упаковка (177)
    "Мини формовой": 387,          # Формовой мини хлеб 300гр (298) + упаковка (89)
    "Бородинский": 595,            # Бородинский хлеб 300гр (427) + упаковка (168)
    "Домашний": 10,                # Домашний хлеб 600гр в упаковке
    "Багет луковый": 56,           # Багет Луковый 300гр в упаковке
    "Багет новый": 312,            # Багет Новый 300гр в упаковке 
    "Багет отрубной": 72,          # Багет Отрубной 300гр в упаковке 
    "Премиум": 11,                 # Багет Премиум 350гр в упаковке
    "Батон Верный": 42,            # Батон Верный 400гр в упаковке
    "Батон Нарезной": 401,         # Батон Нарезной 400гр (317) + упаковка (84)
    "Береке": 181,                 # Береке хлеб 420гр (111) + упаковка (70)
    "Жайлы": 190,                  # Жайлы хлеб 600гр (125) + упаковка (65)
    "Диета": 376,                  # Диетический хлеб (259 без упаковки + 117 в упаковке)
    "Здоровье": 13,                # Здоровье хлеб (5 без упаковки + 8 в упаковке)
    "Любимый": 752,                # Любимый хлеб 500гр (872) + упаковка (80)
    "Немецкий хлеб": 10,           # Немецкий хлеб 250гр в упаковке
    "Отрубной (общий)": 294,       # Отрубной хлеб (216 без упаковки + 78 в упаковке)
    "Плетенка": 152,               # Плетенка (все виды: 87+21+25+19)
    "Семейный": 390,               # Семейный хлеб 600гр (282) + упаковка (108)
    "Славянский": 8,               # Славянский хлеб 600гр в упаковке
    "Зерновой Столичный": 20,      # Столичный хлеб 450гр в упаковке
    "Сэндвич": 3615,               # Все виды сэндвичей (113+53+53+3389+7)
    "Хлеб «Тартин бездрожжевой»": 14,  # Тартин
    "Хлеб «Зерновой»": 173,        # Зерновой хлеб (160 без упаковки + 13 в упаковке)
    "Чиабатта": 21,                # Хлеб Чиабатта шт
    "Булочка для гамбургера большой с кунжутом": 1320,  # Булочка для гамбургера
    "Булочка для хотдога штучно": 585,  # Булочка для хотдога  
    "Датский": 31,                 # Датский хлеб 500гр в упаковке
    "Баварский Деревенский Ржаной": 12  # Деревенский хлеб 500гр
}

machines_available = {
    "Комбинирование": 2, "Смешивание": 3, "Формовка": 2, "Расстойка": 8, "Выпекание": 6, "Остывание": 50
}

# --- Параметры ---
BATCH_SIZE = 100
STAGES = ["Комбинирование", "Смешивание", "Формовка", "Расстойка", "Выпекание", "Остывание"]
proportional_time_stages = ["Комбинирование", "Формовка"]

MAX_WAIT_TIMES = {
    ("Комбинирование", "Смешивание"): 1,
    ("Смешивание", "Формовка"): 1,
    ("Формовка", "Расстойка"): 5,
    ("Расстойка", "Выпекание"): 5,
}
WAIT_TIME_PENALTY = 1000 # Штраф за каждую минуту сверх лимита

# --- Имена выходных файлов ---
script_dir = os.path.dirname(os.path.abspath(__file__))
OUTPUT_CSV_FILE = os.path.join(script_dir, 'ga_production_schedule1.csv')
OUTPUT_TXT_FILE = os.path.join(script_dir, 'ga_production_summary1.txt')

# --- Параметры генетического алгоритма ---
GA_POPULATION_SIZE = 100
GA_GENERATIONS = 250 # Можно увеличить для лучшего поиска
GA_CROSSOVER_PROB = 0.8
GA_MUTATION_PROB = 0.2
GA_TOURNAMENT_SIZE = 5

# 2. Helper Functions & Preprocessing
def time_str_to_minutes_int(time_str):
    try:
        parts = list(map(int, time_str.split(':')))
        if len(parts) == 3: h, m, s = parts; return round(h * 60 + m + s / 60.0)
        elif len(parts) == 2: m, s = parts; return round(m + s / 60.0)
        else: return 0
    except: return 0

tech_map_minutes_int = {p: {s: time_str_to_minutes_int(d.get(s, "0:0:0")) for s in STAGES} for p, d in tech_map_data.items()}

all_batches = []
for product, quantity in orders.items():
    if quantity <= 0 or product not in tech_map_minutes_int: continue
    
    num_batches = math.ceil(quantity / BATCH_SIZE)
    for i in range(num_batches):
        batch_id = f"{product}_batch_{i+1}"
        is_partial = (i == num_batches - 1) and (quantity % BATCH_SIZE != 0)
        batch_size = quantity % BATCH_SIZE if is_partial else BATCH_SIZE
        
        tasks = []
        for stage_idx, stage_name in enumerate(STAGES):
            base_duration = tech_map_minutes_int[product][stage_name]
            if base_duration > 0:
                duration = math.ceil(base_duration * (batch_size / BATCH_SIZE)) if (is_partial and stage_name in proportional_time_stages) else base_duration
                if duration <= 0: duration = 1
                tasks.append({"stage_name": stage_name, "duration": duration})
        all_batches.append({"id": batch_id, "tasks": tasks})

print(f"Всего партий сгенерировано: {len(all_batches)}")
num_tasks_total = sum(len(b['tasks']) for b in all_batches)
print(f"Всего задач (операций): {num_tasks_total}")

# 4. Genetic Algorithm Implementation (Corrected Architecture)
class ProductionScheduler:
    def __init__(self, batches):
        self.batches = batches
        self.num_batches = len(batches)

    def evaluate(self, individual):
        """
        Оценивает индивида, который является перестановкой ИНДЕКСОВ партий.
        Гарантирует правильную последовательность этапов внутри каждой партии.
        """
        # Время освобождения каждой машины на каждом этапе
        machine_free_times = {stage: [0] * count for stage, count in machines_available.items()}
        # Время окончания каждого этапа для каждой партии
        batch_stage_end_times = collections.defaultdict(dict)
        
        makespan = 0
        total_penalty = 0

        # individual - это последовательность индексов партий, например [3, 0, 1, 2, ...]
        for batch_idx in individual:
            batch = self.batches[batch_idx]
            previous_stage_end_time = 0
            
            for task_idx, task in enumerate(batch['tasks']):
                stage = task['stage_name']
                duration = task['duration']
                
                # Находим самую раннюю доступную машину на этом этапе
                earliest_machine_time = min(machine_free_times[stage])
                
                # Задача не может начаться раньше, чем закончится предыдущая задача ЭТОЙ ЖЕ партии
                # и раньше, чем освободится какая-либо машина на ЭТОМ этапе.
                start_time = max(previous_stage_end_time, earliest_machine_time)

                # Проверка и наложение штрафа за время ожидания
                if task_idx > 0:
                    prev_stage = batch['tasks'][task_idx-1]['stage_name']
                    wait_key = (prev_stage, stage)
                    max_wait = MAX_WAIT_TIMES.get(wait_key)
                    if max_wait is not None:
                        wait_time = start_time - previous_stage_end_time
                        if wait_time > max_wait:
                            total_penalty += (wait_time - max_wait) * WAIT_TIME_PENALTY
                
                end_time = start_time + duration
                
                # Обновляем время освобождения использованной машины
                machine_idx = machine_free_times[stage].index(earliest_machine_time)
                machine_free_times[stage][machine_idx] = end_time
                
                # Сохраняем время окончания этого этапа для следующего этапа этой же партии
                previous_stage_end_time = end_time
                
                # Обновляем makespan
                if end_time > makespan:
                    makespan = end_time

        return (makespan + total_penalty,)

    def create_individual(self):
        # Индивид - это перестановка индексов партий
        return random.sample(range(self.num_batches), self.num_batches)

# 5. Setup DEAP
scheduler = ProductionScheduler(all_batches)

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
# Атрибут - это функция, которая создает перестановку индексов партий
toolbox.register("indices", scheduler.create_individual)
# Структура - это индивид, инициализированный с помощью permutation
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", scheduler.evaluate)
toolbox.register("mate", tools.cxOrdered)
toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=GA_TOURNAMENT_SIZE)

# 6. Run Genetic Algorithm
def run_ga():
    print("\n--- Запуск генетического алгоритма (правильная архитектура) ---")
    population = toolbox.population(n=GA_POPULATION_SIZE)
    hall_of_fame = tools.HallOfFame(1)
    stats = tools.Statistics(key=lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    stats.register("max", np.max)

    algorithms.eaSimple(population, toolbox, cxpb=GA_CROSSOVER_PROB, mutpb=GA_MUTATION_PROB,
                        ngen=GA_GENERATIONS, stats=stats, halloffame=hall_of_fame, verbose=True)

    return hall_of_fame[0]

# 7. Execute GA and Process Results
best_individual = run_ga()
best_fitness = best_individual.fitness.values[0]

print(f"\n--- Лучшее найденное решение ---")

# 8. Recreate the best schedule to get detailed data and actual makespan
def create_detailed_schedule(individual):
    schedule_data = []
    machine_free_times = {stage: [0] * count for stage, count in machines_available.items()}
    makespan = 0

    for batch_idx in individual:
        batch = all_batches[batch_idx]
        previous_stage_end_time = 0
        for task in batch['tasks']:
            stage = task['stage_name']
            duration = task['duration']
            
            earliest_machine_time = min(machine_free_times[stage])
            start_time = max(previous_stage_end_time, earliest_machine_time)
            end_time = start_time + duration
            
            machine_idx = machine_free_times[stage].index(earliest_machine_time)
            machine_free_times[stage][machine_idx] = end_time
            
            previous_stage_end_time = end_time
            if end_time > makespan:
                makespan = end_time
            
            schedule_data.append({
                "Batch_ID": batch['id'], "Stage": stage,
                "Start_Time_Min": start_time, "End_Time_Min": end_time,
                "Duration_Min": duration
            })
    return schedule_data, makespan

schedule_output, actual_makespan = create_detailed_schedule(best_individual)

print(f"Фитнес (makespan + штрафы): {best_fitness:.2f} минут")
print(f"Фактический makespan: {actual_makespan:.2f} минут")
tdelta = datetime.timedelta(minutes=actual_makespan)
days = tdelta.days
hours, remainder = divmod(tdelta.seconds, 3600)
minutes, _ = divmod(remainder, 60)
makespan_formatted = f"{days*24 + hours:02}:{minutes:02}"
print(f"Время производства (ЧЧ:ММ): {makespan_formatted}")


# 9. Export Results
schedule_output.sort(key=lambda x: (x['Start_Time_Min'], x['Batch_ID']))
try:
    with open(OUTPUT_CSV_FILE, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["Batch_ID", "Stage", "Start_Time_Min", "End_Time_Min", "Duration_Min"])
        writer.writeheader()
        writer.writerows(schedule_output)
    print(f"\nРасписание записано в CSV файл: '{OUTPUT_CSV_FILE}'")
except Exception as e:
    print(f"\nОшибка записи CSV файла: {e}")

try:
    with open(OUTPUT_TXT_FILE, 'w', encoding='utf-8') as f:
        f.write("--- Сводка по Производственному Расписанию (Генетический Алгоритм) ---\n\n")
        f.write(f"Алгоритм: Генетический алгоритм (DEAP) - Правильная архитектура\n")
        f.write(f"Время производства (Makespan): {actual_makespan:.2f} минут ({makespan_formatted})\n")
        f.write(f"Всего партий: {len(all_batches)}\n")
        f.write(f"Всего задач: {num_tasks_total}\n\n")
        f.write(f"Параметры GA:\n")
        f.write(f"  - Размер популяции: {GA_POPULATION_SIZE}\n")
        f.write(f"  - Количество поколений: {GA_GENERATIONS}\n")
        f.write(f"  - Вероятность скрещивания: {GA_CROSSOVER_PROB}\n")
        f.write(f"  - Вероятность мутации: {GA_MUTATION_PROB}\n\n")
        f.write(f"Параметры ограничений (макс. время ожидания):\n")
        for (s1, s2), t in MAX_WAIT_TIMES.items():
            f.write(f"  - {s1} -> {s2}: {t} мин\n")
        f.write(f"\nДоступные ресурсы (машины):\n")
        for stage, count in machines_available.items():
            f.write(f"  - {stage}: {count}\n")
        f.write(f"\nФайл с детальным расписанием: {OUTPUT_CSV_FILE}\n")
    print(f"Сводная информация записана в TXT файл: '{OUTPUT_TXT_FILE}'")
except Exception as e:
    print(f"\nОшибка записи TXT файла: {e}")

print(f"\n--- Генетический алгоритм завершен ---")