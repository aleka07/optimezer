import math
import random
import csv
import datetime
import collections
from deap import base, creator, tools, algorithms
import numpy as np

# 1. Define Input Data (данные из вашего файла)
# 1. Исходные данные
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
MAX_WAIT_COMBINING_MIXING_MIN = 1
MAX_WAIT_MIXING_FORMING_MIN = 1
MAX_WAIT_FORMING_PROOFING_MIN = 5
MAX_WAIT_PROOFING_BAKING_MIN = 5

# # Выходные файлы
# OUTPUT_CSV_FILE = 'milp_production_schedule.csv'
# OUTPUT_TXT_FILE = 'milp_production_summary.txt'
# --- Имя выходного файла ---
script_dir = os.path.dirname(os.path.abspath(__file__))
OUTPUT_CSV_FILE = os.path.join(script_dir, 'ga_production_schedule.csv')
OUTPUT_TXT_FILE = os.path.join(script_dir, 'ga_production_summary.txt')
# # Выходные файлы
# OUTPUT_CSV_FILE = 'ga_production_schedule.csv'
# OUTPUT_TXT_FILE = 'ga_production_summary.txt'

# Параметры генетического алгоритма
GA_POPULATION_SIZE = 100
GA_GENERATIONS = 200
GA_CROSSOVER_PROB = 0.8
GA_MUTATION_PROB = 0.2
GA_TOURNAMENT_SIZE = 5

# 2. Helper Functions
def time_str_to_minutes_int(time_str):
    """Convert time string to minutes (integer)"""
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

# 3. Data Preprocessing
tech_map_minutes_int = {}
for product, stages_data in tech_map_data.items():
    tech_map_minutes_int[product] = {}
    for stage_name in STAGES:
        time_str = stages_data.get(stage_name, "0:00:00")
        duration = time_str_to_minutes_int(time_str)
        tech_map_minutes_int[product][stage_name] = duration

# Generate all batches
all_batches = []
for product, quantity_ordered in orders.items():
    if quantity_ordered <= 0:
        continue
    num_batches = math.ceil(quantity_ordered / BATCH_SIZE)
    for i in range(num_batches):
        batch_id = f"{product}_batch_{i+1}"
        batch_tasks = []
        for stage_index, stage_name in enumerate(STAGES):
            duration = tech_map_minutes_int[product][stage_name]
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
                "tasks": batch_tasks
            })

print(f"Всего партий сгенерировано: {len(all_batches)}")
print(f"Всего задач: {sum(len(b['tasks']) for b in all_batches)}")

# 4. Genetic Algorithm Implementation
class ProductionScheduler:
    def __init__(self):
        self.all_tasks = []
        self.task_to_batch = {}
        self.batch_stages = collections.defaultdict(list)
        
        # Flatten all tasks
        task_id = 0
        for batch in all_batches:
            for task in batch['tasks']:
                self.all_tasks.append({
                    'id': task_id,
                    'batch_id': task['batch_id'],
                    'stage_name': task['stage_name'],
                    'stage_index': task['stage_index'],
                    'duration': task['duration'],
                    'product': task['product']
                })
                self.task_to_batch[task_id] = batch['id']
                self.batch_stages[batch['id']].append(task_id)
                task_id += 1
        
        self.num_tasks = len(self.all_tasks)
        print(f"Общее количество задач для GA: {self.num_tasks}")

    def evaluate_schedule(self, individual):
        """Evaluate the fitness of a schedule (individual)"""
        # individual представляет собой перестановку задач
        schedule = {}
        machine_schedules = {stage: [] for stage in STAGES}
        
        # Инициализация расписания для каждой задачи
        for task_id in range(self.num_tasks):
            schedule[task_id] = {'start': 0, 'end': 0}
        
        # Обработка задач в порядке, определенном индивидом
        for position, task_id in enumerate(individual):
            task = self.all_tasks[task_id]
            stage_name = task['stage_name']
            duration = task['duration']
            
            # Найти самое раннее время начала для этой задачи
            earliest_start = 0
            
            # Проверить зависимости от предыдущих этапов той же партии
            batch_id = task['batch_id']
            current_stage_index = task['stage_index']
            
            for other_task_id in self.batch_stages[batch_id]:
                other_task = self.all_tasks[other_task_id]
                if (other_task['stage_index'] < current_stage_index and 
                    other_task_id in [ind_task for ind_task in individual[:position+1]]):
                    earliest_start = max(earliest_start, schedule[other_task_id]['end'])
            
            # Найти доступное время на машине
            machine_count = machines_available.get(stage_name, 1)
            machine_schedule = machine_schedules[stage_name]
            
            # Сортировать по времени окончания
            machine_schedule.sort(key=lambda x: x[1])
            
            # Найти первую доступную машину
            start_time = max(earliest_start, 0)
            if len(machine_schedule) >= machine_count:
                # Все машины заняты, найти самое раннее время освобождения
                earliest_free = min(machine_schedule[-machine_count:], key=lambda x: x[1])[1]
                start_time = max(start_time, earliest_free)
            
            end_time = start_time + duration
            
            # Обновить расписание
            schedule[task_id] = {'start': start_time, 'end': end_time}
            machine_schedule.append((start_time, end_time, task_id))
        
        # Вычислить makespan
        makespan = max(schedule[task_id]['end'] for task_id in range(self.num_tasks))
        
        # Вычислить штрафы за нарушение ограничений по времени ожидания
        penalty = 0
        for batch in all_batches:
            batch_id = batch['id']
            batch_task_ids = [tid for tid in range(self.num_tasks) 
                             if self.all_tasks[tid]['batch_id'] == batch_id]
            
            # Сортировать по индексу этапа
            batch_task_ids.sort(key=lambda tid: self.all_tasks[tid]['stage_index'])
            
            # Проверить ограничения по времени ожидания
            for i in range(len(batch_task_ids) - 1):
                curr_task_id = batch_task_ids[i]
                next_task_id = batch_task_ids[i + 1]
                
                curr_stage = self.all_tasks[curr_task_id]['stage_name']
                next_stage = self.all_tasks[next_task_id]['stage_name']
                
                wait_time = schedule[next_task_id]['start'] - schedule[curr_task_id]['end']
                
                max_wait = 0
                if curr_stage == "Комбинирование" and next_stage == "Смешивание":
                    max_wait = MAX_WAIT_COMBINING_MIXING_MIN
                elif curr_stage == "Смешивание" and next_stage == "Формовка":
                    max_wait = MAX_WAIT_MIXING_FORMING_MIN
                elif curr_stage == "Формовка" and next_stage == "Расстойка":
                    max_wait = MAX_WAIT_FORMING_PROOFING_MIN
                elif curr_stage == "Расстойка" and next_stage == "Выпекание":
                    max_wait = MAX_WAIT_PROOFING_BAKING_MIN
                
                if max_wait > 0 and wait_time > max_wait:
                    penalty += (wait_time - max_wait) * 10  # Штраф за превышение
        
        # Фитнес = makespan + штрафы (чем меньше, тем лучше)
        fitness = makespan + penalty
        return (fitness,)

    def create_individual(self):
        """Create a random individual (task sequence)"""
        individual = list(range(self.num_tasks))
        random.shuffle(individual)
        return individual

    def mutate_individual(self, individual):
        """Mutate an individual by swapping two random positions"""
        if len(individual) < 2:
            return individual,
        
        # Выбрать два случайных индекса
        idx1, idx2 = random.sample(range(len(individual)), 2)
        
        # Поменять местами
        individual[idx1], individual[idx2] = individual[idx2], individual[idx1]
        
        return individual,

    def crossover_individuals(self, parent1, parent2):
        """Order crossover (OX) for permutation problems"""
        size = len(parent1)
        if size < 2:
            return creator.Individual(parent1[:]), creator.Individual(parent2[:])
        
        # Выбрать случайный сегмент
        start, end = sorted(random.sample(range(size), 2))
        
        # Создать потомков
        child1 = [-1] * size
        child2 = [-1] * size
        
        # Скопировать сегменты
        child1[start:end] = parent1[start:end]
        child2[start:end] = parent2[start:end]
        
        # Заполнить оставшиеся позиции
        def fill_child(child, parent, other_parent):
            remaining = [x for x in other_parent if x not in child]
            remaining_idx = 0
            for i in range(size):
                if child[i] == -1:
                    child[i] = remaining[remaining_idx]
                    remaining_idx += 1
        
        fill_child(child1, parent1, parent2)
        fill_child(child2, parent2, parent1)
        
        return creator.Individual(child1), creator.Individual(child2)

# 5. Setup DEAP
scheduler = ProductionScheduler()

# Создать типы для DEAP
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("individual", tools.initIterate, creator.Individual, scheduler.create_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", scheduler.evaluate_schedule)
toolbox.register("mate", scheduler.crossover_individuals)
toolbox.register("mutate", scheduler.mutate_individual)
toolbox.register("select", tools.selTournament, tournsize=GA_TOURNAMENT_SIZE)

# 6. Run Genetic Algorithm
def run_ga():
    print("\n--- Запуск генетического алгоритма ---")
    print(f"Размер популяции: {GA_POPULATION_SIZE}")
    print(f"Количество поколений: {GA_GENERATIONS}")
    print(f"Вероятность скрещивания: {GA_CROSSOVER_PROB}")
    print(f"Вероятность мутации: {GA_MUTATION_PROB}")
    
    # Создать начальную популяцию
    population = toolbox.population(n=GA_POPULATION_SIZE)
    
    # Оценить начальную популяцию
    fitnesses = list(map(toolbox.evaluate, population))
    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = fit
    
    # Статистика
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    stats.register("max", np.max)
    
    # Зал славы для лучших решений
    hall_of_fame = tools.HallOfFame(1)
    hall_of_fame.update(population)
    
    print(f"Поколение 0: Лучший фитнес = {hall_of_fame[0].fitness.values[0]:.2f}")
    
    # Основной цикл эволюции
    for generation in range(1, GA_GENERATIONS + 1):
        # Селекция
        offspring = toolbox.select(population, len(population))
        offspring = list(map(toolbox.clone, offspring))
        
        # Скрещивание
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < GA_CROSSOVER_PROB:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values
        
        # Мутация
        for mutant in offspring:
            if random.random() < GA_MUTATION_PROB:
                toolbox.mutate(mutant)
                del mutant.fitness.values
        
        # Оценить потомков с недействительным фитнесом
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        
        # Заменить популяцию
        population[:] = offspring
        
        # Обновить зал славы
        hall_of_fame.update(population)
        
        # Вывести прогресс каждые 20 поколений
        if generation % 20 == 0 or generation == GA_GENERATIONS:
            current_stats = stats.compile(population)
            print(f"Поколение {generation}: Лучший = {hall_of_fame[0].fitness.values[0]:.2f}, "
                  f"Средний = {current_stats['avg']:.2f}")
    
    return hall_of_fame[0], hall_of_fame[0].fitness.values[0], None

# 7. Execute GA and Process Results
best_individual, best_fitness, logbook = run_ga()

print(f"\n--- Лучшее найденное решение ---")
print(f"Фитнес (makespan + штрафы): {best_fitness:.2f} минут")

# Создать детальное расписание для лучшего решения
def create_detailed_schedule(individual):
    schedule = {}
    machine_schedules = {stage: [] for stage in STAGES}
    
    for task_id in range(scheduler.num_tasks):
        schedule[task_id] = {'start': 0, 'end': 0}
    
    for position, task_id in enumerate(individual):
        task = scheduler.all_tasks[task_id]
        stage_name = task['stage_name']
        duration = task['duration']
        
        earliest_start = 0
        batch_id = task['batch_id']
        current_stage_index = task['stage_index']
        
        for other_task_id in scheduler.batch_stages[batch_id]:
            other_task = scheduler.all_tasks[other_task_id]
            if (other_task['stage_index'] < current_stage_index and 
                other_task_id in individual[:position+1]):
                earliest_start = max(earliest_start, schedule[other_task_id]['end'])
        
        machine_count = machines_available.get(stage_name, 1)
        machine_schedule = machine_schedules[stage_name]
        machine_schedule.sort(key=lambda x: x[1])
        
        start_time = max(earliest_start, 0)
        if len(machine_schedule) >= machine_count:
            earliest_free = min(machine_schedule[-machine_count:], key=lambda x: x[1])[1]
            start_time = max(start_time, earliest_free)
        
        end_time = start_time + duration
        schedule[task_id] = {'start': start_time, 'end': end_time}
        machine_schedule.append((start_time, end_time, task_id))
    
    return schedule

detailed_schedule = create_detailed_schedule(best_individual)
actual_makespan = max(detailed_schedule[task_id]['end'] for task_id in range(scheduler.num_tasks))

print(f"Фактический makespan: {actual_makespan:.2f} минут")
total_seconds = int(actual_makespan * 60)
tdelta = datetime.timedelta(seconds=total_seconds)
makespan_formatted = str(tdelta)
print(f"Время производства: {makespan_formatted} (Дни, ЧЧ:ММ:СС)")

# 8. Export Results
# Подготовить данные для экспорта
schedule_data_for_output = []
for task_id in range(scheduler.num_tasks):
    task = scheduler.all_tasks[task_id]
    start_time = detailed_schedule[task_id]['start']
    end_time = detailed_schedule[task_id]['end']
    
    schedule_data_for_output.append({
        "Batch_ID": task['batch_id'],
        "Stage": task['stage_name'],
        "Start_Time_Min": start_time,
        "End_Time_Min": end_time,
        "Duration_Min": task['duration']
    })

# Сортировать по времени начала
schedule_data_for_output.sort(key=lambda x: x['Start_Time_Min'])

# Записать CSV
try:
    csv_fieldnames = ["Batch_ID", "Stage", "Start_Time_Min", "End_Time_Min", "Duration_Min"]
    with open(OUTPUT_CSV_FILE, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_fieldnames)
        writer.writeheader()
        for row_data in schedule_data_for_output:
            writer.writerow(row_data)
    print(f"\nРасписание записано в CSV файл: '{OUTPUT_CSV_FILE}'")
except Exception as e:
    print(f"\nОшибка записи CSV файла: {e}")

# Записать TXT
try:
    with open(OUTPUT_TXT_FILE, 'w', encoding='utf-8') as txtfile:
        txtfile.write("--- Сводка по Производственному Расписанию (Генетический Алгоритм) ---\n\n")
        txtfile.write(f"Алгоритм: Генетический алгоритм (DEAP)\n")
        txtfile.write(f"Время производства (Makespan): {actual_makespan:.2f} минут\n")
        txtfile.write(f"Время производства (формат): {makespan_formatted} (Дни, ЧЧ:ММ:СС)\n")
        txtfile.write(f"Всего партий: {len(all_batches)}\n")
        txtfile.write(f"Всего задач: {len(schedule_data_for_output)}\n")
        txtfile.write(f"\nПараметры GA:\n")
        txtfile.write(f"  - Размер популяции: {GA_POPULATION_SIZE}\n")
        txtfile.write(f"  - Количество поколений: {GA_GENERATIONS}\n")
        txtfile.write(f"  - Вероятность скрещивания: {GA_CROSSOVER_PROB}\n")
        txtfile.write(f"  - Вероятность мутации: {GA_MUTATION_PROB}\n")
        txtfile.write(f"  - Размер турнира: {GA_TOURNAMENT_SIZE}\n")
        txtfile.write(f"\nПараметры ограничений (макс. время ожидания):\n")
        txtfile.write(f"  - Комбинирование -> Смешивание: {MAX_WAIT_COMBINING_MIXING_MIN} мин\n")
        txtfile.write(f"  - Смешивание -> Формовка: {MAX_WAIT_MIXING_FORMING_MIN} мин\n")
        txtfile.write(f"  - Формовка -> Расстойка: {MAX_WAIT_FORMING_PROOFING_MIN} мин\n")
        txtfile.write(f"  - Расстойка -> Выпекание: {MAX_WAIT_PROOFING_BAKING_MIN} мин\n")
        txtfile.write(f"\nДоступные ресурсы (машины):\n")
        for stage, count in machines_available.items():
            txtfile.write(f"  - {stage}: {count}\n")
        txtfile.write(f"\nФайл с детальным расписанием: {OUTPUT_CSV_FILE}\n")
        
        # Добавить статистику эволюции
        txtfile.write(f"\nСтатистика эволюции:\n")
        txtfile.write(f"  - Лучший фитнес: {best_fitness:.2f}\n")
        txtfile.write(f"  - Поколений обработано: {GA_GENERATIONS}\n")
    
    print(f"Сводная информация записана в TXT файл: '{OUTPUT_TXT_FILE}'")
except Exception as e:
    print(f"\nОшибка записи TXT файла: {e}")

print(f"\n--- Генетический алгоритм завершен ---")
print(f"Лучшее время производства: {actual_makespan:.2f} минут ({makespan_formatted})")