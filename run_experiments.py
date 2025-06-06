# --- START OF FILE run_experiments.py ---
import math
import collections
import csv # Для записи FIFO результатов, если нужно
import datetime # Для time_minimizer
from ortools.sat.python import cp_model # Для time_minimizer

# Импортируем FIFO симулятор
from fifo_simulator import simulate_fifo

# --- 1. ОБЩИЕ ВХОДНЫЕ ДАННЫЕ (Идентичны тем, что в time_minimizer.py) ---
tech_map_data = {
    # ВАША БОЛЬШАЯ tech_map_data СЮДА
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
    "Чиабатта (ориг.)":                      {"Комбинирование": "0:03:39", "Смешивание": "0:15:00", "Формовка": "0:20:00", "Расстойка": "0:30:00", "Выпекание": "0:25:00", "Остывание": "1:00:00", "Упаковка": "0:15:00"},
    "Булочка (ориг.)":               {"Комбинирование": "0:03:39", "Смешивание": "0:11:00", "Формовка": "0:21:30", "Расстойка": "0:45:00", "Выпекание": "0:22:00", "Остывание": "1:15:00", "Упаковка": "0:15:00"},
    "Булочка для хот-дог/гамбургера (ориг.)": {"Комбинирование": "0:03:39", "Смешивание": "0:12:00", "Формовка": "0:25:00", "Расстойка": "0:00:00", "Выпекание": "0:40:00", "Остывание": "2:45:00", "Упаковка": "0:15:00"},
    "Бриошь (ориг.)":                        {"Комбинирование": "0:03:39", "Смешивание": "0:07:00", "Формовка": "0:25:00", "Расстойка": "0:20:00", "Выпекание": "0:03:00", "Остывание": "0:20:00", "Упаковка": "0:15:00"},
    "Булочка для гамбургера большой/ с кунжутом (ориг.)": {"Комбинирование": "0:03:39", "Смешивание": "0:10:00", "Формовка": "0:30:00", "Расстойка": "1:00:00", "Выпекание": "0:35:00", "Остывание": "2:30:00", "Упаковка": "0:15:00"},
    "Немецкий хлеб (ориг.)":                 {"Комбинирование": "0:03:39", "Смешивание": "0:12:00", "Формовка": "0:30:00", "Расстойка": "1:10:00", "Выпекание": "0:40:00", "Остывание": "2:40:00", "Упаковка": "0:15:00"},
    "Хлеб «Зерновой» (ориг.)":               {"Комбинирование": "0:03:39", "Смешивание": "0:18:00", "Формовка": "0:35:00", "Расстойка": "1:00:00", "Выпекание": "0:18:00", "Остывание": "1:00:00", "Упаковка": "0:15:00"},
    "Хот-дог/Гамбургер солодовый с семечками (ориг.)": {"Комбинирование": "0:03:39", "Смешивание": "0:09:00", "Формовка": "0:40:00", "Расстойка": "0:40:00", "Выпекание": "0:20:00", "Остывание": "0:45:00", "Упаковка": "0:15:00"},
    "Сэндвич (ориг.)":                       {"Комбинирование": "0:03:39", "Смешивание": "0:08:00", "Формовка": "1:00:00", "Расстойка": "0:50:00", "Выпекание": "0:25:00", "Остывание": "2:00:00", "Упаковка": "0:15:00"},
    "Хлеб «Тартин бездрожжевой» (ориг.)":     {"Комбинирование": "0:03:39", "Смешивание": "0:08:00", "Формовка": "3:00:00", "Расстойка": "0:00:00", "Выпекание": "0:18:00", "Остывание": "0:40:00", "Упаковка": "0:15:00"},
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
    "Комбинирование": 1, "Смешивание": 3, "Формовка": 2, "Расстойка": 8,
    "Выпекание": 6, "Остывание": 25, "Упаковка": 10,
}
BATCH_SIZE = 100
STAGES = [
    "Комбинирование", "Смешивание", "Формовка", "Расстойка",
    "Выпекание", "Остывание", "Упаковка",
]

# Параметры ожидания (для анализа FIFO и для CP модели)
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

# Словарь для передачи в FIFO симулятор для проверки нарушений
critical_pairs_for_fifo_check = {
    (CRITICAL_STAGE_BEFORE_0, CRITICAL_STAGE_AFTER_0): MAX_WAIT_COMBINING_MIXING_MIN,
    (CRITICAL_STAGE_BEFORE_1, CRITICAL_STAGE_AFTER_1): MAX_WAIT_MIXING_FORMING_MIN,
    (CRITICAL_STAGE_BEFORE_2, CRITICAL_STAGE_AFTER_2): MAX_WAIT_FORMING_PROOFING_MIN,
    (CRITICAL_STAGE_BEFORE_3, CRITICAL_STAGE_AFTER_3): MAX_WAIT_PROOFING_BAKING_MIN,
}

# --- 2. ФУНКЦИИ ПРЕДОБРАБОТКИ (из time_minimizer.py) ---
def time_str_to_minutes_int_local(time_str): # Переименовал, чтобы не конфликтовать, если импортируем из time_minimizer
    try:
        parts = list(map(int, time_str.split(':')))
        if len(parts) == 3: h, m, s = parts; return round(h * 60 + m + s / 60.0)
        elif len(parts) == 2: m, s = parts; return round(m + s / 60.0)
        else: return 0
    except: return 0

def prepare_data_for_scheduling(tech_map_data_input, orders_input, batch_size_input, stages_list_input):
    tech_map_minutes_int_local = {}
    for product_name_map, stages_data_map in tech_map_data_input.items():
        if product_name_map in orders_input and orders_input[product_name_map] > 0:
            tech_map_minutes_int_local[product_name_map] = {}
            for stage_name_map in stages_list_input:
                time_str = stages_data_map.get(stage_name_map, "0:00:00")
                duration = time_str_to_minutes_int_local(time_str)
                tech_map_minutes_int_local[product_name_map][stage_name_map] = duration

    all_batches_local = []
    product_order_for_fifo = list(orders_input.keys()) # Сохраняем исходный порядок продуктов для FIFO

    # Для FIFO: сначала все партии первого продукта, потом все партии второго и т.д.
    # Или можно сделать более сложную логику FIFO, если нужно (например, по времени поступления заказа)
    # Пока что: сортируем партии по продукту (в порядке их появления в `orders`), затем по номеру партии.
    
    temp_batches_for_sorting = []
    for product_idx, product in enumerate(product_order_for_fifo):
        quantity_ordered = orders_input.get(product, 0)
        if quantity_ordered <= 0: continue
        if product not in tech_map_minutes_int_local:
            print(f"Эксперимент: Продукт '{product}' из заказа отсутствует в tech_map_minutes_int_local. Пропускается.")
            continue
        
        num_batches = math.ceil(quantity_ordered / batch_size_input)
        for i in range(num_batches):
            batch_id = f"{product}_batch_{i+1}"
            batch_tasks = []
            for stage_index, stage_name in enumerate(stages_list_input):
                duration = tech_map_minutes_int_local[product].get(stage_name, 0)
                # В FIFO добавляем все задачи, даже с нулевой длительностью, если они есть в STAGES,
                # т.к. функция simulate_fifo сама их обработает (пропустит).
                # Но для CP в all_batches обычно добавляют только >0.
                # Здесь для консистентности с CP, добавляем только >0, а FIFO учтет это.
                # Или нужно пересмотреть generate_all_batches для CP, чтобы он тоже включал stage_index
                if duration > 0: # Для CP и для передачи в FIFO (FIFO проигнорирует duration=0)
                    batch_tasks.append({
                        "batch_id": batch_id, # Не используется в simulate_fifo, но полезно для консистентности
                        "stage_index": stage_index, 
                        "stage_name": stage_name, 
                        "duration": duration, 
                    })
            if batch_tasks:
                temp_batches_for_sorting.append({
                    "id": batch_id, 
                    "product": product, 
                    "tasks": batch_tasks,
                    "product_order_idx": product_idx, # Для сортировки FIFO
                    "batch_num_in_product": i # Для сортировки FIFO
                })
    
    # Сортировка для FIFO: по порядку продукта в словаре orders, затем по номеру партии
    all_batches_local = sorted(temp_batches_for_sorting, key=lambda b: (b["product_order_idx"], b["batch_num_in_product"]))
    
    # Убираем вспомогательные ключи сортировки, если они не нужны дальше
    for b in all_batches_local:
        b.pop("product_order_idx", None)
        b.pop("batch_num_in_product", None)

    if not all_batches_local:
        print("Эксперимент: Нет партий для производства.")
        return None, None
    
    return all_batches_local, tech_map_minutes_int_local


# --- 3. ОСНОВНОЙ БЛОК ЭКСПЕРИМЕНТА ---
if __name__ == "__main__":
    print("--- Начало Эксперимента по Сравнению CP и FIFO ---")

    # 3.1 Подготовка общих данных (партий)
    # Используем `orders` (переименованный `orders_one_day`)
    all_batches_for_experiment, tech_map_minutes_int_for_experiment = prepare_data_for_scheduling(
        tech_map_data, orders, BATCH_SIZE, STAGES
    )

    if not all_batches_for_experiment:
        print("Не удалось подготовить партии для эксперимента. Выход.")
        exit()
        
    print(f"\nВсего партий для эксперимента: {len(all_batches_for_experiment)}")
    # print("Первые 3 партии для FIFO (порядок важен):")
    # for i in range(min(3, len(all_batches_for_experiment))):
    # print(f"  - {all_batches_for_experiment[i]['id']}")


    # 3.2 Запуск CP Решателя (Предполагается, что time_minimizer.py уже настроен и запускается отдельно)
    # Здесь мы просто считаем, что он был запущен, и мы можем прочитать его результаты
    # или, если бы time_minimizer.py был модулем, мы бы вызвали его функцию.
    # Для простоты, сейчас мы просто выведем сообщение.
    # В реальном сценарии вы бы запустили `python time_minimizer.py` с текущими `orders` и `BATCH_SIZE`,
    # а затем считали бы makespan из `production_summary.txt`.
    
    print("\n--- CP Оптимизация (запустите time_minimizer.py отдельно) ---")
    print("Пожалуйста, запустите 'time_minimizer.py' с текущими настройками 'orders',")
    print("'BATCH_SIZE', 'machines_available' и 'tech_map_data', как определено в этом скрипте.")
    print("Затем введите полученный Makespan (в минутах) из 'production_summary.txt'.")
    
    cp_makespan_input = input("Введите Makespan от CP решателя (или 0, если не запускали): ")
    try:
        cp_makespan = float(cp_makespan_input)
    except ValueError:
        print("Некорректный ввод, используется Makespan CP = 0.")
        cp_makespan = 0

    # 3.3 Запуск FIFO Симулятора
    # all_batches_for_experiment уже отсортированы в порядке FIFO по продуктам и номеру партии
    fifo_schedule, fifo_makespan, fifo_wait_log, fifo_violations = simulate_fifo(
        all_batches_for_experiment,
        machines_available, # Используем тот же словарь
        STAGES,             # Тот же список этапов
        critical_pairs_for_fifo_check # Передаем лимиты для анализа
    )

    # 3.4 Вывод результатов сравнения
    print("\n\n--- Результаты Сравнения ---")
    print(f"Makespan (CP Оптимизация): {cp_makespan:.2f} минут")
    print(f"Makespan (FIFO Симуляция): {fifo_makespan:.2f} минут")

    if cp_makespan > 0 and fifo_makespan > 0:
        improvement = ((fifo_makespan - cp_makespan) / fifo_makespan) * 100
        print(f"Улучшение Makespan с помощью CP: {improvement:.2f}% (по сравнению с FIFO)")
    
    print(f"\nFIFO - Нарушения максимального времени ожидания: {len(fifo_violations)}")
    if fifo_violations:
        print("Примеры нарушений в FIFO:")
        for i, viol in enumerate(fifo_violations):
            if i < 5: # Показать не более 5 примеров
                print(f"  - Партия: {viol['batch_id']}, Этапы: {viol['from_stage']}->{viol['to_stage']}, Ожидание: {viol['actual_wait']:.0f} мин, Лимит: {viol['limit']} мин, Превышение: {viol['violation_amount']:.0f} мин")
            else:
                print(f"  ... и еще {len(fifo_violations) - 5} нарушений.")
                break
    
    # Можно также сохранить расписание FIFO в CSV для сравнения с диаграммой Ганта CP
    if fifo_schedule:
        fifo_csv_file = 'fifo_production_schedule.csv'
        csv_fieldnames_fifo = ["Batch_ID", "Stage", "Start_Time_Min", "End_Time_Min", "Duration_Min", "Machine_ID"]
        try:
            with open(fifo_csv_file, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=csv_fieldnames_fifo)
                writer.writeheader()
                # Сортируем для лучшей читаемости CSV (не обязательно для логики FIFO)
                # fifo_schedule_sorted = sorted(fifo_schedule, key=lambda x: (x['Start_Time_Min'], x['Batch_ID'], STAGES.index(x['Stage'])))
                # writer.writerows(fifo_schedule_sorted) # Если хотим сортированный
                writer.writerows(fifo_schedule) # Если хотим в порядке генерации
            print(f"\nРасписание FIFO успешно записано в CSV файл: '{fifo_csv_file}'")
            print(f"Вы можете использовать 'diagram.py', изменив INPUT_CSV_FILE на '{fifo_csv_file}', для визуализации FIFO.")
        except Exception as e:
            print(f"\nОшибка записи CSV файла для FIFO '{fifo_csv_file}': {e}")

    print("\n--- Эксперимент Завершен ---")

# --- END OF FILE run_experiments.py ---