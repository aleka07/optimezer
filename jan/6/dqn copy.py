import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from collections import deque, namedtuple
import pandas as pd
import math
from typing import Dict, List, Tuple, Optional
import copy
import os
import datetime

# For plotting without a display
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# Transition for experience replay
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class ProductionEnvironment:
    """Среда для планирования производства (архитектура, ориентированная на партии)"""
    
    def __init__(self, tech_map_data: Dict, orders: Dict, machines_available: Dict,
                 batch_size: int, proportional_time_stages: List[str]):
        self.tech_map_data_str = tech_map_data
        self.orders_template = copy.deepcopy(orders)
        self.machines_available_template = copy.deepcopy(machines_available)
        self.batch_size = batch_size
        self.proportional_time_stages = proportional_time_stages
        self.stages = ["Комбинирование", "Смешивание", "Формовка", "Расстойка", "Выпекание", "Остывание"]
        
        self.max_wait_times = {
            ("Комбинирование", "Смешивание"): 1,
            ("Смешивание", "Формовка"): 1,
            ("Формовка", "Расстойка"): 5,
            ("Расстойка", "Выпекание"): 5
        }
        
        self._tech_map_minutes = self._parse_time_maps()
        self._initial_batches = self._prepare_batches()
        self.reset()

    def _time_to_minutes(self, time_str: str) -> int:
        try:
            parts = [int(p) for p in time_str.split(':')]
            if len(parts) == 3: return round(parts[0] * 60 + parts[1] + parts[2] / 60.0)
            if len(parts) == 2: return round(parts[0] + parts[1] / 60.0)
            return 0
        except: return 0
    
    def _parse_time_maps(self):
        return {p: {s: self._time_to_minutes(d.get(s, "0:0:0")) for s in self.stages} for p, d in self.tech_map_data_str.items()}

    def _prepare_batches(self) -> List[Dict]:
        """Подготовка партий (логика как в time_min.py и ga.py)"""
        batches = []
        batch_id_counter = 0
        for product, quantity in self.orders_template.items():
            if quantity <= 0 or product not in self._tech_map_minutes: continue
            
            num_batches = math.ceil(quantity / self.batch_size)
            for i in range(num_batches):
                is_partial = (i == num_batches - 1) and (quantity % self.batch_size != 0)
                current_batch_size = quantity % self.batch_size if is_partial else self.batch_size
                
                tasks = []
                for stage_idx, stage_name in enumerate(self.stages):
                    base_duration = self._tech_map_minutes[product][stage_name]
                    if base_duration > 0:
                        duration = math.ceil(base_duration * (current_batch_size / self.batch_size)) if (is_partial and stage_name in self.proportional_time_stages) else base_duration
                        if duration <= 0: duration = 1
                        tasks.append({"stage_idx": stage_idx, "stage": stage_name, "duration": duration, "start_time": None, "end_time": None, "scheduled": False})
                
                if tasks:
                    batches.append({'id': batch_id_counter, 'product': product, 'tasks': tasks, 'last_task_end_time': 0, 'completed_tasks': 0})
                    batch_id_counter += 1
        return batches

    def reset(self):
        """Сброс среды к начальному состоянию"""
        self.batches = copy.deepcopy(self._initial_batches)
        self.machine_free_time = {stage: [0] * count for stage, count in self.machines_available_template.items()}
        self.current_time = 0
        self.scheduled_tasks_count = 0
        self.total_tasks = sum(len(b['tasks']) for b in self.batches)
        return self._get_state()

    def _get_next_task_for_batch(self, batch: Dict) -> Optional[Dict]:
        """Находит следующую незапланированную задачу для партии."""
        for task in batch['tasks']:
            if not task['scheduled']:
                return task
        return None

    def _get_available_batches(self) -> List[Dict]:
        """Возвращает список партий, готовых к следующему этапу."""
        return [b for b in self.batches if b['completed_tasks'] < len(b['tasks'])]

    def _get_state(self) -> np.ndarray:
        """Получение текущего состояния среды, ориентированного на партии."""
        state = []
        
        # 1. Глобальная информация
        progress = self.scheduled_tasks_count / self.total_tasks if self.total_tasks > 0 else 0
        state.append(progress)
        state.append(self.current_time / 5000) # Нормализованное общее время

        # 2. Информация о загрузке машин
        for stage in self.stages:
            times = self.machine_free_time.get(stage, [0])
            state.append(np.mean(times) / 5000)
            state.append(np.std(times) / 5000)

        # 3. Информация о следующих доступных партиях
        available_batches = self._get_available_batches()
        # Сортируем, чтобы дать агенту консистентное представление
        available_batches.sort(key=lambda b: (b['last_task_end_time'], b['id']))

        max_batches_to_consider = 15 # Фокусируем внимание агента
        for i in range(max_batches_to_consider):
            if i < len(available_batches):
                batch = available_batches[i]
                next_task = self._get_next_task_for_batch(batch)
                if next_task:
                    # One-hot для следующего этапа
                    stage_one_hot = [0] * len(self.stages)
                    stage_one_hot[next_task['stage_idx']] = 1
                    state.extend(stage_one_hot)
                    # Длительность следующей задачи
                    state.append(next_task['duration'] / 100.0)
                    # Время ожидания партии
                    wait_time = self.current_time - batch['last_task_end_time']
                    state.append(wait_time / 100.0)
                    # Прогресс выполнения партии
                    state.append(batch['completed_tasks'] / len(batch['tasks']))
                else: # Партия закончилась, но попала в список - заполняем нулями
                    state.extend([0] * (len(self.stages) + 3))
            else: # Если доступных партий меньше, чем мы рассматриваем
                state.extend([0] * (len(self.stages) + 3))
        
        return np.array(state, dtype=np.float32)
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """Выполнение действия (выбор ПАРТИИ) в среде"""
        available_batches = self._get_available_batches()
        available_batches.sort(key=lambda b: (b['last_task_end_time'], b['id']))

        if not available_batches:
            return self._get_state(), 0, True, {'message': 'No available batches'}
        
        if action >= len(available_batches):
             action = len(available_batches) - 1

        batch_to_process = available_batches[action]
        task_to_schedule = self._get_next_task_for_batch(batch_to_process)

        if not task_to_schedule:
            # Этого не должно произойти, если available_batches не пуст
            return self._get_state(), -100, True, {'message': 'Logical error'}
        
        reward = self._schedule_task(batch_to_process, task_to_schedule)
        
        self.scheduled_tasks_count += 1
        done = self.scheduled_tasks_count >= self.total_tasks
        
        if done:
            # Финальная награда за makespan
            reward += max(0, 5000 - self.get_makespan())

        next_state = self._get_state()
        info = {'scheduled_task': f"{batch_to_process['product']}_batch_{batch_to_process['id']}_{task_to_schedule['stage']}"}
        
        return next_state, reward, done, info

    def _schedule_task(self, batch: Dict, task: Dict) -> float:
        """Планирование задачи для выбранной партии и расчет награды"""
        stage = task['stage']
        duration = task['duration']
        
        machine_times = self.machine_free_time[stage]
        earliest_machine_idx = np.argmin(machine_times)
        earliest_machine_time = machine_times[earliest_machine_idx]
        
        # Задача не может начаться раньше, чем закончился предыдущий этап этой же партии
        start_time = max(earliest_machine_time, batch['last_task_end_time'])
        end_time = start_time + duration
        
        task['start_time'], task['end_time'], task['scheduled'] = start_time, end_time, True
        machine_times[earliest_machine_idx] = end_time
        
        reward = self._calculate_reward(batch, task, start_time)
        
        batch['last_task_end_time'] = end_time
        batch['completed_tasks'] += 1
        
        # Обновляем общее время, если оно увеличилось
        if end_time > self.current_time:
             self.current_time = end_time
             
        return reward
    
    def _calculate_reward(self, batch: Dict, task: Dict, start_time: int) -> float:
        """Расчет награды за планирование задачи"""
        # Базовая отрицательная награда, чтобы агент стремился закончить быстрее
        reward = -5

        # Штраф за увеличение makespan
        makespan_increase = max(0, task['end_time'] - self.current_time)
        reward -= makespan_increase * 0.5
        
        # Штраф за время ожидания между этапами
        if batch['completed_tasks'] > 0:
            prev_task_end = max(t['end_time'] for t in batch['tasks'] if t['end_time'] is not None)
            prev_task_stage = batch['tasks'][batch['completed_tasks']-1]['stage']
            
            wait_time = start_time - prev_task_end
            max_wait = self.max_wait_times.get((prev_task_stage, task['stage']))
            
            if max_wait is not None and wait_time > max_wait:
                reward -= (wait_time - max_wait) * 2

        # Бонус за завершение всей партии
        if batch['completed_tasks'] + 1 == len(batch['tasks']):
            reward += 100
        
        return reward
    
    def get_makespan(self) -> int:
        return int(self.current_time)
    
    def get_schedule_dataframe(self) -> pd.DataFrame:
        schedule_data = []
        for batch in self.batches:
            batch_name = f"{batch['product']}_batch_{batch['id']}"
            for task in batch['tasks']:
                if task['scheduled']:
                    schedule_data.append({
                        'Batch_ID': batch_name, 'Stage': task['stage'],
                        'Start_Time_Min': task['start_time'], 'End_Time_Min': task['end_time'],
                        'Duration_Min': task['duration']
                    })
        df = pd.DataFrame(schedule_data)
        return df.sort_values(['Start_Time_Min', 'Batch_ID']) if not df.empty else df

class DQNNetwork(nn.Module):
    def __init__(self, state_size: int, action_size: int, hidden_size: int = 512):
        super(DQNNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class DQNAgent:
    def __init__(self, state_size: int, action_size: int, lr: float = 1e-4):
        self.state_size, self.action_size = state_size, action_size
        self.memory = deque(maxlen=20000)
        self.epsilon, self.epsilon_min, self.epsilon_decay = 1.0, 0.01, 0.999
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Используется устройство: {self.device}")
        
        self.q_network = DQNNetwork(state_size, action_size).to(self.device)
        self.target_network = DQNNetwork(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.update_target_network()
    
    def update_target_network(self): self.target_network.load_state_dict(self.q_network.state_dict())
    def remember(self, state, action, reward, next_state, done): self.memory.append(Transition(state, action, next_state, reward))
    
    def act(self, state, available_actions_count):
        if np.random.rand() <= self.epsilon:
            return random.randrange(available_actions_count)
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)[0]
            # Маскируем недоступные действия
            q_values[available_actions_count:] = -float('inf')
            return torch.argmax(q_values).item()
    
    def replay(self, batch_size=64):
        if len(self.memory) < batch_size: return
        
        minibatch = random.sample(self.memory, batch_size)
        
        states = torch.FloatTensor(np.array([t.state for t in minibatch])).to(self.device)
        actions = torch.LongTensor([t.action for t in minibatch]).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor([t.reward for t in minibatch]).to(self.device)
        
        non_final_mask = torch.tensor([t.next_state is not None for t in minibatch], device=self.device, dtype=torch.bool)
        non_final_next_states = torch.FloatTensor(np.array([t.next_state for t in minibatch if t.next_state is not None])).to(self.device)
        
        current_q_values = self.q_network(states).gather(1, actions)
        
        next_q_values = torch.zeros(batch_size, device=self.device)
        if len(non_final_next_states) > 0:
            next_q_values[non_final_mask] = self.target_network(non_final_next_states).max(1)[0].detach()

        target_q_values = (next_q_values * 0.99) + rewards
        
        loss = F.smooth_l1_loss(current_q_values, target_q_values.unsqueeze(1))
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1)
        self.optimizer.step()
        
        if self.epsilon > self.epsilon_min: self.epsilon *= self.epsilon_decay

class ProductionSchedulerDQN:
    def __init__(self, env: ProductionEnvironment):
        self.env = env
        state_size = len(self.env.reset())
        self.action_size = 50 # Макс. кол-во партий для выбора
        self.agent = DQNAgent(state_size, self.action_size)
        self.training_scores, self.training_makespans = [], []
    
    def train(self, episodes: int = 1000, target_update_freq: int = 10):
        print(f"Начинаем обучение DQN агента на {episodes} эпизодах...")
        for episode in range(1, episodes + 1):
            state, total_reward, done = self.env.reset(), 0, False
            
            while not done:
                available_batches_count = len(self.env._get_available_batches())
                if available_batches_count == 0: break
                
                effective_action_space = min(available_batches_count, self.action_size)
                action = self.agent.act(state, effective_action_space)
                
                next_state, reward, done, _ = self.env.step(action)
                total_reward += reward
                self.agent.remember(state, action, reward, next_state if not done else None, done)
                state = next_state
            
            self.agent.replay()
            if episode % target_update_freq == 0: self.agent.update_target_network()
            
            self.training_scores.append(total_reward)
            self.training_makespans.append(self.env.get_makespan())
            
            if episode % 50 == 0:
                avg_score = np.mean(self.training_scores[-50:])
                avg_makespan = np.mean(self.training_makespans[-50:])
                print(f"Эпизод {episode}/{episodes}, Средняя награда: {avg_score:.2f}, "
                      f"Средний makespan: {avg_makespan:.0f} мин, Epsilon: {self.agent.epsilon:.3f}")
    
    def generate_schedule(self) -> pd.DataFrame:
        print("\nГенерируем финальное расписание с обученным агентом...")
        self.agent.epsilon = 0.0
        state, done = self.env.reset(), False
        while not done:
            available_batches_count = len(self.env._get_available_batches())
            if available_batches_count == 0: break
            effective_action_space = min(available_batches_count, self.action_size)
            action = self.agent.act(state, effective_action_space)
            state, _, done, _ = self.env.step(action)
        
        makespan = self.env.get_makespan()
        print(f"Финальный makespan: {makespan} минут ({makespan/60:.2f} часов)")
        return self.env.get_schedule_dataframe()
    
    def plot_training_progress(self, filename='dqn_training_progress.png'):
        fig, ax1 = plt.subplots(figsize=(12, 5))
        ax2 = ax1.twinx()
        ax1.plot(self.training_scores, 'g-')
        ax2.plot(self.training_makespans, 'b-')
        ax1.set_xlabel('Episode'); ax1.set_ylabel('Total Reward', color='g'); ax2.set_ylabel('Makespan (minutes)', color='b')
        plt.title('DQN Training Progress')
        plt.savefig(filename)
        plt.close(fig)
        print(f"График обучения сохранен в {filename}")


if __name__ == "__main__":
    # --- СИНХРОНИЗИРОВАННЫЕ ДАННЫЕ ---
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
    machines_available = { "Комбинирование": 2, "Смешивание": 3, "Формовка": 2, "Расстойка": 8, "Выпекание": 6, "Остывание": 50 }
    BATCH_SIZE = 100
    proportional_time_stages = ["Комбинирование", "Формовка"]

    # --- НАСТРОЙКИ ВЫВОДА ---
    script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
    OUTPUT_CSV_FILE = os.path.join(script_dir, 'dqn_production_schedule.csv')
    OUTPUT_TXT_FILE = os.path.join(script_dir, 'dqn_production_summary.txt')
    
    # --- ЗАПУСК ---
    env = ProductionEnvironment(tech_map_data, orders, machines_available, BATCH_SIZE, proportional_time_stages)
    scheduler = ProductionSchedulerDQN(env)
    scheduler.train(episodes=1000)
    schedule_df = scheduler.generate_schedule()
    
    # --- СОХРАНЕНИЕ РЕЗУЛЬТАТОВ ---
    schedule_df.to_csv(OUTPUT_CSV_FILE, index=False, encoding='utf-8')
    print(f"\nРасписание сохранено в файл: {OUTPUT_CSV_FILE}")
    
    try:
        with open(OUTPUT_TXT_FILE, 'w', encoding='utf-8') as f:
            makespan = schedule_df['End_Time_Min'].max() if not schedule_df.empty else 0
            tdelta = datetime.timedelta(minutes=float(makespan))
            f.write("--- Сводка по Производственному Расписанию (DQN) ---\n\n")
            f.write(f"Общее время производства (Makespan): {makespan:.2f} минут\n")
            f.write(f"Общее время производства (формат): {str(tdelta)}\n\n")
            f.write(f"Всего партий: {len(env._initial_batches)}\n")
            f.write(f"Эпизодов обучения: 1000\n")
            f.write(f"Файл с детальным расписанием: {os.path.basename(OUTPUT_CSV_FILE)}\n")
    except Exception as e:
        print(f"Ошибка записи TXT файла: {e}")
    
    scheduler.plot_training_progress()