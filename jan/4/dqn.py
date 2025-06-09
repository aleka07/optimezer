import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from collections import deque, namedtuple
import matplotlib.pyplot as plt
import pandas as pd
import math
from typing import Dict, List, Tuple, Optional
import copy
import os

# Transition for experience replay
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class ProductionEnvironment:
    """Среда для планирования производства хлебобулочных изделий"""
    
    def __init__(self, tech_map_data: Dict, orders: Dict, machines_available: Dict, 
                 batch_size: int = 100, max_wait_times: Dict = None):
        self.tech_map_data = tech_map_data
        self.orders_template = copy.deepcopy(orders)
        self.machines_available_template = copy.deepcopy(machines_available)
        self.batch_size = batch_size
        self.stages = ["Комбинирование", "Смешивание", "Формовка", "Расстойка", 
                       "Выпекание", "Остывание"]
        
        self.max_wait_times = max_wait_times or {
            ("Комбинирование", "Смешивание"): 1,
            ("Смешивание", "Формовка"): 1,
            ("Формовка", "Расстойка"): 5,
            ("Расстойка", "Выпекание"): 5
        }
        
        self._initial_batches = self._prepare_batches()
        self.reset()
    
    def _time_to_minutes(self, time_str: str) -> int:
        """Конвертация времени в минуты"""
        try:
            parts = [float(p) for p in time_str.split(':')]
            if len(parts) == 3:
                h, m, s = parts
                return round(h * 60 + m + s / 60.0)
            elif len(parts) == 2:
                m, s = parts
                return round(m + s / 60.0)
        except:
            return 0
    
    def _prepare_batches(self) -> List[Dict]:
        """Подготовка партий для производства"""
        batches = []
        batch_id = 0
        
        for product, quantity in self.orders_template.items():
            if quantity <= 0:
                continue
            
            if product not in self.tech_map_data:
                print(f"Предупреждение: Продукт '{product}' из заказа не найден в технологической карте. Пропускается.")
                continue

            num_batches = math.ceil(quantity / self.batch_size)
            
            for i in range(num_batches):
                batch = {
                    'id': batch_id,
                    'product': product,
                    'tasks': []
                }
                
                for stage_idx, stage in enumerate(self.stages):
                    duration = self._time_to_minutes(
                        self.tech_map_data[product].get(stage, "0:00:00")
                    )
                    if duration > 0:
                        batch['tasks'].append({
                            'stage_idx': stage_idx,
                            'stage': stage,
                            'duration': duration,
                            'start_time': None,
                            'end_time': None,
                            'scheduled': False
                        })
                
                if batch['tasks']:
                    batches.append(batch)
                    batch_id += 1
        return batches

    def reset(self):
        """Сброс среды к начальному состоянию"""
        self.batches = copy.deepcopy(self._initial_batches)
        self.num_batches = len(self.batches)
        self.total_tasks = sum(len(batch['tasks']) for batch in self.batches)
        
        self.machine_free_time = {stage: [0] * count 
                                 for stage, count in self.machines_available_template.items()}
        
        self.current_time = 0
        self.scheduled_tasks = 0
        self.total_penalty = 0
        
        return self._get_state()
    
    def _get_state(self) -> np.ndarray:
        """Получение текущего состояния среды"""
        state = []
        
        progress = self.scheduled_tasks / self.total_tasks if self.total_tasks > 0 else 0
        state.append(progress)
        
        state.append(self.current_time / 1000)
        
        for stage in self.stages:
            if stage in self.machine_free_time:
                min_free_time = min(self.machine_free_time[stage])
                state.append(min_free_time / 1000)
            else:
                state.append(0)
        
        next_tasks_info = self._get_next_tasks_info()
        state.extend(next_tasks_info)
        
        return np.array(state, dtype=np.float32)
    
    def _get_next_tasks_info(self) -> List[float]:
        """Получение информации о следующих задачах"""
        info = []
        max_tasks_to_consider = 10
        
        available_tasks = []
        for batch in self.batches:
            for task in batch['tasks']:
                if not task['scheduled'] and self._can_schedule_task(batch, task):
                    available_tasks.append((batch, task))
        
        available_tasks.sort(key=lambda x: (x[1]['stage_idx'], x[1]['duration']))
        
        for i in range(max_tasks_to_consider):
            if i < len(available_tasks):
                batch, task = available_tasks[i]
                stage_one_hot = [0] * len(self.stages)
                stage_one_hot[task['stage_idx']] = 1
                info.extend(stage_one_hot)
                info.append(task['duration'] / 100)
                info.append(self._calculate_urgency(batch, task))
            else:
                info.extend([0] * (len(self.stages) + 2))
        
        return info
    
    # --- ИСПРАВЛЕНИЕ №1 ЗДЕСЬ ---
    def _can_schedule_task(self, batch: Dict, task: Dict) -> bool:
        """
        Проверка возможности планирования задачи.
        Задача может быть запланирована, если ее прямой предшественник (если есть) уже запланирован.
        """
        # Находим задачу-предшественника с самым большим stage_idx < текущего stage_idx
        predecessor_task = None
        max_prev_idx = -1
        for t in batch['tasks']:
            if t['stage_idx'] < task['stage_idx'] and t['stage_idx'] > max_prev_idx:
                max_prev_idx = t['stage_idx']
                predecessor_task = t
        
        # Если предшественник существует, он должен быть запланирован
        if predecessor_task is not None:
            return predecessor_task['scheduled']
        
        # Если предшественника нет (это первая задача в партии), ее можно планировать
        return True

    def _calculate_urgency(self, batch: Dict, task: Dict) -> float:
        """Расчет срочности задачи"""
        return task['stage_idx'] / len(self.stages)
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """Выполнение действия в среде"""
        available_tasks = []
        # Собираем задачи вместе с их партиями
        for batch in self.batches:
            for task in batch['tasks']:
                if not task['scheduled'] and self._can_schedule_task(batch, task):
                    available_tasks.append((batch, task))
        
        available_tasks.sort(key=lambda x: (x[1]['stage_idx'], x[1]['duration']))

        if not available_tasks:
            return self._get_state(), 0, True, {'message': 'No available tasks'}
        
        if action >= len(available_tasks):
             action = len(available_tasks) - 1

        batch_of_task, task_to_schedule = available_tasks[action]
        
        reward = self._schedule_task(batch_of_task, task_to_schedule)
        
        self.scheduled_tasks += 1
        
        done = self.scheduled_tasks >= self.total_tasks
        
        next_state = self._get_state()
        info = {'scheduled_task': f"{batch_of_task['product']}_{task_to_schedule['stage']}"}
        
        return next_state, reward, done, info
    
    # --- ИСПРАВЛЕНИЕ №2 ЗДЕСЬ ---
    def _schedule_task(self, batch: Dict, task: Dict) -> float:
        """Планирование задачи и расчет награды"""
        stage = task['stage']
        duration = task['duration']
        
        if stage not in self.machine_free_time:
            return -1000
        
        machine_times = self.machine_free_time[stage]
        earliest_machine_idx = np.argmin(machine_times)
        earliest_time = machine_times[earliest_machine_idx]
        
        # Надежный поиск предшественника
        predecessor_task = None
        max_prev_idx = -1
        for t in batch['tasks']:
            if t['stage_idx'] < task['stage_idx'] and t['stage_idx'] > max_prev_idx:
                max_prev_idx = t['stage_idx']
                predecessor_task = t

        # Получаем время окончания предшественника, если он существует
        prev_task_end = 0
        prev_stage_name = None
        if predecessor_task:
            # .get() с default значением 0 - это защита от None, если вдруг состояние окажется некорректным
            prev_task_end = predecessor_task.get('end_time', 0) or 0
            prev_stage_name = predecessor_task['stage']

        # Теперь prev_task_end гарантированно является числом
        min_start_time = max(earliest_time, prev_task_end)
        
        start_time = min_start_time
        end_time = start_time + duration
        
        task['start_time'] = start_time
        task['end_time'] = end_time
        task['scheduled'] = True
        
        machine_times[earliest_machine_idx] = end_time
        
        self.current_time = max(self.current_time, end_time)
        
        reward = self._calculate_reward(task, start_time, prev_task_end, prev_stage_name)
        
        return reward
    
    def _calculate_reward(self, task: Dict, start_time: int, 
                         prev_task_end: int, prev_stage_name: Optional[str]) -> float:
        """Расчет награды за планирование задачи"""
        reward = 10
        
        if prev_task_end > 0 and prev_stage_name:
            wait_time = start_time - prev_task_end
            max_wait = self.max_wait_times.get((prev_stage_name, task['stage']))
            if max_wait is not None and wait_time > max_wait:
                penalty = (wait_time - max_wait) * 2
                reward -= penalty
                self.total_penalty += penalty
        
        if start_time <= self.current_time + 60:
            reward += 5
        
        makespan_increase = max(0, task['end_time'] - self.current_time)
        reward -= makespan_increase * 0.1
        
        return reward
    
    def get_makespan(self) -> int:
        """Получение общего времени производства"""
        max_end_time = 0
        for batch in self.batches:
            for task in batch['tasks']:
                if task['scheduled'] and task['end_time'] is not None:
                    max_end_time = max(max_end_time, task['end_time'])
        return int(max_end_time)
    
    def get_schedule_dataframe(self) -> pd.DataFrame:
        """Получение расписания в виде DataFrame"""
        schedule_data = []
        for batch in self.batches:
            batch_id = f"{batch['product']}_batch_{batch['id']}"
            for task in batch['tasks']:
                if task['scheduled']:
                    schedule_data.append({
                        'Batch_ID': batch_id,
                        'Product': batch['product'],
                        'Stage': task['stage'],
                        'Start_Time_Min': task['start_time'],
                        'End_Time_Min': task['end_time'],
                        'Duration_Min': task['duration']
                    })
        
        df = pd.DataFrame(schedule_data)
        if not df.empty:
            df = df.sort_values(['Start_Time_Min', 'Batch_ID'])
        return df


class DQNNetwork(nn.Module):
    """Deep Q-Network для планирования производства"""
    def __init__(self, state_size: int, action_size: int, hidden_size: int = 512):
        super(DQNNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc4 = nn.Linear(hidden_size // 2, action_size)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x


class DQNAgent:
    """DQN агент для планирования производства"""
    def __init__(self, state_size: int, action_size: int, lr: float = 1e-3):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=10000)
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Используется устройство: {self.device}")
        
        self.q_network = DQNNetwork(state_size, action_size).to(self.device)
        self.target_network = DQNNetwork(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.update_target_network()
    
    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append(Transition(state, action, next_state, reward))
    
    def act(self, state, available_actions_count):
        if available_actions_count == 0:
            return 0 # Should not happen if called correctly
        if np.random.random() <= self.epsilon:
            return random.randrange(available_actions_count)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            
            # Маскируем недоступные действия, присваивая им очень низкое значение
            mask = torch.ones(1, self.action_size, device=self.device) * float('-inf')
            mask[0, :available_actions_count] = 0
            masked_q_values = q_values + mask

            return masked_q_values.argmax().item()
    
    def replay(self, batch_size=32):
        if len(self.memory) < batch_size:
            return
        
        minibatch = random.sample(self.memory, batch_size)
        
        states = np.array([t.state for t in minibatch])
        actions = np.array([t.action for t in minibatch])
        rewards = np.array([t.reward for t in minibatch])
        next_states = np.array([t.next_state for t in minibatch if t.next_state is not None])
        
        # Маска для состояний, которые не являются последними в эпизоде
        non_final_mask = torch.tensor([t.next_state is not None for t in minibatch], device=self.device, dtype=torch.bool)
        
        state_batch = torch.FloatTensor(states).to(self.device)
        action_batch = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        reward_batch = torch.FloatTensor(rewards).to(self.device)
        
        current_q_values = self.q_network(state_batch).gather(1, action_batch)
        
        next_q_values = torch.zeros(batch_size, device=self.device)
        if len(next_states) > 0:
            non_final_next_states_batch = torch.FloatTensor(next_states).to(self.device)
            next_q_values[non_final_mask] = self.target_network(non_final_next_states_batch).max(1)[0].detach()

        target_q_values = (next_q_values * 0.99) + reward_batch
        
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


class ProductionSchedulerDQN:
    """Планировщик производства на основе DQN"""
    
    def __init__(self, tech_map_data: Dict, orders: Dict, machines_available: Dict, batch_size: int = 100):
        self.env = ProductionEnvironment(tech_map_data, orders, machines_available, batch_size=batch_size)
        
        sample_state = self.env.reset()
        state_size = len(sample_state)
        self.action_size = 100 
        
        self.agent = DQNAgent(state_size, self.action_size)
        
        self.training_scores = []
        self.training_makespans = []
    
    def train(self, episodes: int = 1000):
        print(f"Начинаем обучение DQN агента на {episodes} эпизодах...")
        
        for episode in range(episodes):
            state = self.env.reset()
            total_reward = 0
            steps = 0
            
            while steps < 3000:
                available_tasks = []
                for batch in self.env.batches:
                    for task in batch['tasks']:
                        if not task['scheduled'] and self.env._can_schedule_task(batch, task):
                            available_tasks.append(task)
                
                available_actions_count = len(available_tasks)

                if available_actions_count == 0:
                    break
                
                effective_action_space = min(available_actions_count, self.agent.action_size)
                action = self.agent.act(state, effective_action_space)
                
                next_state, reward, done, info = self.env.step(action)
                
                self.agent.remember(state, action, reward, next_state if not done else None, done)

                state = next_state
                total_reward += reward
                steps += 1

                if done:
                    break
            
            if len(self.agent.memory) > 32:
                self.agent.replay()
            
            if episode > 0 and episode % 10 == 0:
                self.agent.update_target_network()
            
            makespan = self.env.get_makespan()
            self.training_scores.append(total_reward)
            self.training_makespans.append(makespan)
            
            if episode % 50 == 0:
                avg_score = np.mean(self.training_scores[-50:]) if self.training_scores else 0
                avg_makespan = np.mean(self.training_makespans[-50:]) if self.training_makespans else 0
                print(f"Эпизод {episode}, Средняя награда: {avg_score:.2f}, "
                      f"Средний makespan: {avg_makespan:.0f} мин, "
                      f"Epsilon: {self.agent.epsilon:.3f}")
    
    def generate_schedule(self) -> pd.DataFrame:
        print("Генерируем оптимальное расписание...")
        
        original_epsilon = self.agent.epsilon
        self.agent.epsilon = 0.0
        
        state = self.env.reset()
        steps = 0
        
        while steps < 3000:
            available_tasks = []
            for batch in self.env.batches:
                for task in batch['tasks']:
                    if not task['scheduled'] and self.env._can_schedule_task(batch, task):
                        available_tasks.append(task)
            
            available_actions_count = len(available_tasks)
            
            if available_actions_count == 0:
                break

            effective_action_space = min(available_actions_count, self.agent.action_size)
            action = self.agent.act(state, effective_action_space)
            
            state, _, done, _ = self.env.step(action)
            steps += 1
            
            if done:
                break
        
        self.agent.epsilon = original_epsilon
        
        makespan = self.env.get_makespan()
        print(f"Финальный makespan: {makespan} минут ({makespan/60:.1f} часов)")
        
        return self.env.get_schedule_dataframe()
    
    def plot_training_progress(self):
        """Визуализация прогресса обучения"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        ax1.plot(self.training_scores)
        ax1.set_title('Training Rewards')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Total Reward')
        ax1.grid(True)
        
        ax2.plot(self.training_makespans)
        ax2.set_title('Training Makespan')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Makespan (minutes)')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()

# Пример использования
if __name__ == "__main__":
    # --- START OF USER DATA ---
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
        "Булочка для гамбургера большой с кунжутом": {"Комбинирование": "0:21:00", "Смешивание": "0:10:30", "Формовка": "0:11:00", "Расстойка": "0:25:00", "Выпекание": "0:22:00", "Остывание": "0:45:00"},
        "Булочка для хотдога штучно": {"Комбинирование": "0:21:00", "Смешивание": "0:10:30", "Формовка": "0:11:00", "Расстойка": "0:25:00", "Выпекание": "0:22:00", "Остывание": "0:45:00"},
        "Сэндвич":              {"Комбинирование": "0:21:00", "Смешивание": "0:10:30", "Формовка": "0:11:00", "Расстойка": "0:25:00", "Выпекание": "0:22:00", "Остывание": "0:45:00"},
        "Хлеб «Тартин бездрожжевой»": {"Комбинирование": "0:21:00", "Смешивание": "0:10:30", "Формовка": "0:11:00", "Расстойка": "0:20:00", "Выпекание": "0:16:30", "Остывание": "1:00:00"},
        "Береке":               {"Комбинирование": "0:21:00", "Смешивание": "0:10:30", "Формовка": "0:11:00", "Расстойка": "0:20:00", "Выпекание": "0:16:30", "Остывание": "1:00:00"},
        "Баварский Деревенский Ржаной": {"Комбинирование": "0:21:00", "Смешивание": "0:10:30", "Формовка": "0:11:00", "Расстойка": "0:30:00", "Выпекание": "0:18:30", "Остывание": "1:30:00"},
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
        "Формовой": 1910, "Мини формовой": 306, "Бородинский": 488, "Домашний": 17, "Багет луковый": 33,
        "Багет новый": 219, "Багет отрубной": 49, "Премиум": 20, "Батон Верный": 54, "Батон Нарезной": 336,
        "Береке": 109, "Жайлы": 131, "Диета": 210, "Здоровье": 30, "Любимый": 459, "Немецкий хлеб": 15,
        "Отрубной (общий)": 161, "Плетенка": 94, "Семейный": 212, "Славянский": 6, "Зерновой Столичный": 16,
        "Сэндвич": 1866, "Хлеб «Тартин бездрожжевой»": 18, "Хлеб «Зерновой»": 113, "Чиабатта": 18,
        "Булочка для гамбургера большой с кунжутом": 160
    }
    machines_available = { "Комбинирование": 2, "Смешивание": 3, "Формовка": 2, "Расстойка": 8, "Выпекание": 6, "Остывание": 25 }
    BATCH_SIZE = 100
    script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
    OUTPUT_CSV_FILE = os.path.join(script_dir, 'dqn_production_schedule.csv')
    OUTPUT_TXT_FILE = os.path.join(script_dir, 'dqn_production_summary.txt')
    # --- END OF USER DATA ---
    
    scheduler = ProductionSchedulerDQN(tech_map_data, orders, machines_available, batch_size=BATCH_SIZE)
    scheduler.train(episodes=500)
    schedule_df = scheduler.generate_schedule()
    
    schedule_df.to_csv(OUTPUT_CSV_FILE, index=False, encoding='utf-8')
    print(f"\nРасписание сохранено в файл: {OUTPUT_CSV_FILE}")
    
    try:
        with open(OUTPUT_TXT_FILE, 'w', encoding='utf-8') as txtfile:
            txtfile.write("--- Сводка по Производственному Расписанию (DQN) ---\n\n")
            makespan = schedule_df['End_Time_Min'].max() if not schedule_df.empty else 0
            txtfile.write(f"Общее время производства (Makespan): {makespan:.2f} минут\n")
            import datetime
            tdelta = datetime.timedelta(minutes=float(makespan))
            txtfile.write(f"Общее время производства (формат): {str(tdelta)} (Дни, ЧЧ:ММ:СС)\n")
            total_batches = len(schedule_df['Batch_ID'].unique()) if not schedule_df.empty else 0
            txtfile.write(f"Всего партий: {total_batches}\n")
            txtfile.write(f"Всего задач (операций): {len(schedule_df)}\n\n")
            txtfile.write(f"Параметры модели (макс. время ожидания):\n")
            for (s1, s2), t in scheduler.env.max_wait_times.items():
                txtfile.write(f"  - {s1} -> {s2}: {t} мин\n")
            txtfile.write(f"\nДоступные ресурсы (машины):\n")
            for stage, count in machines_available.items():
                txtfile.write(f"  - {stage}: {count}\n")
            txtfile.write(f"\nЭпизодов обучения: 500\n")
            txtfile.write(f"Файл с детальным расписанием: {os.path.basename(OUTPUT_CSV_FILE)}\n")
        print(f"Сводная информация сохранена в файл: {OUTPUT_TXT_FILE}")
    except Exception as e:
        print(f"Ошибка записи TXT файла: {e}")
    
    print("\n--- Статистика DQN планировщика ---")
    if not schedule_df.empty:
        print(f"Всего задач запланировано: {len(schedule_df)}")
        print(f"Общее время производства: {schedule_df['End_Time_Min'].max()} минут")
        print(f"Средняя длительность задач: {schedule_df['Duration_Min'].mean():.1f} минут")
    else:
        print("Не удалось составить расписание.")
    
    print("\nПример расписания:")
    print(schedule_df.head(10))
    
    try:
        scheduler.plot_training_progress()
    except Exception as e:
        print(f"Matplotlib недоступен для визуализации: {e}")