import numpy as np
import matplotlib.pyplot as plt
from scipy.special import softmax
import time

class EnergyImpactModel:
    """Модель влияния сенсорных свойств на энергию"""
    def __init__(self):
        self.sensory_weights = {
            'texture': 0.0,
            'temperature': 0.0, 
            'softness': 0.0
        }
        self.interaction_history = []
        
    def update(self, sensory_input, energy_change, learning_rate=0.1):
        """Обновляет веса на основе опыта"""
        for key in self.sensory_weights:
            self.sensory_weights[key] += learning_rate * sensory_input[key] * energy_change
        
    def predict_energy_change(self, sensory_input):
        """Предсказывает изменение энергии на основе сенсорных свойств"""
        prediction = 0
        # Игнорируем служебные поля
        for key in ['texture', 'temperature', 'softness']:
            if key in sensory_input and key in self.sensory_weights:
                prediction += self.sensory_weights[key] * sensory_input[key]
        return prediction

class WorldModel:
    """Контекстуальная модель мира (приобретенные знания)"""
    def __init__(self, grid_size):
        self.grid_size = grid_size
        # Карта ожидаемых сенсорных свойств
        self.sensory_map = np.full((grid_size, grid_size, 3), 0.5)  # texture, temp, softness
        # Неопределенность для каждой позиции
        self.uncertainty_map = np.ones((grid_size, grid_size))
        # Посещенные позиции
        self.visited_positions = set()
        
    def update_sensory_expectation(self, position, sensory_data, learning_rate=0.3):
        """Обновляет ожидания для позиции"""
        x, y = position
        self.visited_positions.add(position)
        
        # Обновляем сенсорные ожидания
        self.sensory_map[x, y] = (1 - learning_rate) * self.sensory_map[x, y] + learning_rate * np.array([
            sensory_data['texture'], 
            sensory_data['temperature'], 
            sensory_data['softness']
        ])
        
        # Уменьшаем неопределенность
        self.uncertainty_map[x, y] *= (1 - learning_rate)
        
    def get_sensory_prediction(self, position):
        """Возвращает предсказанные сенсорные свойства для позиции"""
        x, y = position
        if not (0 <= x < self.grid_size and 0 <= y < self.grid_size):
            return {'texture': 1.0, 'temperature': 0.1, 'softness': 0.0, 'is_wall': True}
        
        return {
            'texture': self.sensory_map[x, y, 0],
            'temperature': self.sensory_map[x, y, 1],
            'softness': self.sensory_map[x, y, 2],
            'uncertainty': self.uncertainty_map[x, y]
        }

class ActiveInferenceAgent:
    """Агент, реализующий активный вывод"""
    
    def __init__(self, simulation):
        self.sim = simulation
        self.grid_size = simulation.grid.shape[0]
        
        # Иерархическая модель мира
        self.world_model = WorldModel(self.grid_size)
        self.energy_model = EnergyImpactModel()
        
        # Текущие убеждения и состояние
        self.current_position = tuple(simulation.agent_pos)
        self.current_energy = simulation.agent_energy
        self.current_sensory = None
        
        # Фундаментальные приоритеты
        self.prior_preferences = {
            'energy_gain': 2.0,      # Сильное предпочтение к получению энергии
            'energy_maintenance': 1.0, # Предпочтение к сохранению энергии
            'exploration': 0.3,      # Умеренное предпочтение к исследованию
            'safety': 1.5           # Предпочтение к безопасности
        }
        
        # История для отладки
        self.history = {
            'actions': [],
            'predictions': [],
            'errors': [],
            'energies': [],
            'positions': [],
            'belief_updates': []
        }
        
        # Статистика
        self.step_count = 0
        self.food_found = 0
        self.dangers_encountered = 0
        
    def get_available_actions(self):
        """Возвращает доступные действия с учетом границ"""
        actions = []
        x, y = self.current_position
        
        # Кардинальные направления
        if x > 0: actions.append('move_up')
        if x < self.grid_size - 1: actions.append('move_down')
        if y > 0: actions.append('move_left')
        if y < self.grid_size - 1: actions.append('move_right')
        
        # Диагональные направления
        if x > 0 and y > 0: actions.append('move_up_left')
        if x > 0 and y < self.grid_size - 1: actions.append('move_up_right')
        if x < self.grid_size - 1 and y > 0: actions.append('move_down_left')
        if x < self.grid_size - 1 and y < self.grid_size - 1: actions.append('move_down_right')
        
        actions.append('stay')
        return actions
    
    def predict_position(self, action):
        """Предсказывает новую позицию после действия"""
        x, y = self.current_position
        
        action_map = {
            'move_up': (-1, 0),
            'move_down': (1, 0),
            'move_left': (0, -1),
            'move_right': (0, 1),
            'move_up_left': (-1, -1),
            'move_up_right': (-1, 1),
            'move_down_left': (1, -1),
            'move_down_right': (1, 1),
            'stay': (0, 0)
        }
        
        dx, dy = action_map[action]
        new_x, new_y = x + dx, y + dy
        
        # Проверяем границы
        if not (0 <= new_x < self.grid_size and 0 <= new_y < self.grid_size):
            return self.current_position  # Не можем выйти за границы
            
        return (new_x, new_y)
    
    def generate_predictions(self):
        """Генерирует предсказания для всех возможных действий"""
        predictions = {}
        
        for action in self.get_available_actions():
            new_position = self.predict_position(action)
            sensory_pred = self.world_model.get_sensory_prediction(new_position)
            
            # Предсказание изменения энергии
            energy_cost = 1.5 if 'diagonal' in action else 1.0 if action != 'stay' else 0.0
            energy_gain_pred = self.energy_model.predict_energy_change(sensory_pred)
            net_energy_change = energy_gain_pred - energy_cost
            
            predictions[action] = {
                'position': new_position,
                'sensory': sensory_pred,
                'energy_change': net_energy_change,
                'uncertainty': sensory_pred.get('uncertainty', 1.0)
            }
        
        return predictions
    
    def compute_prediction_errors(self, actual_observation, predictions):
        """Вычисляет ошибки предсказания"""
        errors = {}
        
        for action, prediction in predictions.items():
            # Ошибка сенсорного предсказания
            sensory_error = 0
            for key in ['texture', 'temperature', 'softness']:
                if key in actual_observation and key in prediction['sensory']:
                    error = abs(actual_observation[key] - prediction['sensory'][key])
                    sensory_error += error
            
            # Ошибка энергетического предсказания (если это текущая позиция)
            energy_error = 0
            if action == 'stay':
                actual_energy_change = self.sim.agent_energy - self.current_energy
                energy_error = abs(actual_energy_change - prediction['energy_change'])
            
            # Общая ошибка
            errors[action] = {
                'sensory_error': sensory_error,
                'energy_error': energy_error,
                'total_error': sensory_error + energy_error * 2.0  # Вес для энергии
            }
        
        return errors
    
    def update_beliefs(self, actual_observation, actual_energy_change):
        """Обновляет убеждения на основе реальных наблюдений"""
        # Обновляем мировую модель
        self.world_model.update_sensory_expectation(
            self.current_position, 
            actual_observation,
            learning_rate=0.4
        )
        
        # Обновляем энергетическую модель
        self.energy_model.update(
            actual_observation,
            actual_energy_change,
            learning_rate=0.2
        )
        
        # Записываем обновление
        self.history['belief_updates'].append({
            'position': self.current_position,
            'sensory_actual': actual_observation,
            'energy_change': actual_energy_change
        })
    
    def calculate_expected_free_energy(self, predictions, errors):
        """Вычисляет ожидаемую свободную энергию для каждого действия"""
        efe_scores = {}
        
        for action in self.get_available_actions():
            pred = predictions[action]
            error = errors[action]
            
            # Компонент точности (минимизация ошибки)
            accuracy_term = -error['total_error']
            
            # Компонент энергии (максимизация ожидаемого прироста)
            energy_term = pred['energy_change'] * self.prior_preferences['energy_gain']
            
            # Компонент исследования (максимизация уменьшения неопределенности)
            exploration_term = pred['uncertainty'] * self.prior_preferences['exploration']
            
            # Компонент безопасности (избегание неизвестного с высокой неопределенностью)
            safety_term = -pred['uncertainty'] * self.prior_preferences['safety'] if pred['uncertainty'] > 0.7 else 0
            
            # Итоговый score (минимизируем EFE = максимизируем -EFE)
            efe_scores[action] = - (accuracy_term + energy_term + exploration_term + safety_term)
        
        return efe_scores
    
    def select_action(self, efe_scores):
        """Выбирает действие на основе ожидаемой свободной энергии"""
        # Преобразуем в вероятности с помощью softmax
        actions = list(efe_scores.keys())
        scores = np.array([efe_scores[action] for action in actions])
        probabilities = softmax(-scores)  # Меньший EFE -> высокая вероятность
        
        # Выбираем действие вероятностно (можно выбрать argmax для жадного поведения)
        selected_action = np.random.choice(actions, p=probabilities)
        
        return selected_action, probabilities
    
    def step(self):
        """Выполняет один шаг активного вывода"""
        self.step_count += 1
        
        # Сохраняем текущее состояние
        self.current_position = tuple(self.sim.agent_pos)
        self.current_energy = self.sim.agent_energy
        self.current_sensory = self.sim.get_agent_sensory_experience()
        
        # Шаг 1: Генерация предсказаний
        predictions = self.generate_predictions()
        
        # Шаг 2: Получение реальных наблюдений
        actual_observation = self.current_sensory
        actual_observation.pop('energy_change', None)  # Убираем energy_change для сравнения
        actual_observation.pop('position', None)
        
        # Шаг 3: Вычисление ошибок предсказания
        errors = self.compute_prediction_errors(actual_observation, predictions)
        
        # Шаг 4: Обновление убеждений
        actual_energy_change = self.sim.agent_energy - self.current_energy
        self.update_beliefs(actual_observation, actual_energy_change)
        
        # Шаг 5: Расчет ожидаемой свободной энергии
        efe_scores = self.calculate_expected_free_energy(predictions, errors)
        
        # Шаг 6: Выбор и выполнение действия
        selected_action, action_probabilities = self.select_action(efe_scores)
        
        # Выполняем действие
        action_map = {
            'move_up': 'w',
            'move_down': 's',
            'move_left': 'a',
            'move_right': 'd',
            'move_up_left': 'q',
            'move_up_right': 'e',
            'move_down_left': 'z',
            'move_down_right': 'c',
            'stay': 'x'
        }
        
        self.sim.move_agent(action_map[selected_action])
        
        # Обновляем статистику
        if actual_energy_change > 0:
            self.food_found += 1
        elif actual_energy_change < 0:
            self.dangers_encountered += 1
        
        # Сохраняем историю для отладки
        self.history['actions'].append(selected_action)
        self.history['predictions'].append(predictions[selected_action])
        self.history['errors'].append(errors[selected_action])
        self.history['energies'].append(self.sim.agent_energy)
        self.history['positions'].append(self.current_position)
        
        return selected_action, predictions[selected_action], errors[selected_action]
    
    def get_debug_info(self):
        """Возвращает отладочную информацию"""
        return {
            'step': self.step_count,
            'position': self.current_position,
            'energy': self.sim.agent_energy,
            'food_found': self.food_found,
            'dangers_encountered': self.dangers_encountered,
            'world_model_accuracy': self.calculate_world_model_accuracy(),
            'energy_model_weights': self.energy_model.sensory_weights,
            'visited_cells': len(self.world_model.visited_positions),
            'average_uncertainty': np.mean(self.world_model.uncertainty_map)
        }
    
    def calculate_world_model_accuracy(self):
        """Оценивает точность мировой модели"""
        if not self.history['belief_updates']:
            return 0.0
        
        recent_updates = self.history['belief_updates'][-10:]  # Последние 10 обновлений
        if not recent_updates:
            return 0.0
            
        errors = []
        for update in recent_updates:
            pred = self.world_model.get_sensory_prediction(update['position'])
            error = 0
            for key in ['texture', 'temperature', 'softness']:
                error += abs(pred[key] - update['sensory_actual'][key])
            errors.append(error / 3.0)
        
        return 1.0 - np.mean(errors) if errors else 0.0
    
    def visualize_beliefs(self):
        """Визуализирует текущие убеждения агента"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Карта текстуры
        im1 = axes[0, 0].imshow(self.world_model.sensory_map[:, :, 0], cmap='viridis', vmin=0, vmax=1)
        axes[0, 0].set_title('Texture Beliefs')
        plt.colorbar(im1, ax=axes[0, 0])
        
        # Карта температуры
        im2 = axes[0, 1].imshow(self.world_model.sensory_map[:, :, 1], cmap='coolwarm', vmin=0, vmax=1)
        axes[0, 1].set_title('Temperature Beliefs')
        plt.colorbar(im2, ax=axes[0, 1])
        
        # Карта мягкости
        im3 = axes[1, 0].imshow(self.world_model.sensory_map[:, :, 2], cmap='plasma', vmin=0, vmax=1)
        axes[1, 0].set_title('Softness Beliefs')
        plt.colorbar(im3, ax=axes[1, 0])
        
        # Карта неопределенности
        im4 = axes[1, 1].imshow(self.world_model.uncertainty_map, cmap='gray_r', vmin=0, vmax=1)
        axes[1, 1].set_title('Uncertainty Map')
        plt.colorbar(im4, ax=axes[1, 1])
        
        # Добавляем позицию агента
        for ax in axes.flat:
            ax.plot(self.current_position[1], self.current_position[0], 'ro', markersize=10)
        
        plt.tight_layout()
        return fig
    
    def run_episode(self, max_steps=1000, render=False):
        """Запускает полный эпизод"""
        for step in range(max_steps):
            action, prediction, error = self.step()
            
            if render and step % 10 == 0:
                debug_info = self.get_debug_info()
                print(f"Step {step}: {action}, Energy: {debug_info['energy']:.1f}, "
                      f"Food: {debug_info['food_found']}, Dangers: {debug_info['dangers_encountered']}")
            
            if self.sim.agent_energy <= 0:
                print("Energy depleted! Episode ended.")
                break
                
        return self.get_debug_info()