import numpy as np
import os
import time
from enum import Enum

# Размеры сетки
GRID_SIZE = 10

# Символы для отображения (только для визуализации, агент их не видит)
AGENT_SYMBOL = '@'
EMPTY_SYMBOL = '.'

class ObjectType(Enum):
    EMPTY = 0
    FOOD = 1
    DANGER = 2

class Simulation:
    def __init__(self):
        # Инициализируем сетку объектов
        self.grid = np.full((GRID_SIZE, GRID_SIZE), ObjectType.EMPTY)
        
        # Сенсорные свойства для каждого типа объекта
        self.object_properties = {
            ObjectType.EMPTY: {
                'name': 'empty',
                'texture': 0.2,   # гладкая
                'temperature': 0.5, # нейтральная
                'softness': 0.4,   # упругая
                'nutrition': 0     # никакой
            },
            ObjectType.FOOD: {
                'name': 'food',
                'texture': 0.7,   # шероховатая
                'temperature': 0.3, # прохладная
                'softness': 0.9,   # мягкая
                'nutrition': 25    # дает энергию
            },
            ObjectType.DANGER: {
                'name': 'danger', 
                'texture': 1.0,   # колючая
                'temperature': 0.9, # горячая
                'softness': 0.1,   # твердая
                'nutrition': -30   # отнимает энергию
            }
        }
        
        # Создаем агента в центре
        self.agent_pos = np.array([GRID_SIZE // 2, GRID_SIZE // 2])
        self.agent_energy = 100
        self.agent_max_energy = 100
        
        # Списки для позиций объектов
        self.food_positions = []
        self.danger_positions = []
        
        # Размещаем начальные объекты
        self.spawn_objects(ObjectType.FOOD, 3)
        self.spawn_objects(ObjectType.DANGER, 3)
        
        # История сенсорных опытов агента
        self.sensory_experiences = []
        
        # Для визуализации
        self.display_grid = np.full((GRID_SIZE, GRID_SIZE), EMPTY_SYMBOL)
        self.update_display_grid()
    
    def get_agent_sensory_experience(self):
        """
        Возвращает сенсорный опыт агента на текущей клетке.
        Агент не знает тип объекта, только его raw-свойства!
        """
        obj_type = self.grid[tuple(self.agent_pos)]
        properties = self.object_properties[obj_type]
        
        sensory_experience = {
            'texture': properties['texture'],
            'temperature': properties['temperature'], 
            'softness': properties['softness'],
            'position': tuple(self.agent_pos),
            'energy_change': 0  # будет заполнено после взаимодействия
        }
        
        return sensory_experience
        
    def spawn_objects(self, obj_type, count):
        """Размещает объекты на сетке в случайных свободных местах"""
        for _ in range(count):
            pos = np.random.randint(0, GRID_SIZE, 2)
            # Ищем свободную клетку
            while (tuple(pos) in [tuple(self.agent_pos)] + 
                   self.food_positions + self.danger_positions):
                pos = np.random.randint(0, GRID_SIZE, 2)
                
            self.grid[tuple(pos)] = obj_type
            if obj_type == ObjectType.FOOD:
                self.food_positions.append(tuple(pos))
            elif obj_type == ObjectType.DANGER:
                self.danger_positions.append(tuple(pos))
    
    def update_display_grid(self):
        """Обновляет сетку для отображения (только для визуализации)"""
        self.display_grid.fill(EMPTY_SYMBOL)
        
        # Размещаем еду
        for pos in self.food_positions:
            self.display_grid[pos] = 'F'
            
        # Размещаем опасности
        for pos in self.danger_positions:
            self.display_grid[pos] = 'D'
            
        # Размещаем агента
        self.display_grid[tuple(self.agent_pos)] = AGENT_SYMBOL
    
    def get_agent_sensory_experience(self):
        """
        Возвращает сенсорный опыт агента на текущей клетке.
        Агент не знает тип объекта, только его raw-свойства!
        """
        obj_type = self.grid[tuple(self.agent_pos)]
        properties = self.object_properties[obj_type]
        
        sensory_experience = {
            'texture': properties['texture'],
            'temperature': properties['temperature'], 
            'softness': properties['softness'],
            'position': tuple(self.agent_pos),
            'energy_change': 0  # будет заполнено после взаимодействия
        }
        
        return sensory_experience
    
    def get_agent_observation(self):
        """
        Возвращает наблюдения агента в виде двумерного массива 3x3,
        где центр - позиция агента, а вокруг - сенсорные свойства объектов
        """
        observation_grid = np.full((3, 3), None, dtype=object)

        # Центральная клетка - позиция агента (текущие ощущения)
        current_exp = self.get_agent_sensory_experience()
        observation_grid[1, 1] = {
            'texture': current_exp['texture'],
            'temperature': current_exp['temperature'],
            'softness': current_exp['softness'],
            'is_self': True
        }

        # Заполняем окружающие клетки
        for i in range(-1, 2):
            for j in range(-1, 2):
                if i == 0 and j == 0:
                    continue  # Пропускаем клетку самого агента

                check_pos = self.agent_pos + np.array([i, j])
                grid_i, grid_j = i + 1, j + 1  # Индексы в массиве 3x3

                # Проверяем границы
                if (0 <= check_pos[0] < GRID_SIZE and
                    0 <= check_pos[1] < GRID_SIZE):
                    obj_type = self.grid[tuple(check_pos)]
                    properties = self.object_properties[obj_type]
                    
                    observation_grid[grid_i, grid_j] = {
                        'texture': properties['texture'],
                        'temperature': properties['temperature'],
                        'softness': properties['softness'],
                        'is_self': False
                    }
                else:
                    # Стенка за пределами карты
                    observation_grid[grid_i, grid_j] = {
                        'texture': 1.0,  # очень шершавая
                        'temperature': 0.1,  # очень холодная
                        'softness': 0.0,  # очень твердая
                        'is_self': False,
                        'is_wall': True
                    }

        return observation_grid
    
    def move_agent(self, direction):
        """Двигает агента и обрабатывает взаимодействия"""
        old_pos = self.agent_pos.copy()
        old_energy = self.agent_energy
        
        # Двигаем агента
        if direction == 'w' and self.agent_pos[0] > 0:
            self.agent_pos[0] -= 1
        elif direction == 's' and self.agent_pos[0] < GRID_SIZE - 1:
            self.agent_pos[0] += 1
        elif direction == 'a' and self.agent_pos[1] > 0:
            self.agent_pos[1] -= 1
        elif direction == 'd' and self.agent_pos[1] < GRID_SIZE - 1:
            self.agent_pos[1] += 1
        elif direction == 'q' and self.agent_pos[0] > 0 and self.agent_pos[1] > 0:
            self.agent_pos[0] -= 1
            self.agent_pos[1] -= 1
        elif direction == 'e' and self.agent_pos[0] > 0 and self.agent_pos[1] < GRID_SIZE - 1:
            self.agent_pos[0] -= 1
            self.agent_pos[1] += 1
        elif direction == 'z' and self.agent_pos[0] < GRID_SIZE - 1 and self.agent_pos[1] > 0:
            self.agent_pos[0] += 1
            self.agent_pos[1] -= 1
        elif direction == 'c' and self.agent_pos[0] < GRID_SIZE - 1 and self.agent_pos[1] < GRID_SIZE - 1:
            self.agent_pos[0] += 1
            self.agent_pos[1] += 1
        elif direction == 'x':  # остаться на месте
            pass
        
        # Если позиция не изменилась - выходим
        if np.array_equal(old_pos, self.agent_pos) and direction != 'x':
            return False
        
        # Тратим энергию на движение (1 за шаг, 1.5 за диагональ)
        energy_cost = 1.5 if direction in ['q', 'e', 'z', 'c'] else 1
        self.agent_energy -= energy_cost
        
        # Обрабатываем столкновения (если переместились)
        if not np.array_equal(old_pos, self.agent_pos):
            self.handle_interaction(old_energy)
        
        # Обновляем дисплей
        self.update_display_grid()
        
        # Проверяем условия окончания игры
        if self.agent_energy <= 0:
            print("Game Over! Energy depleted.")
            return False
        elif self.agent_energy > self.agent_max_energy:
            self.agent_energy = self.agent_max_energy
        
        return True
    
    def handle_interaction(self, old_energy):
        """Обрабатывает взаимодействие с объектом на новой клетке"""
        current_pos = tuple(self.agent_pos)
        obj_type = self.grid[current_pos]
        properties = self.object_properties[obj_type]
        
        # Изменяем энергию агента
        self.agent_energy += properties['nutrition']
        
        # Записываем сенсорный опыт
        sensory_exp = self.get_agent_sensory_experience()
        sensory_exp['energy_change'] = self.agent_energy - old_energy
        self.sensory_experiences.append(sensory_exp)
        
        # Если это была еда или опасность - заменяем на пустоту
        if obj_type != ObjectType.EMPTY:
            self.grid[current_pos] = ObjectType.EMPTY
            if obj_type == ObjectType.FOOD:
                self.food_positions.remove(current_pos)
                self.spawn_objects(ObjectType.FOOD, 1)
            else:
                self.danger_positions.remove(current_pos)
                self.spawn_objects(ObjectType.DANGER, 1)
            
            print(f"{'Yum' if properties['nutrition'] > 0 else 'Ouch'}! {properties['nutrition']:+} energy")
    
    def render(self):
        """Отрисовывает сетку в консоли"""
        os.system('cls' if os.name == 'nt' else 'clear')
        print(f"Energy: {self.agent_energy:.1f}/100\n")
        
        # Рендерим основную сетку
        for i, row in enumerate(self.display_grid):
            print(' '.join(row))
        
        # Показываем текущие сенсорные ощущения агента
        current_exp = self.get_agent_sensory_experience()
        print(f"\nCurrent sensory experience:")
        print(f"  Texture: {current_exp['texture']:.2f} (0=smooth, 1=rough)")
        print(f"  Temperature: {current_exp['temperature']:.2f} (0=cold, 1=hot)")
        print(f"  Softness: {current_exp['softness']:.2f} (0=hard, 1=soft)")
        
        # Показываем историю сенсорного опыта (последние 3)
        print(f"\nRecent sensory experiences:")
        for exp in self.sensory_experiences[-3:]:
            print(f"  Pos{exp['position']}: Txt{exp['texture']:.2f}, "
                  f"Temp{exp['temperature']:.2f}, Soft{exp['softness']:.2f} "
                  f"-> Energy{exp['energy_change']:+}")
        
        print("\nControls: wasd/qezc (movement), x (stay), p (quit)")
    
    def get_available_actions(self):
        """Возвращает список доступных действий"""
        return ['move_up', 'move_down', 'move_left', 'move_right', 
                'move_up_left', 'move_up_right', 'move_down_left', 
                'move_down_right', 'stay']
    
    def step(self, action):
        """Выполняет действие в среде"""
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
        
        result = self.move_agent(action_map[action])
        observation = self.get_agent_observation()
        done = self.agent_energy <= 0
        reward = 0  # Для будущего использования с RL
        info = {
            'energy': self.agent_energy,
            'sensory_experience': self.get_agent_sensory_experience()
        }
        
        return observation, reward, done, info
    
    def reset(self):
        """Сбрасывает симуляцию в начальное состояние"""
        self.__init__()
        return self.get_agent_observation()

# Запуск симуляции для человека
if __name__ == "__main__":
    sim = Simulation()
    sim.render()
    
    while True:
        command = input().lower()
        
        if command == 'p':
            break
            
        if command in ['w', 'a', 's', 'd', 'q', 'e', 'z', 'c', 'x']:
            if not sim.move_agent(command):
                print("Can't move there!")
            else:
                sim.render()
        else:
            print("Invalid command! Use wasd/qezc/x/p")