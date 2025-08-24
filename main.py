import numpy as np
from walk import Simulation  # Ваш исходный файл с симуляцией
from energy_impact import ActiveInferenceAgent  # Файл с агентом
import matplotlib.pyplot as plt

def run_simulation():
    # Создаем симуляцию
    sim = Simulation()
    
    # Создаем агента
    agent = ActiveInferenceAgent(sim)
    
    print("🚀 Запуск симуляции с агентом Active Inference!")
    print("Настройки агента:")
    print(f"- Приоритет энергии: {agent.prior_preferences['energy_gain']}")
    print(f"- Приоритет исследования: {agent.prior_preferences['exploration']}")
    print(f"- Приоритет безопасности: {agent.prior_preferences['safety']}")
    print("\nНажмите Enter для начала...")
    input()
    
    # Запускаем эпизод
    results = agent.run_episode(max_steps=500, render=True)
    
    # Выводим результаты
    print("\n📊 Результаты эпизода:")
    print(f"Шагов выполнено: {results['step']}")
    print(f"Финальная энергия: {results['energy']:.1f}")
    print(f"Найдено еды: {results['food_found']}")
    print(f"Столкновений с опасностями: {results['dangers_encountered']}")
    print(f"Посещено клеток: {results['visited_cells']}")
    print(f"Точность модели мира: {results['world_model_accuracy']:.3f}")
    print(f"Веса энергетической модели: {results['energy_model_weights']}")
    
    # Визуализируем убеждения
    print("\n🖼️ Визуализация убеждений агента...")
    fig = agent.visualize_beliefs()
    plt.show()
    
    # График энергии по шагам
    plt.figure(figsize=(10, 5))
    plt.plot(agent.history['energies'])
    plt.title('Энергия агента по шагам')
    plt.xlabel('Шаг')
    plt.ylabel('Энергия')
    plt.grid(True)
    plt.show()

def debug_single_step():
    """Режим отладки по шагам"""
    sim = Simulation()
    agent = ActiveInferenceAgent(sim)
    
    print("🔍 Режим отладки по шагам")
    print("Нажмите Enter для выполнения шага, 'q' для выхода")
    
    step = 0
    while sim.agent_energy > 0:
        cmd = input(f"\nШаг {step} - Энергия: {sim.agent_energy:.1f} > ")
        if cmd.lower() == 'q':
            break
            
        action, prediction, error = agent.step()
        
        debug_info = agent.get_debug_info()
        print(f"Действие: {action}")
        print(f"Ошибка предсказания: {error['total_error']:.3f}")
        print(f"Позиция: {debug_info['position']}")
        print(f"Веса модели: {debug_info['energy_model_weights']}")
        
        step += 1

if __name__ == "__main__":
    print("Выберите режим:")
    print("1 - Полная симуляция")
    print("2 Пошаговая отладка")
    
    choice = input("Ваш выбор (1 или 2): ")
    
    if choice == "2":
        debug_single_step()
    else:
        run_simulation()