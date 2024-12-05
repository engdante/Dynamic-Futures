import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

def simulate_market_with_profit_based_changes(
    initial_balance=100, leverage=10, initial_price=1,
    price_change_range=(-0.005, 0.005), steps=100,
    decision_threshold=0.01, rounding_precision=5):
    """
    Симулира пазарна динамика и коригира позициите базирано на печалба/загуба
    """
    # Инициализация за активната стратегия
    balance = initial_balance
    price = initial_price
    initial_position_value = balance * leverage
    position_size = initial_position_value / price
    history = []
    previous_position_value = initial_position_value
    
    # Инициализация за пасивната стратегия
    passive_position_size = initial_position_value / price
    initial_passive_value = passive_position_size * price
    passive_history = []
    price_history = [price]
    passive_liquidated = False

    for _ in range(steps):
        # Симулиране на движението на пазара
        price_change = round(np.random.uniform(*price_change_range), rounding_precision)
        new_price = round(price * (1 + price_change), rounding_precision)
        price_history.append(new_price)
        
        # Изчисления за активната стратегия
        current_position_value = position_size * new_price
        profit_loss = current_position_value - previous_position_value
        profit_loss_percentage = profit_loss / previous_position_value

        if abs(profit_loss_percentage) >= decision_threshold:
            if profit_loss > 0:
                additional_value = profit_loss * leverage
                additional_units = additional_value / new_price
                position_size += additional_units
            else:
                reduction_value = abs(profit_loss) * leverage
                reduction_units = reduction_value / new_price
                position_size = max(0, position_size - reduction_units)
            
            previous_position_value = position_size * new_price

        # Актуализиране на стойностите за активната стратегия
        balance = position_size * new_price / leverage
        price = new_price
        
        history.append((balance, position_size, price))
        
        # Изчисления за пасивната стратегия с коректен ефект на ливъриджа
        if not passive_liquidated:
            price_change_percentage = (new_price - initial_price) / initial_price
            passive_balance = initial_balance * (1 + price_change_percentage * leverage)
            
            # Проверка за ликвидация (загуба на целия баланс)
            if passive_balance <= 0:
                passive_balance = 0
                passive_position_size = 0
                passive_liquidated = True
        else:
            passive_balance = 0
            passive_position_size = 0
            
        passive_history.append((passive_balance, passive_position_size, price))
    
    return history, passive_history, price_history

class SimulationVisualizer:
    def __init__(self, params):
        self.params = params
        self.fig = plt.figure(figsize=(15, 12))
        
        # Създаване на място за бутона
        self.fig.subplots_adjust(bottom=0.1, top=0.95)
        
        # Създаване на графиките
        self.ax1 = plt.subplot(3, 1, 1)
        self.ax2 = plt.subplot(3, 1, 2)
        self.ax3 = plt.subplot(3, 1, 3)
        
        # Създаване на бутона
        ax_button = plt.axes([0.8, 0.02, 0.15, 0.04])
        self.button = Button(ax_button, 'Нова симулация')
        self.button.on_clicked(self.update)
        
        # Създаване на текстовото поле за резултати
        self.results_text = self.fig.text(0.02, 0.02, "", fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
        
        # Показване на първата симулация
        self.update(None)

    def update(self, event):
        # Изчистване на графиките
        self.ax1.clear()
        self.ax2.clear()
        self.ax3.clear()
        
        # Премахване на стария текст с резултати
        self.results_text.remove()
        
        # Нова симулация
        history, passive_history, price_history = simulate_market_with_profit_based_changes(**self.params)
        balances, positions, _ = zip(*history)
        passive_balances, passive_positions, _ = zip(*passive_history)
        
        # График на балансите
        self.ax1.plot(balances, label="Активна стратегия", color='blue')
        self.ax1.plot(passive_balances, label="Пасивна стратегия", color='red', linestyle='--')
        self.ax1.set_title("Баланс във времето")
        self.ax1.grid(True, which='both', linestyle='--', alpha=0.6)
        self.ax1.legend()
        
        # График на позициите
        self.ax2.plot(positions, label="Размер на позицията (активна)", color='orange')
        self.ax2.plot(passive_positions, label="Размер на позицията (пасивна)", color='red', linestyle='--')
        self.ax2.set_title("Размер на позицията във времето")
        self.ax2.grid(True, which='both', linestyle='--', alpha=0.6)
        self.ax2.legend()
        
        # График на цената
        self.ax3.plot(price_history, label="Цена", color='green')
        self.ax3.set_title("Пазарна цена във времето")
        self.ax3.grid(True, which='both', linestyle='--', alpha=0.6)
        self.ax3.legend()
        
        # Добавяне на резултатите в по-компактен формат
        final_balance = balances[-1]
        final_passive = passive_balances[-1]
        profit_active = ((final_balance - self.params['initial_balance']) / self.params['initial_balance']) * 100
        profit_passive = ((final_passive - self.params['initial_balance']) / self.params['initial_balance']) * 100
        
        results_text = f"""Резултати:
        Активна стратегия: Краен баланс: ${final_balance:.2f} / Печалба: {profit_active:.2f}%
        Пасивна стратегия: Краен баланс: ${final_passive:.2f} / Печалба: {profit_passive:.2f}%
        Цена: Начална: ${price_history[0]:.4f} / Крайна: ${price_history[-1]:.4f} / Промяна: {((price_history[-1] - price_history[0]) / price_history[0] * 100):.2f}%"""
        
        self.results_text = self.fig.text(0.02, 0.02, results_text, fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
        
        plt.draw()

# Параметри
params = {
    "initial_balance": 100,
    "leverage": 3,
    "initial_price": 1,
    "price_change_range": (-0.0045, 0.005),
    "steps": 1000,
    "decision_threshold": 0.01,
    "rounding_precision": 5,
}

# Създаване на визуализатора
viz = SimulationVisualizer(params)
plt.show()