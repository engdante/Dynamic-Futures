import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, RadioButtons, TextBox
import requests
from functools import lru_cache


@lru_cache(maxsize=32)
def get_historical_prices(symbol='BTCUSDT', interval='1h', limit=1000):
    base_url = "https://api.binance.com/api/v3/klines"
    params = {
        'symbol': symbol.upper(),
        'interval': interval,
        'limit': limit
    }

    try:
        response = requests.get(base_url, params=params)
        data = response.json()
        prices = np.array([float(d[4]) for d in data])
        return prices
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None


def simulate_market_with_profit_based_changes(
        initial_balance=100, leverage=10,
        price_change_range=(-0.005, 0.005), steps=100,
        start_step=0, end_step=None,
        profit_threshold=0.01, loss_threshold=0.01,
        profit_coefficient=1.0, loss_coefficient=1.0,
        rounding_precision=5,
        use_historical=True, symbol='BTCUSDT'):

    if use_historical:
        price_history = get_historical_prices(symbol, limit=steps)
        if price_history is None:
            return None, None, None

        # Обработка на start_step и end_step
        if end_step is None or end_step > len(price_history):
            end_step = len(price_history)
        price_history = price_history[start_step:end_step]
        if len(price_history) < 2:
            return None, None, None

        initial_price = price_history[0]
        steps = len(price_history) - 1
    else:
        initial_price = 1
        price = initial_price
        price_history = np.zeros(steps + 1)
        price_history[0] = price
        for i in range(steps):
            price_change = round(np.random.uniform(
                *price_change_range), rounding_precision)
            price = round(price * (1 + price_change), rounding_precision)
            price_history[i + 1] = price
        price_history = price_history[start_step:end_step]
        steps = len(price_history) - 1

    balance = initial_balance
    position_size = (balance * leverage) / initial_price
    history = np.zeros((steps + 1, 3))
    history[0] = [balance, position_size, initial_price]
    previous_position_value = position_size * initial_price

    passive_position_size = (initial_balance * leverage) / initial_price
    passive_history = np.zeros((steps + 1, 3))
    passive_history[0] = [initial_balance,
                          passive_position_size, initial_price]
    passive_liquidated = False

    for i in range(steps):
        current_price = price_history[i + 1]

        current_position_value = position_size * current_price
        profit_loss = current_position_value - previous_position_value
        profit_loss_percentage = profit_loss / previous_position_value

        if profit_loss_percentage >= profit_threshold/100:  # Profit case
            additional_value = profit_loss * leverage * profit_coefficient
            additional_units = additional_value / current_price
            position_size += additional_units
            previous_position_value = position_size * current_price
        elif profit_loss_percentage <= -loss_threshold/100:  # Loss case
            reduction_value = abs(profit_loss) * leverage * loss_coefficient
            reduction_units = reduction_value / current_price
            position_size = max(0, position_size - reduction_units)
            previous_position_value = position_size * current_price

        balance = position_size * current_price / leverage
        history[i + 1] = [balance, position_size, current_price]

        if not passive_liquidated:
            price_change_percentage = (
                current_price - initial_price) / initial_price
            passive_balance = initial_balance * \
                (1 + price_change_percentage * leverage)

            if passive_balance <= 0:
                passive_balance = 0
                passive_position_size = 0
                passive_liquidated = True
        else:
            passive_balance = 0
            passive_position_size = 0

        passive_history[i + 1] = [passive_balance,
                                  passive_position_size, current_price]

    return history, passive_history, price_history


class SimulationVisualizer:
    def __init__(self, params):
        self.params = params
        self.fig = plt.figure(figsize=(15, 12))

        self.fig.subplots_adjust(bottom=0.2, top=0.95)

        self.ax1 = plt.subplot(3, 1, 1)
        self.ax2 = plt.subplot(3, 1, 2)
        self.ax3 = plt.subplot(3, 1, 3)

        # Radio buttons
        ax_radio = plt.axes([0.05, 0.05, 0.10, 0.08])
        self.radio = RadioButtons(ax_radio, ('Historical', 'Random'))

        # Input fields
        # Symbol
        ax_symbol = plt.axes([0.22, 0.05, 0.10, 0.02])
        self.symbol_input = TextBox(ax_symbol, 'Symbol:', initial='BTCUSDT')

        # Start Step
        ax_start_step = plt.axes([0.38, 0.01, 0.10, 0.02])
        self.start_step_input = TextBox(
            ax_start_step, 'Start Step:', initial='0')

        # End Step
        ax_end_step = plt.axes([0.54, 0.01, 0.10, 0.02])
        self.end_step_input = TextBox(
            ax_end_step, 'End Step:', initial=str(params['steps']))

        # Initial Balance
        ax_balance = plt.axes([0.22, 0.09, 0.10, 0.02])
        self.balance_input = TextBox(
            ax_balance, 'Balance:', initial=str(params['initial_balance']))

        # Leverage
        ax_leverage = plt.axes([0.22, 0.13, 0.10, 0.02])
        self.leverage_input = TextBox(
            ax_leverage, 'Leverage:', initial=str(params['leverage']))

        # Steps
        ax_steps = plt.axes([0.38, 0.05, 0.10, 0.02])
        self.steps_input = TextBox(
            ax_steps, 'Steps:', initial=str(params['steps']))

        # Profit Threshold
        ax_profit_threshold = plt.axes([0.38, 0.09, 0.10, 0.02])
        self.profit_threshold_input = TextBox(ax_profit_threshold, 'Profit %:',
                                              initial=str(params['profit_threshold'] * 100))

        # Loss Threshold
        ax_loss_threshold = plt.axes([0.38, 0.13, 0.10, 0.02])
        self.loss_threshold_input = TextBox(ax_loss_threshold, 'Loss %:',
                                            initial=str(params['loss_threshold'] * 100))

        # Profit Coefficient
        ax_profit_coef = plt.axes([0.54, 0.09, 0.10, 0.02])
        self.profit_coef_input = TextBox(ax_profit_coef, 'Profit Coef:',
                                         initial=str(params['profit_coefficient']))

        # Loss Coefficient
        ax_loss_coef = plt.axes([0.54, 0.13, 0.10, 0.02])
        self.loss_coef_input = TextBox(ax_loss_coef, 'Loss Coef:',
                                       initial=str(params['loss_coefficient']))

        # Rounding Precision
        ax_precision = plt.axes([0.54, 0.05, 0.10, 0.02])
        self.precision_input = TextBox(ax_precision, 'Precision:',
                                       initial=str(params['rounding_precision']))

        # New simulation button
        ax_button = plt.axes([0.75, 0.02, 0.15, 0.04])
        self.button = Button(ax_button, 'Нова симулация')
        self.button.on_clicked(self.update)

        self.results_text = None
        self.use_historical = True

        # Initial plot
        self.update(None)

    def update(self, event):
        try:
            # Update parameters from input fields
            self.use_historical = self.radio.value_selected == 'Historical'
            self.params.update({
                'symbol': self.symbol_input.text.upper(),
                'initial_balance': float(self.balance_input.text),
                'leverage': float(self.leverage_input.text),
                'steps': int(self.steps_input.text),
                'start_step': int(self.start_step_input.text),
                'end_step': int(self.end_step_input.text),
                # Вече в проценти
                'profit_threshold': float(self.profit_threshold_input.text),
                # Вече в проценти
                'loss_threshold': float(self.loss_threshold_input.text),
                'profit_coefficient': float(self.profit_coef_input.text),
                'loss_coefficient': float(self.loss_coef_input.text),
                'rounding_precision': int(self.precision_input.text),
                'use_historical': self.use_historical
            })

            # Clear plots
            self.ax1.clear()
            self.ax2.clear()
            self.ax3.clear()

            if self.results_text is not None:
                self.results_text.remove()

            # Run simulation with updated parameters
            history, passive_history, price_history = simulate_market_with_profit_based_changes(
                **self.params)

            if history is None:
                error_str = "Error fetching data for {}".format(
                    self.params['symbol'])
                self.results_text = self.fig.text(0.4, 0.02, error_str, fontsize=10,
                                                  bbox=dict(facecolor='red', alpha=0.8))
                plt.draw()
                return

            # Plot results using numpy arrays for better performance
            self.ax1.plot(
                history[:, 0], label="Активна стратегия", color='blue')
            self.ax1.plot(passive_history[:, 0], label="Пасивна стратегия",
                          color='red', linestyle='--')
            self.ax1.set_title("Баланс във времето")
            self.ax1.grid(True)
            self.ax1.legend()

            self.ax2.plot(history[:, 1], label="Размер на позицията (активна)",
                          color='orange')
            self.ax2.plot(passive_history[:, 1], label="Размер на позицията (пасивна)",
                          color='red', linestyle='--')
            self.ax2.set_title("Размер на позицията във времето")
            self.ax2.grid(True)
            self.ax2.legend()

            self.ax3.plot(price_history, label="Цена", color='green')
            if self.use_historical:
                title_prefix = f"Историческа цена"
                title_symbol = self.params['symbol']
            else:
                title_prefix = f"Симулирана цена"
                title_symbol = ""
            self.ax3.set_title(f"{title_prefix} {title_symbol}")
            self.ax3.grid(True)
            self.ax3.legend()

            # Update results text
            final_balance = history[-1, 0]
            final_passive = passive_history[-1, 0]
            profit_active = ((final_balance - self.params['initial_balance']) /
                             self.params['initial_balance']) * 100
            profit_passive = ((final_passive - self.params['initial_balance']) /
                              self.params['initial_balance']) * 100

            if self.use_historical:
                mode_text = f"Historical {self.params['symbol']}"
            else:
                mode_text = "Random simulation"

            results_text = f"""Режим: {mode_text}
            Активна стратегия: Баланс: ${final_balance:.2f} / Печалба: {profit_active:.2f}%
            Пасивна стратегия: Баланс: ${final_passive:.2f} / Печалба: {profit_passive:.2f}%
            Цена: Начална: ${price_history[0]:.4f} / Крайна: ${price_history[-1]:.4f} /
            Промяна: {((price_history[-1] - price_history[0]) / price_history[0] * 100):.2f}%"""

            self.results_text = self.fig.text(0.70, 0.09, results_text, fontsize=8,
                                              bbox=dict(facecolor='white', alpha=0.8))

            plt.draw()

        except ValueError as e:
            if self.results_text is not None:
                self.results_text.remove()
            self.results_text = self.fig.text(0.4, 0.02, f"Invalid input: {str(e)}",
                                              fontsize=10, bbox=dict(facecolor='red', alpha=0.8))
            plt.draw()


# Initial parameters
params = {
    "initial_balance": 100,
    "leverage": 3,
    "price_change_range": (-0.005, 0.005),
    "steps": 1000,
    "start_step": 0,
    "end_step": 1000,
    "profit_threshold": 0.01,
    "loss_threshold": 0.01,
    "profit_coefficient": 1.0,
    "loss_coefficient": 1.0,
    "rounding_precision": 5,
    "symbol": "BTCUSDT",
    "use_historical": True
}

viz = SimulationVisualizer(params)
plt.show()
