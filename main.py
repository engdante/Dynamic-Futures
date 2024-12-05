import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, RadioButtons, TextBox
import requests

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
        prices = [float(d[4]) for d in data]
        return prices
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None

def simulate_market_with_profit_based_changes(
    initial_balance=100, leverage=10,
    price_change_range=(-0.005, 0.005), steps=100,
    decision_threshold=0.01, rounding_precision=5,
    use_historical=True, symbol='BTCUSDT'):

    if use_historical:
        price_history = get_historical_prices(symbol, limit=steps)
        if price_history is None:
            return None, None, None
        initial_price = price_history[0]
        steps = len(price_history) - 1
    else:
        initial_price = 1
        price = initial_price
        price_history = [price]
        for _ in range(steps):
            price_change = round(np.random.uniform(*price_change_range), rounding_precision)
            price = round(price * (1 + price_change), rounding_precision)
            price_history.append(price)

    balance = initial_balance
    position_size = (balance * leverage) / initial_price
    history = []
    previous_position_value = position_size * initial_price

    passive_position_size = (initial_balance * leverage) / initial_price
    passive_history = []
    passive_liquidated = False

    for i in range(steps):
        current_price = price_history[i + 1]

        current_position_value = position_size * current_price
        profit_loss = current_position_value - previous_position_value
        profit_loss_percentage = profit_loss / previous_position_value

        if abs(profit_loss_percentage) >= decision_threshold:
            if profit_loss > 0:
                additional_value = profit_loss * leverage
                additional_units = additional_value / current_price
                position_size += additional_units
            else:
                reduction_value = abs(profit_loss) * leverage
                reduction_units = reduction_value / current_price
                position_size = max(0, position_size - reduction_units)

            previous_position_value = position_size * current_price

        balance = position_size * current_price / leverage
        history.append((balance, position_size, current_price))

        if not passive_liquidated:
            price_change_percentage = (current_price - initial_price) / initial_price
            passive_balance = initial_balance * (1 + price_change_percentage * leverage)

            if passive_balance <= 0:
                passive_balance = 0
                passive_position_size = 0
                passive_liquidated = True
        else:
            passive_balance = 0
            passive_position_size = 0

        passive_history.append((passive_balance, passive_position_size, current_price))

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
        self.radio.on_clicked(self.mode_changed)

        # Input fields
        # Symbol
        ax_symbol = plt.axes([0.22, 0.05, 0.10, 0.02])
        self.symbol_input = TextBox(ax_symbol, 'Symbol:', initial='BTCUSDT')
        self.symbol_input.on_submit(self.params_changed)

        # Initial Balance
        ax_balance = plt.axes([0.22, 0.09, 0.10, 0.02])
        self.balance_input = TextBox(ax_balance, 'Balance:', initial=str(params['initial_balance']))
        self.balance_input.on_submit(self.params_changed)

        # Leverage
        ax_leverage = plt.axes([0.22, 0.13, 0.10, 0.02])
        self.leverage_input = TextBox(ax_leverage, 'Leverage:', initial=str(params['leverage']))
        self.leverage_input.on_submit(self.params_changed)

        # Steps
        ax_steps = plt.axes([0.38, 0.05, 0.10, 0.02])
        self.steps_input = TextBox(ax_steps, 'Steps:', initial=str(params['steps']))
        self.steps_input.on_submit(self.params_changed)

        # Decision Threshold
        ax_threshold = plt.axes([0.38, 0.09, 0.10, 0.02])
        self.threshold_input = TextBox(ax_threshold, 'Threshold:', initial=str(params['decision_threshold']))
        self.threshold_input.on_submit(self.params_changed)

        # Rounding Precision
        ax_precision = plt.axes([0.38, 0.13, 0.10, 0.02])
        self.precision_input = TextBox(ax_precision, 'Precision:', initial=str(params['rounding_precision']))
        self.precision_input.on_submit(self.params_changed)

        # New simulation button
        ax_button = plt.axes([0.75, 0.02, 0.15, 0.04])
        self.button = Button(ax_button, 'Нова симулация')
        self.button.on_clicked(self.update)

        self.results_text = None

        self.use_historical = True
        self.update(None)

    def params_changed(self, _):
        try:
            self.params.update({
                'symbol': self.symbol_input.text.upper(),
                'initial_balance': float(self.balance_input.text),
                'leverage': float(self.leverage_input.text),
                'steps': int(self.steps_input.text),
                'decision_threshold': float(self.threshold_input.text),
                'rounding_precision': int(self.precision_input.text)
            })
            print(f"Updated params: {self.params}")  # Debugging line
            self.update(None)
        except ValueError as e:
            if self.results_text is not None:
                self.results_text.set_text(f"Invalid input: {str(e)}")
            plt.draw()

    def mode_changed(self, label):
        self.use_historical = (label == 'Historical')
        self.update(None)

    def update(self, event):
        self.ax1.clear()
        self.ax2.clear()
        self.ax3.clear()

        if self.results_text is not None:
            self.results_text.remove()

        params = self.params.copy()
        params['use_historical'] = self.use_historical
        params['symbol'] = self.symbol_input.text.upper()

        history, passive_history, price_history = simulate_market_with_profit_based_changes(**params)

        if history is None:
            self.results_text = self.fig.text(0.4, 0.02, f"Error fetching data for {params['symbol']}",
                                            fontsize=10, bbox=dict(facecolor='red', alpha=0.8))
            plt.draw()
            return

        balances, positions, _ = zip(*history)
        passive_balances, passive_positions, _ = zip(*passive_history)

        self.ax1.plot(balances, label="Активна стратегия", color='blue')
        self.ax1.plot(passive_balances, label="Пасивна стратегия", color='red', linestyle='--')
        self.ax1.set_title("Баланс във времето")
        self.ax1.grid(True)
        self.ax1.legend()

        self.ax2.plot(positions, label="Размер на позицията (активна)", color='orange')
        self.ax2.plot(passive_positions, label="Размер на позицията (пасивна)", color='red', linestyle='--')
        self.ax2.set_title("Размер на позицията във времето")
        self.ax2.grid(True)
        self.ax2.legend()

        self.ax3.plot(price_history, label="Цена", color='green')
        title_prefix = f"{'Историческа' if self.use_historical else 'Симулирана'} цена"
        title_symbol = f" - {params['symbol']}" if self.use_historical else ""
        self.ax3.set_title(f"{title_prefix}{title_symbol}")
        self.ax3.grid(True)
        self.ax3.legend()

        final_balance = balances[-1]
        final_passive = passive_balances[-1]
        profit_active = ((final_balance - self.params['initial_balance']) / self.params['initial_balance']) * 100
        profit_passive = ((final_passive - self.params['initial_balance']) / self.params['initial_balance']) * 100

        mode_text = f"Historical {params['symbol']}" if self.use_historical else "Random simulation"
        results_text = f"""Режим: {mode_text}
        Активна стратегия: Баланс: ${final_balance:.2f} / Печалба: {profit_active:.2f}%
        Пасивна стратегия: Баланс: ${final_passive:.2f} / Печалба: {profit_passive:.2f}%
        Цена: Начална: ${price_history[0]:.4f} / Крайна: ${price_history[-1]:.4f} / Промяна: {((price_history[-1] - price_history[0]) / price_history[0] * 100):.2f}%"""

        self.results_text = self.fig.text(0.53, 0.1, results_text, fontsize=10, bbox=dict(facecolor='white', alpha=0.8))

        plt.draw()

params = {
    "initial_balance": 100,
    "leverage": 3,
    "price_change_range": (-0.005, 0.005),
    "steps": 1000,
    "decision_threshold": 0.01,
    "rounding_precision": 5,
    "symbol": "BTCUSDT",
    "use_historical": True
}

viz = SimulationVisualizer(params)
plt.show()
