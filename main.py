import streamlit as st
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns

# Set style for plots
sns.set_style("darkgrid")
np.random.seed(42)

# Streamlit settings for the app
st.title("Casino Simulation with Kelly Criterion and Markov Models")
st.sidebar.header("Simulation Parameters")

# Sidebar inputs for tunable parameters
n_spins = st.sidebar.number_input(
    "Number of Spins", min_value=100, max_value=5000, value=1000, step=100
)
initial_capital = st.sidebar.number_input(
    "Initial Capital", min_value=100, max_value=10000, value=1000, step=100
)
safety_value = st.sidebar.slider(
    "Safety Factor for Kelly Criterion",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.1,
)
n_simulations = st.sidebar.number_input(
    "Number of Simulations", min_value=10, max_value=1000, value=100, step=10
)


# Define functions
def generate_roulette_data(n_spins=1000):
    red_numbers = {1, 3, 5, 7, 9, 12, 14, 16, 18, 19, 21, 23, 25, 27, 30, 32, 34, 36}
    numbers = np.random.randint(0, 38, n_spins)
    returns = []
    for num in numbers:
        if num == 0 or num == 37:
            returns.append(-1)
        else:
            returns.append(1 if num in red_numbers else -1)
    base_time = datetime(2024, 1, 1)
    timestamps = [base_time + timedelta(minutes=i * 3) for i in range(n_spins)]
    return pd.DataFrame(
        {"timestamp": timestamps, "number": numbers, "returns": returns}
    )


def create_state_classifications(returns_data, n_states_per_side=3):
    states = pd.Series(np.zeros(len(returns_data)), dtype=int)
    negative_mask = returns_data < 0
    positive_mask = returns_data > 0
    negative_indices = np.where(negative_mask)[0]
    positive_indices = np.where(positive_mask)[0]
    if len(negative_indices) > 0:
        neg_splits = np.array_split(negative_indices, n_states_per_side)
        for i, split_indices in enumerate(neg_splits):
            states[split_indices] = i
    if len(positive_indices) > 0:
        pos_splits = np.array_split(positive_indices, n_states_per_side)
        for i, split_indices in enumerate(pos_splits):
            states[split_indices] = i + n_states_per_side
    return states


def calculate_transition_matrix(states):
    n_states = len(np.unique(states))
    transitions = np.zeros((n_states, n_states))
    for i, j in zip(states[:-1], states[1:]):
        transitions[i, j] += 1
    row_sums = transitions.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        transition_matrix = np.nan_to_num(transitions / row_sums)
    return transition_matrix


def kelly_criterion(win_prob, odds=1, safety=0.5):
    kelly = max(0, win_prob - ((1 - win_prob) / odds))
    return kelly * safety


def simulate_portfolio(
    initial_capital, states, transition_matrix, safety=0.5, n_simulations=100
):
    states_array = np.array(states, dtype=int)
    portfolio_values = np.zeros((n_simulations, len(states_array)))
    portfolio_values[:, 0] = initial_capital
    n_states = transition_matrix.shape[0]
    mid_state = n_states // 2

    for sim in range(n_simulations):
        current_capital = initial_capital
        for t in range(1, len(states_array)):
            current_state = states_array[t - 1]
            win_prob = np.sum(transition_matrix[current_state, mid_state:])
            win_prob = np.clip(win_prob + np.random.normal(0, 0.05), 0, 1)
            kelly_fraction = kelly_criterion(win_prob, safety=safety)
            min_bet = 10
            max_bet_fraction = 0.25
            bet_amount = min(
                max(min_bet, current_capital * kelly_fraction),
                current_capital * max_bet_fraction,
            )
            result = 1 if states_array[t] >= mid_state else -1
            current_capital += bet_amount * result
            transaction_cost = np.random.uniform(0.001, 0.003) * bet_amount
            current_capital -= transaction_cost
            portfolio_values[sim, t] = max(0, current_capital)
    return portfolio_values


def analyze_portfolio_performance(portfolio_values):
    final_values = portfolio_values[:, -1]
    mean_final = np.mean(final_values)
    median_final = np.median(final_values)
    std_final = np.std(final_values)
    running_max = np.maximum.accumulate(portfolio_values, axis=1)
    drawdowns = (running_max - portfolio_values) / running_max
    max_drawdown = np.max(drawdowns)
    return {
        "mean_final_value": mean_final,
        "median_final_value": median_final,
        "std_final_value": std_final,
        "max_drawdown": max_drawdown,
        "profit_rate": np.mean(final_values > portfolio_values[:, 0]),
        "loss_rate": np.mean(final_values < portfolio_values[:, 0]),
    }


# Generate and prepare data
data = generate_roulette_data(n_spins)
returns = data["returns"].values
states_2 = (returns > 0).astype(int)
states_6 = create_state_classifications(returns)
transition_matrix_2 = calculate_transition_matrix(states_2)
transition_matrix_6 = calculate_transition_matrix(states_6)

# Simulate portfolios
portfolio_2state = simulate_portfolio(
    initial_capital,
    states_2,
    transition_matrix_2,
    safety=safety_value,
    n_simulations=n_simulations,
)
portfolio_6state = simulate_portfolio(
    initial_capital,
    states_6,
    transition_matrix_6,
    safety=safety_value,
    n_simulations=n_simulations,
)

# Analyze performance
metrics_2state = analyze_portfolio_performance(portfolio_2state)
metrics_6state = analyze_portfolio_performance(portfolio_6state)


# Display performance summary with Streamlit metrics
st.subheader("Performance Summary")

# Create columns for 2-state model metrics
st.write("### 2-State Model")
col1, col2, col3 = st.columns(3)
col1.metric("Mean Final Value", f"${metrics_2state['mean_final_value']:.2f}")
col2.metric("Median Final Value", f"${metrics_2state['median_final_value']:.2f}")
col3.metric("Standard Deviation", f"${metrics_2state['std_final_value']:.2f}")

col4, col5, col6 = st.columns(3)
col4.metric("Max Drawdown", f"{metrics_2state['max_drawdown']:.2%}")
col5.metric("Profit Rate", f"{metrics_2state['profit_rate']:.2%}")
col6.metric("Loss Rate", f"{metrics_2state['loss_rate']:.2%}")

# Create a horizontal divider between the models
st.write("---")

# Create columns for 6-state model metrics
st.write("### 6-State Model")
col1, col2, col3 = st.columns(3)
col1.metric("Mean Final Value", f"${metrics_6state['mean_final_value']:.2f}")
col2.metric("Median Final Value", f"${metrics_6state['median_final_value']:.2f}")
col3.metric("Standard Deviation", f"${metrics_6state['std_final_value']:.2f}")

col4, col5, col6 = st.columns(3)
col4.metric("Max Drawdown", f"{metrics_6state['max_drawdown']:.2%}")
col5.metric("Profit Rate", f"{metrics_6state['profit_rate']:.2%}")
col6.metric("Loss Rate", f"{metrics_6state['loss_rate']:.2%}")

# Optional: Add a brief summary text at the bottom for additional insights
st.write(
    "**Note**: The above metrics help compare the stability and profitability of each model."
)

# Plotting
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 20))

sns.heatmap(transition_matrix_2, annot=True, cmap="coolwarm", ax=ax1, fmt=".3f")
ax1.set_title("Transition Matrix - 2-State Model")

sns.heatmap(transition_matrix_6, annot=True, cmap="coolwarm", ax=ax2, fmt=".3f")
ax2.set_title("Transition Matrix - 6-State Model")

mean_2state = np.mean(portfolio_2state, axis=0)
mean_6state = np.mean(portfolio_6state, axis=0)
std_2state = np.std(portfolio_2state, axis=0)
std_6state = np.std(portfolio_6state, axis=0)

ax3.plot(mean_2state, label="2-State Model", color="blue")
ax3.plot(mean_6state, label="6-State Model", color="red")
ax3.fill_between(
    range(len(mean_2state)),
    mean_2state - std_2state,
    mean_2state + std_2state,
    alpha=0.2,
    color="blue",
)
ax3.fill_between(
    range(len(mean_6state)),
    mean_6state - std_6state,
    mean_6state + std_6state,
    alpha=0.2,
    color="red",
)
ax3.set_title("Portfolio Value Comparison")
ax3.set_xlabel("Number of Spins")
ax3.set_ylabel("Portfolio Value")
ax3.legend()

st.pyplot(fig)
