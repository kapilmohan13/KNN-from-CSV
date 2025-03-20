import numpy as np
import matplotlib.pyplot as plt

# Parameters
S0 = 100  # initial stock price
T = 1.0  # time horizon (in years)
N = 1000  # number of time steps
mu = 0.1  # expected return
sigma = 0.2  # volatility

# Generate Brownian motion
dt = T / N
t = np.linspace(0, T, N)
W = np.random.standard_normal(size=N)
W = np.cumsum(W) * np.sqrt(dt)  # cumulative sum to simulate Brownian motion

# Simulate stock price
X = (mu - 0.5 * sigma**2) * t + sigma * W
S = S0 * np.exp(X)

# Plot stock price
plt.figure(figsize=(10, 6))
plt.plot(t, S)
plt.title('Simulated Stock Price Using Brownian Motion')
plt.xlabel('Time (years)')
plt.ylabel('Stock Price')
plt.grid(True)
plt.show()
