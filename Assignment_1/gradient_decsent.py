import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("D3.csv")
X1 = data.iloc[:, 0].values.astype(float)
X2 = data.iloc[:, 1].values.astype(float)
X3 = data.iloc[:, 2].values.astype(float)
y = data.iloc[:, -1].values.astype(float)
m = len(y)
y = y.reshape(m, 1)

def compute_cost(X, y, theta):
    m = len(y)
    predictions = X @ theta
    error = predictions - y
    return (1 / (2 * m)) * np.sum(error ** 2)

def gradient_descent(X, y, theta, alpha, iterations):
    cost_history = []
    for _ in range(iterations):
        predictions = X @ theta
        error = predictions - y
        gradient = (1 / m) * (X.T @ error)
        theta -= alpha * gradient
        cost_history.append(compute_cost(X, y, theta))
    return theta, cost_history

alphas = [0.1, 0.05, 0.01]
iterations = 1000
features = {"X1": X1, "X2": X2, "X3": X3}

for name, feature in features.items():
    print(f"\n===== Training with {name} only =====")
    X = np.hstack((np.ones((m, 1)), feature.reshape(m, 1)))
    for alpha in alphas:
        theta = np.zeros((2, 1), dtype=float)
        theta, cost_history = gradient_descent(X, y, theta, alpha, iterations)
        print(f"Alpha = {alpha} -> Final theta:\n{theta.ravel()}")
        print(f"Final cost: {cost_history[-1]:.4f}")
        plt.scatter(feature, y, color='red', marker='x', label='Training data')
        plt.plot(feature, X @ theta, color='blue', label=f'LR (alpha={alpha})')
        plt.xlabel(name)
        plt.ylabel("y")
        plt.legend()
        plt.title(f"Linear Regression with {name}")
        plt.show()
        plt.plot(range(iterations), cost_history, color='purple')
        plt.xlabel("Iteration")
        plt.ylabel("Cost J(theta)")
        plt.title(f"Cost Convergence for {name} (alpha={alpha})")
        plt.show()

print("\n===== Training with all features =====")
X = np.hstack((np.ones((m, 1)), np.column_stack((X1, X2, X3))))
for alpha in alphas:
    theta = np.zeros((4, 1), dtype=float)
    theta, cost_history = gradient_descent(X, y, theta, alpha, iterations)
    print(f"\nAlpha = {alpha} -> Final theta:\n{theta.ravel()}")
    print(f"Final cost: {cost_history[-1]:.4f}")
    plt.plot(range(iterations), cost_history, color='purple')
    plt.xlabel("Iteration")
    plt.ylabel("Cost J(theta)")
    plt.title(f"Cost Convergence for All Features (alpha={alpha})")
    plt.show()