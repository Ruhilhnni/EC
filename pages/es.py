# Full Streamlit App Code to Visualize and Optimize Ackley Function
import streamlit as st
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Objective function for Ackley
def objective(x, y):
    return -20.0 * np.exp(-0.2 * np.sqrt(0.5 * (x**2 + y**2))) - np.exp(0.5 * (np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y))) + np.e + 20

# Function to plot the Ackley surface
def plot_ackley():
    r_min, r_max = -5.0, 5.0
    xaxis = np.arange(r_min, r_max, 0.1)
    yaxis = np.arange(r_min, r_max, 0.1)
    x, y = np.meshgrid(xaxis, yaxis)
    results = objective(x, y)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x, y, results, cmap='jet')
    st.pyplot(fig)

# Evolution Strategy (mu, lambda) with comma selection
def es_comma(objective, bounds, n_iter, step_size, mu, lam):
    best, best_eval = None, 1e+10
    n_children = int(lam / mu)
    population = [create_candidate(bounds) for _ in range(lam)]
    for epoch in range(n_iter):
        scores = [objective(c[0], c[1]) for c in population]
        ranks = np.argsort(scores)
        selected = [population[ranks[i]] for i in range(mu)]
        children = []
        for i in range(mu):
            if scores[ranks[i]] < best_eval:
                best, best_eval = population[ranks[i]], scores[ranks[i]]
                st.write(f'{epoch}, Best: f({best}) = {best_eval:.5f}')
            for _ in range(n_children):
                child = None
                while child is None or not in_bounds(child, bounds):
                    child = population[ranks[i]] + np.random.randn(len(bounds)) * step_size
                children.append(child)
        population = children
    return best, best_eval

# Function to create a candidate within bounds
def create_candidate(bounds):
    candidate = None
    while candidate is None or not in_bounds(candidate, bounds):
        candidate = bounds[:, 0] + np.random.rand(len(bounds)) * (bounds[:, 1] - bounds[:, 0])
    return candidate

# Check if a candidate is within bounds
def in_bounds(point, bounds):
    return all(bounds[d, 0] <= point[d] <= bounds[d, 1] for d in range(len(bounds)))

# Streamlit UI
st.title("Ackley Function Optimization using Evolution Strategy")
st.write("This app demonstrates the optimization of the Ackley function using the (μ, λ) evolution strategy.")

# Display Ackley surface plot
st.subheader("3D Plot of the Ackley Function")
plot_ackley()

# Parameters for Evolution Strategy
np.random.seed(1)
bounds = np.array([[-5.0, 5.0], [-5.0, 5.0]])
n_iter = 500
step_size = 0.15
mu = 20
lam = 100

st.subheader("Evolution Strategy (μ, λ) Results")
best, score = es_comma(objective, bounds, n_iter, step_size, mu, lam)
st.write('Optimization Complete!')
st.write(f'Best Solution Found: f({best}) = {score:.5f}')
