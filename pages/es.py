import streamlit as st
from numpy import arange, exp, sqrt, cos, e, pi, meshgrid, asarray, argsort, randn, rand, seed
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D

# Objective function
def objective(x, y):
    return -20.0 * exp(-0.2 * sqrt(0.5 * (x**2 + y**2))) - exp(0.5 * (cos(2 * pi * x) + cos(2 * pi * y))) + e + 20

# Create a surface plot of the Ackley function
def plot_ackley():
    r_min, r_max = -5.0, 5.0
    xaxis = arange(r_min, r_max, 0.1)
    yaxis = arange(r_min, r_max, 0.1)
    x, y = meshgrid(xaxis, yaxis)
    results = objective(x, y)
    fig = pyplot.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x, y, results, cmap='jet')
    st.pyplot(fig)

# Evolution Strategy (μ, λ) Algorithm
def es_comma(objective, bounds, n_iter, step_size, mu, lam):
    best, best_eval = None, 1e+10
    n_children = int(lam / mu)
    population = list()
    for _ in range(lam):
        candidate = None
        while candidate is None or not in_bounds(candidate, bounds):
            candidate = bounds[:, 0] + rand(len(bounds)) * (bounds[:, 1] - bounds[:, 0])
        population.append(candidate)
    for epoch in range(n_iter):
        scores = [objective(c) for c in population]
        ranks = argsort(argsort(scores))
        selected = [i for i, _ in enumerate(ranks) if ranks[i] < mu]
        children = list()
        for i in selected:
            if scores[i] < best_eval:
                best, best_eval = population[i], scores[i]
                st.write(f'{epoch}, Best: f({best}) = {best_eval:.5f}')
            for _ in range(n_children):
                child = None
                while child is None or not in_bounds(child, bounds):
                    child = population[i] + randn(len(bounds)) * step_size
                children.append(child)
        population = children
    return [best, best_eval]

def in_bounds(point, bounds):
    for d in range(len(bounds)):
        if point[d] < bounds[d, 0] or point[d] > bounds[d, 1]:
            return False
    return True

# Run the App
st.title("Ackley Function Optimization using Evolution Strategy")
st.write("This app demonstrates the optimization of the Ackley function using the (μ, λ) evolution strategy.")

# Plot the function
st.subheader("3D Plot of the Ackley Function")
plot_ackley()

# Evolution Strategy parameters
seed(1)
bounds = asarray([[-5.0, 5.0], [-5.0, 5.0]])
n_iter = 500
step_size = 0.15
mu = 20
lam = 100

st.subheader("Evolution Strategy (μ, λ) Results")
best, score = es_comma(objective, bounds, n_iter, step_size, mu, lam)
st.write('Done!')
st.write(f'Optimal Solution: f({best}) = {score}')
