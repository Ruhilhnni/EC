import matplotlib.pyplot as plt
from itertools import permutations
import random
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

# Streamlit Title
st.title("Traveling Salesman Problem")

# Coordinates of Cities
x = [0,3,6,7,15,10,16,5,8,1.5]
y = [1,2,1,4.5,-1,2.5,11,6,9,12]
cities_names = ["Gliwice", "Cairo", "Rome", "Krakow", "Paris", "Alexandria", "Berlin", "Tokyo", "Rio", "Budapest"]
city_coords = dict(zip(cities_names, zip(x, y)))

# Input Parameters
city_name = st.text_input("Enter Your Name")
n_population = st.number_input("Population Size", value=0, min_value=0, max_value=250)
crossover_per = st.number_input("Crossover Percentage", value=0.10, min_value=0.0, max_value=1.0, step=0.10)
mutation_per = st.number_input("Mutation Percentage", value=0.10, min_value=0.0, max_value=1.0, step=0.10)
n_generations = st.number_input("Number of Generations", value=0, min_value=0, max_value=200)

# Button to Start GA
st.button("Find The Best Route")

# Pastel Palette
colors = sns.color_palette("pastel", len(cities_names))

# City Icons
city_icons = {
    "Gliwice": "♕",
    "Cairo": "♖",
    "Rome": "♗",
    "Krakow": "♘",
    "Paris": "♙",
    "Alexandria": "♔",
    "Berlin": "♚",
    "Tokyo": "♛",
    "Rio": "♜",
    "Budapest": "♝",
}

# Plot City Locations
fig, ax = plt.subplots()
ax.grid(False)  # No Grid
for i, (city, (city_x, city_y)) in enumerate(city_coords.items()):
    color = colors[i]
    icon = city_icons[city]
    ax.scatter(city_x, city_y, c=[color], s=1200, zorder=2)
    ax.annotate(icon, (city_x, city_y), fontsize=40, ha='center', va='center', zorder=3)
    ax.annotate(city, (city_x, city_y), fontsize=12, ha='center', va='bottom', xytext=(0, -30), textcoords='offset points')
    
    # Connect cities with opaque lines
    for j, (other_city, (other_x, other_y)) in enumerate(city_coords.items()):
        if i != j:
            ax.plot([city_x, other_x], [city_y, other_y], color='gray', linestyle='-', linewidth=1, alpha=0.1)

fig.set_size_inches(16, 12)
#st.pyplot(fig)

# Initialize Population
def initial_population(cities_list, n_population=250):
    population_perms = []
    possible_perms = list(permutations(cities_list))
    random_ids = random.sample(range(0, len(possible_perms)), n_population)
    for i in random_ids:
        population_perms.append(list(possible_perms[i]))
    return population_perms

# Distance Calculation
def dist_two_cities(city_1, city_2):
    city_1_coords = city_coords[city_1]
    city_2_coords = city_coords[city_2]
    return np.sqrt(np.sum((np.array(city_1_coords) - np.array(city_2_coords)) ** 2))

# Total Distance of Individual
def total_dist_individual(individual):
    total_dist = 0
    for i in range(len(individual)):
        if i == len(individual) - 1:
            total_dist += dist_two_cities(individual[i], individual[0])
        else:
            total_dist += dist_two_cities(individual[i], individual[i + 1])
    return total_dist

# Fitness Probability
def fitness_prob(population):
    total_dist_all_individuals = [total_dist_individual(ind) for ind in population]
    max_population_cost = max(total_dist_all_individuals)
    population_fitness = max_population_cost - np.array(total_dist_all_individuals)
    population_fitness_sum = sum(population_fitness)
    return population_fitness / population_fitness_sum

# Roulette Wheel Selection
def roulette_wheel(population, fitness_probs):
    population_fitness_probs_cumsum = fitness_probs.cumsum()
    selected_individual_index = np.searchsorted(population_fitness_probs_cumsum, np.random.uniform(0,1))
    return population[selected_individual_index]

# Crossover Function
def crossover(parent_1, parent_2):
    cut = random.randint(1, len(cities_names) - 2)
    offspring_1 = parent_1[:cut] + [city for city in parent_2 if city not in parent_1[:cut]]
    offspring_2 = parent_2[:cut] + [city for city in parent_1 if city not in parent_2[:cut]]
    return offspring_1, offspring_2

Baby 💌, [11/3/2024 11:50 AM]
# Mutation Function
def mutation(offspring):
    index_1, index_2 = random.sample(range(len(offspring)), 2)
    offspring[index_1], offspring[index_2] = offspring[index_2], offspring[index_1]
    return offspring

# Genetic Algorithm
def run_ga(cities_names, n_population, n_generations, crossover_per, mutation_per):
    population = initial_population(cities_names, n_population)
    for generation in range(n_generations):
        fitness_probs = fitness_prob(population)
        
        # Selection
        parents_list = [roulette_wheel(population, fitness_probs) for _ in range(int(crossover_per * n_population))]

        # Crossover and Mutation
        offspring_list = []
        for i in range(0, len(parents_list), 2):
            if i+1 < len(parents_list):
                offspring_1, offspring_2 = crossover(parents_list[i], parents_list[i + 1])
                if random.random() < mutation_per:
                    offspring_1 = mutation(offspring_1)
                if random.random() < mutation_per:
                    offspring_2 = mutation(offspring_2)
                offspring_list.extend([offspring_1, offspring_2])

        # Combine and Select Best Individuals
        mixed_offspring = population + offspring_list
        fitness_probs = fitness_prob(mixed_offspring)
        sorted_indices = np.argsort(fitness_probs)[::-1]
        population = [mixed_offspring[i] for i in sorted_indices[:n_population]]
    
    return population

# Run Genetic Algorithm
best_mixed_offspring = run_ga(cities_names, n_population, n_generations, crossover_per, mutation_per)

# Calculate Total Distances
total_dist_all_individuals = [total_dist_individual(ind) for ind in best_mixed_offspring]
index_minimum = np.argmin(total_dist_all_individuals)
minimum_distance = min(total_dist_all_individuals)

# Output Results
st.write(f"Minimum Distance: {minimum_distance}")
shortest_path = best_mixed_offspring[index_minimum]
st.write("Best Route:", shortest_path)

# Plot the Shortest Path
x_shortest = [city_coords[city][0] for city in shortest_path] + [city_coords[shortest_path[0]][0]]
y_shortest = [city_coords[city][1] for city in shortest_path] + [city_coords[shortest_path[0]][1]]

fig, ax = plt.subplots()
ax.plot(x_shortest, y_shortest, '--go', label='Best Route', linewidth=2.5)
plt.legend()

for i in range(len(x)):
    for j in range(i + 1, len(x)):
        ax.plot([x[i], x[j]], [y[i], y[j]], 'k-', alpha=0.09, linewidth=1)

plt.title("TSP Best Route Using GA", fontsize=25)
str_params = f'\n{n_generations} Generations\n{n_population} Population Size\n{crossover_per} Crossover\n{mutation_per} Mutation'
plt.suptitle(f"Total Distance Travelled: {round(minimum_distance, 3)}" + str_params, fontsize=18, y=1.047)

for i, txt in enumerate(shortest_path):
    ax.annotate(f"{i+1}- {txt}", (x_shortest[i], y_shortest[i]), fontsize=20)

fig.set_size_inches(16, 12)
st.pyplot(fig)
