import streamlit as st
import matplotlib.pyplot as plt
from itertools import permutations
import random
import numpy as np
import seaborn as sns

# Streamlit Title
st.title("City Coordinates Input")

# Subtitle and instructions for city input
st.write("Enter up to 10 cities with their coordinates (x, y) in range 1-10.")

# Create input fields for up to 10 cities with x and y coordinates
cities = []
for i in range(1, 11):
    col1, col2, col3, col4 = st.columns([3, 2, 2, 1])
    
    # City Name input
    with col1:
        city_name = st.text_input(f"City {i}", value="", key=f"city_{i}_name", placeholder="Enter city name")
    
    # X-coordinate input
    with col2:
        x_coord = st.number_input(f"x-coordinate (City {i})", min_value=1, max_value=10, value=1, key=f"city_{i}_x")
    
    # Y-coordinate input
    with col3:
        y_coord = st.number_input(f"y-coordinate (City {i})", min_value=1, max_value=10, value=1, key=f"city_{i}_y")
    
    # If city name is entered, add to cities list
    if city_name:
        cities.append((city_name, x_coord, y_coord))

# If there are entered cities, create the coordinates dictionary
if cities:
    city_coords = {city[0]: (city[1], city[2]) for city in cities}
else:
    st.warning("Please enter at least one city to proceed.")
    st.stop()

# Parameters for the Genetic Algorithm
n_population = 250
crossover_per = 0.8
mutation_per = 0.2
n_generations = 200

# Button to Start GA
start_ga = st.button("Find The Best Route")

# Only run the algorithm and display the results if the button is clicked
if start_ga:
    # Pastel Palette
    colors = sns.color_palette("pastel", len(cities))

    # Plot City Locations
    fig, ax = plt.subplots()
    for i, (city, (city_x, city_y)) in enumerate(city_coords.items()):
        ax.scatter(city_x, city_y, c=[colors[i]], s=1200)
        ax.annotate(city, (city_x, city_y), fontsize=12, ha='center', va='bottom', xytext=(0, -30), textcoords='offset points')
        for j, (other_city, (other_x, other_y)) in enumerate(city_coords.items()):
            if i != j:
                ax.plot([city_x, other_x], [city_y, other_y], color='gray', linestyle='-', linewidth=1, alpha=0.1)

    fig.set_size_inches(16, 12)
    st.pyplot(fig)

    # Distance Calculation
    def dist_two_cities(city_1, city_2):
        return np.linalg.norm(np.array(city_coords[city_1]) - np.array(city_coords[city_2]))

    # Total Distance of an Individual
    def total_dist_individual(individual):
        return sum(dist_two_cities(individual[i], individual[(i + 1) % len(individual)]) for i in range(len(individual)))

    # Fitness Probability
    def fitness_prob(population):
        distances = np.array([total_dist_individual(ind) for ind in population])
        fitness = np.max(distances) - distances
        return fitness / fitness.sum()

    # Roulette Wheel Selection
    def roulette_wheel(population, fitness_probs):
        cumsum = fitness_probs.cumsum()
        return population[np.searchsorted(cumsum, np.random.rand())]

    # Crossover and Mutation Functions
    def crossover(parent_1, parent_2):
        cut = random.randint(1, len(cities) - 2)
        offspring_1 = parent_1[:cut] + [city for city in parent_2 if city not in parent_1[:cut]]
        offspring_2 = parent_2[:cut] + [city for city in parent_1 if city not in parent_2[:cut]]
        return offspring_1, offspring_2

    def mutation(offspring):
        index_1, index_2 = random.sample(range(len(offspring)), 2)
        offspring[index_1], offspring[index_2] = offspring[index_2], offspring[index_1]
        return offspring

    # Genetic Algorithm
    def run_ga(cities_names, n_population, n_generations, crossover_per, mutation_per):
        population = [random.sample(cities_names, len(cities_names)) for _ in range(n_population)]
        for _ in range(n_generations):
            fitness_probs = fitness_prob(population)
            parents = [roulette_wheel(population, fitness_probs) for _ in range(int(crossover_per * n_population))]
            offspring_list = []
            for i in range(0, len(parents), 2):
                if i + 1 < len(parents):
                    offspring_1, offspring_2 = crossover(parents[i], parents[i + 1])
                    if random.random() < mutation_per:
                        offspring_1 = mutation(offspring_1)
                    if random.random() < mutation_per:
                        offspring_2 = mutation(offspring_2)
                    offspring_list.extend([offspring_1, offspring_2])
            population = sorted(population + offspring_list, key=total_dist_individual)[:n_population]
        return population

    # Run Genetic Algorithm
    best_population = run_ga(list(city_coords.keys()), n_population, n_generations, crossover_per, mutation_per)

    # Calculate and Display Best Route
    best_route = min(best_population, key=total_dist_individual)
    min_distance = total_dist_individual(best_route)
    st.write(f"Minimum Distance: {min_distance:.2f}")
    st.write("Best Route:", best_route)

    # Plot Best Route
    x_best = [city_coords[city][0] for city in best_route] + [city_coords[best_route[0]][0]]
    y_best = [city_coords[city][1] for city in best_route] + [city_coords[best_route[0]][1]]
    fig, ax = plt.subplots()
    ax.plot(x_best, y_best, 'go-', label='Best Route', linewidth=2.5)
    plt.legend()

    for i in range(len(city_coords)):
        for j in range(i + 1, len(city_coords)):
            ax.plot([x_best[i], x_best[j]], [y_best[i], y_best[j]], 'k-', alpha=0.09, linewidth=1)

    plt.title("TSP Best Route Using GA", fontsize=25)
    plt.suptitle(f"Total Distance: {min_distance:.2f} | Generations: {n_generations} | Population Size: {n_population} | Crossover: {crossover_per} | Mutation: {mutation_per}", fontsize=18, y=1.047)

    for i, city in enumerate(best_route):
        ax.annotate(f"{i+1}- {city}", (x_best[i], y_best[i]), fontsize=20)

    fig.set_size_inches(16, 12)
    st.pyplot(fig)
