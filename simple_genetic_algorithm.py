import numpy as np
import matplotlib.pyplot as plt
import statistics as stat
import random

POPULATION_SIZE = 10
GENERATIONS = [50, 100, 200]
CROSSOVER_RATE = [0.1, 0.5, 0.9]
MUTATION_RATE = [0.1, 0.2, 0.5]
X_RANGE = (-1, 21)
BIN_LENGTH = len(bin(X_RANGE[1])[2:])

def fitness_fun(x):
    return -0.25 * x**2 + 5 * x + 6

def int_to_bin(number):
    if number < 0:
        return bin((1 << BIN_LENGTH) + number)[2:]
    else:
        return bin(number)[2:].zfill(BIN_LENGTH)

def bin_to_int(binary_str):
    if len(binary_str) == BIN_LENGTH and binary_str[0] == '1':
        return -((1 << BIN_LENGTH) - int(binary_str, 2))
    else:
        return int(binary_str, 2)


def create_first_population(population_size, x_range):
    return [np.random.randint(x_range[0], x_range[1]) for _ in range(population_size)]

def calculate_population_fitness(population):
    return [fitness_fun(x) for x in population]

def roulette_selection(population, population_fitness):
    min_fitness = min(population_fitness)
    if min_fitness < 0:
        # shift fitness values to be non-negative => it will be easier to work with them
        population_fitness = [x - min_fitness for x in population_fitness]
    fitness_sum = sum(population_fitness)
    probabilities = [x / fitness_sum for x in population_fitness]
    selected_population = []
    for _ in range(len(population)):
        selected_population.append(population[np.random.choice(len(population), p=probabilities)])
    return selected_population

def random_pair_selection(population):
    random.shuffle(population)
    return population

def crossover(parents, crossover_rate):
    parents = [int_to_bin(x) for x in parents]
    children = []
    # random crossover point
    # every parent pair is crossed => two children are created from each pair, new population size is the same as the old one
    for i in range(0, len(parents), 2):
        if random.random() < crossover_rate:
            crossover_point = random.randint(0, BIN_LENGTH - 1)
            children1 = bin_to_int(parents[i][:crossover_point] + parents[i + 1][crossover_point:])
            children2 = bin_to_int(parents[i + 1][:crossover_point] + parents[i][crossover_point:])
            if children1 >= X_RANGE[0] and children1 <= X_RANGE[1] and children2 >= X_RANGE[0] and children2 <= X_RANGE[1]:
                children.append(children1)
                children.append(children2)
            else:
                children.append(bin_to_int(parents[i]))
                children.append(bin_to_int(parents[i + 1]))
        else:
            children.append(bin_to_int(parents[i]))
            children.append(bin_to_int(parents[i + 1]))
    return children
            
def children_mutation(children, mutation_rate):
    mutated_children = []
    for child in children:
        if random.random() < mutation_rate:
            mutation_point = random.randint(0, BIN_LENGTH - 1)
            mutated_child = child ^ (1 << mutation_point)
            # check if after mutation child is still in the range, else return original child
            if mutated_child >= X_RANGE[0] and mutated_child <= X_RANGE[1]:
                mutated_children.append(mutated_child)
            else:
                mutated_children.append(child)
        else:
            mutated_children.append(child)
    return mutated_children
            
def generate_new_population(population, population_fitness, crossover_rate, mutation_rate):
    selected_population = roulette_selection(population, population_fitness)
    parents = random_pair_selection(selected_population)
    children = crossover(parents, crossover_rate)
    return children_mutation(children, mutation_rate)

def plot_fitness_function(x_range):
    x_values = np.linspace(x_range[0], x_range[1], 10000)
    y_values = fitness_fun(x_values)
    max_y = np.max(y_values)
    max_x = x_values[np.argmax(y_values)]
    plt.figure(figsize=(10, 4))
    plt.plot(x_values, y_values, label="Fitness Function")
    plt.scatter(max_x, max_y, color='red')
    plt.text(max_x, max_y, f'Maximum ({max_x:.2f}, {max_y:.2f})', 
             ha='right', va='bottom')
    plt.title("Fitness Function")
    plt.xlabel("x")
    plt.ylabel("Fitness")
    plt.legend()
    plt.show()
    
plot_fitness_function(X_RANGE)  # fitness function plot

for generation in GENERATIONS:
    fig, axes = plt.subplots(len(CROSSOVER_RATE), len(MUTATION_RATE), figsize=(20, 10))  # each generation has its own set of plots

    for i, crossover_rate in enumerate(CROSSOVER_RATE):
        for j, mutation_rate in enumerate(MUTATION_RATE):
            min_fitness = []
            avg_fitness = []
            max_fitness = []

            population = create_first_population(POPULATION_SIZE, X_RANGE)

            for gen in range(generation):
                population_fitness = calculate_population_fitness(population)

                min_fitness.append(min(population_fitness))
                avg_fitness.append(stat.mean(population_fitness))
                max_fitness.append(max(population_fitness))

                population = generate_new_population(population, population_fitness, crossover_rate, mutation_rate)

            ax = axes[i, j]

            ax.plot(min_fitness, label='Min Fitness')
            ax.plot(avg_fitness, label='Average Fitness')
            ax.plot(max_fitness, label='Max Fitness')
            ax.set_title(f"CR: {crossover_rate}, MR: {mutation_rate}")
            ax.set_xlabel("Generation")
            ax.set_ylabel("Fitness")
            ax.legend()

            final_max_fitness = max_fitness[-1]
            ax.text(generation - 1, final_max_fitness, f"{final_max_fitness:.2f}", 
                    ha='right', va='bottom')

    plt.suptitle(f"Results for Generations: {generation}")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


