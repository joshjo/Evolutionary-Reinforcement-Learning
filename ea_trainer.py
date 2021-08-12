from array import array
import gym
import gym_anytrading
import random
import numpy as np, time
import matplotlib.pyplot as plt
from random import randint
from statistics import mean, median
from collections import Counter
# env = gym.make("CartPole-v0")
# env = gym.make('forex-v0')
env = gym.make('stocks-v0')
env.reset()
#Number of frames
goal_steps = 500
score_requirement = 50
initial_games = 1000

def create_data():
    training_data, scores, accepted_scores = [], [], []
    for _ in range(initial_games):
        score = 0
        #Moves from current environment and previous observations
        game_memory, prev_observation = [], []
        for _ in range(goal_steps):
            action = random.randrange(0,2)
            observation, reward, done, info = env.step(action)

            if len(prev_observation) > 0:
                game_memory.append([prev_observation, action])

            prev_observation = observation
            score += reward

            if done:
                break

        if score >= score_requirement:
            accepted_scores.append(score)
            for data in game_memory:
                training_data.append(data)

        env.reset()
        scores.append(score)

    print('Average accepted score:', mean(accepted_scores))
    print('Median accepted score:', median(accepted_scores))
    return training_data

def create_initial_pop(pop_size, internal_pop_size):
    initial_population = []
    if internal_pop_size > 0:
        for j in range(0,pop_size[0]):
                initial_population.append(np.random.uniform(low = -2.0, high = 2.0,size = (pop_size[1],internal_pop_size)))

        print('Initial Population IN:\n{}'.format(np.array(initial_population)))
        return np.array(initial_population)
    else:
        initial_pop = np.random.uniform(low = -2.0, high = 2.0, size = pop_size)
        print('Initial Population:\n{}'.format(initial_pop))
        return initial_pop

def sigmoid(z):
    return 1/(1+np.exp(-z))

def predict(X):
    pred = np.empty((X.shape[0], 1))
    for i in range(X.shape[0]):
        if X[i] >= 0.5:
            pred[i] = 0
        else:
            pred[i] = 1
    return pred

def cal_fitness(population, X, y, pop_size):
    fitness = np.empty((pop_size[0], 1))
    for i in range(pop_size[0]):
        hx  = X@(population[i]).T
        fitness[i][0] = np.sum(hx)
    return fitness

def selection(population, fitness, num_parents, internal_pop_size):
    fitness = list(fitness)
    parents = np.empty((num_parents, population.shape[1]))
    parents_change = []
    for i in range(num_parents):
        max_fitness_idx = np.where(fitness == np.max(fitness))
        if internal_pop_size > 0:
            max_population = [np.max(p) for p in population[max_fitness_idx[0][0]]]
            parents[i,:] = max_population[max_fitness_idx[0][0]:]
            parents_change = population[max_fitness_idx[0][0], :]
        else:
            parents[i,:] = population[max_fitness_idx[0][0], :]
        fitness[max_fitness_idx[0][0]] = -999999
    return parents, parents_change

def crossover(parents, num_offsprings):
    offsprings = np.empty((num_offsprings, parents.shape[1]))
    crossover_point = int(parents.shape[1]/2)
    crossover_rate = 0.8
    i=0
    while (parents.shape[0] < num_offsprings):
        parent1_index = i%parents.shape[0]
        parent2_index = (i+1)%parents.shape[0]
        x = random.random()
        if x > crossover_rate:
            continue
        parent1_index = i%parents.shape[0]
        parent2_index = (i+1)%parents.shape[0]
        offsprings[i,0:crossover_point] = parents[parent1_index,0:crossover_point]
        offsprings[i,crossover_point:] = parents[parent2_index,crossover_point:]
        i=+1
    return offsprings

def mutation(offsprings):
    mutants = np.empty((offsprings.shape))
    mutation_rate = 0.4
    for i in range(mutants.shape[0]):
        random_value = random.random()
        mutants[i,:] = offsprings[i,:]
        if random_value > mutation_rate:
            continue
        int_random_value = randint(0,offsprings.shape[1]-1)
        mutants[i,int_random_value] += np.random.uniform(-1.0, 1.0, 1)

    return mutants

def GA_model(training_data):
    X = np.array([i[0] for i in training_data])
    y = np.array([i[1] for i in training_data]).reshape(-1, 1)

    weights, fitness_history, list_selected, test_mean, times = [], [], [], [], []
    num_solutions = 2
    pop_size = (num_solutions, X.shape[1])
    num_parents = int(pop_size[0]/2)
    num_offsprings = pop_size[0] - num_parents
    num_generations = 150
    internal_pop_size = 0

    if isinstance(X[0][0],(list, tuple, np.ndarray)):
        internal_pop_size = len(X[0][0])

    population = create_initial_pop(pop_size, internal_pop_size)

    time_start = time.time()


    for i in range(num_generations):
        fitness = cal_fitness(population, X, y, pop_size)
        fitness_history.append(fitness)
        parents, parents_change = selection(population, fitness, num_parents,internal_pop_size)
        offsprings = crossover(parents, num_offsprings)
        mutants = mutation(offsprings)
        list_selected.append(np.max(mutants))
        test_mean.append(np.mean(mutants))
        if internal_pop_size > 0:
            population[0:parents.shape[0]] = parents_change
        else:
            population[0:parents.shape[0], :] = parents
            population[parents.shape[0]:, :] = mutants
        times.append(time.time()-time_start)

    fitness_last_gen = cal_fitness(population, X, y, pop_size)
    max_fitness = np.where(fitness_last_gen == np.max(fitness_last_gen))
    weights.append(population[max_fitness[0][0],:])
    return weights, fitness_history, list_selected, test_mean, times

def GA_model_predict(test_data, weights):
    hx = sigmoid(test_data@(weights).T)
    pred = predict(hx)
    pred = pred.astype(int)
    return pred[0][0]

training_data = create_data()
weights, fitness_history, selected, test_mean, times = GA_model(training_data)
print('Weights: {}'.format(weights))
weights = np.asarray(weights)
num_generations = 150

fitness_history_mean = [np.mean(fitness) for fitness in fitness_history]
fitness_history_max = [np.max(fitness) for fitness in fitness_history]
# print('fitness_history--->',fitness_history)
plt.plot(list(range(num_generations)), fitness_history_mean, label = 'Mean Fitness')
plt.plot(list(range(num_generations)), fitness_history_max, label = 'Max Fitness')
plt.legend()
plt.title('Fitness through the generations')
plt.xlabel('Generations')
plt.ylabel('Fitness')
plt.show()
plt.plot(list(range(num_generations)), selected, label = 'Max Selection')
plt.legend()
plt.title('Selection through the generations')
plt.xlabel('Generations')
plt.ylabel('Selection')
plt.show()
plt.plot(list(range(num_generations)), times, label = 'Time')
plt.legend()
plt.title('Time through the generations')
plt.xlabel('Generations')
plt.ylabel('Time')
plt.show()
plt.plot(list(range(num_generations)), test_mean, label = 'Mean Test')
plt.legend()
plt.title('Test through the generations')
plt.xlabel('Generations')
plt.ylabel('Test')
plt.show()


