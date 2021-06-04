import random

from deap import base, algorithms
from deap import creator
from deap import tools
import numpy as np
import pandas as pd
import time

CXPB = 0.5  # probability with which two individuals are crossed
MUTPB = 0.2  # probability for mutating an individual
GEN = 1000  # number of generations
TOURNSIZE = 9  # tournament size
POP = 300  # number of population
PRINT = True  # if True it prints the intermediate distances. False to print only the best result


# function to read the excel with the distance matrix
def parse_input(path):
    distances = pd.read_excel(path, index_col=0)
    return distances


distance_map = parse_input('Lab10DistancesMatrix.xlsx')


# Returns True if the route contains all cities in list_of_cities ONCE and ONLY ONCE
def is_valid_route(list_of_cities):
    return not any(list_of_cities.count(element) > 1 for element in list_of_cities)


creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)


# function to create the genes
# receives a list with the the cities, and transforms it to indexes.
def create_input(initial_list):
    int_list = [int(i[1:]) - 1 for i in initial_list]
    a = random.sample(int_list, len(int_list))
    return a


# function to create the toolbox
def create_toolbox(cities_list):
    toolbox = base.Toolbox()

    # Attribute generator
    #                      define 'indices' to be an attribute ('gene')
    toolbox.register("indices", create_input, cities_list)

    # Structure initializers
    #                         define 'individual' to be an individual
    #                         consisting of a list of genes, i.e a route
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)

    # population is a list of individuals, i.e a list of routes
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # register the fitness function
    toolbox.register("evaluate", evalTSP)

    # register the crossover operator
    toolbox.register("mate", crossover)
    # register a mutation operator with a probability to
    # flip each attribute/gene of 0.05
    toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.05)

    # operator for selecting individuals for breeding the next
    # generation: each individual of the current generation
    # is replaced by the 'fittest' (best) of nine individuals
    # drawn randomly from the current generation.
    toolbox.register("select", tools.selTournament, tournsize=TOURNSIZE)
    return toolbox


# the fitness function
# function to get the distance between two individuals
def evalTSP(individual):
    point1 = 'P' + str((individual[-1] + 1))
    point2 = 'P' + str((individual[0] + 1))
    distance = distance_map[point1][point2]
    for gene1, gene2 in zip(individual[0:-1], individual[1:]):
        point1 = 'P' + str((gene1 + 1))
        point2 = 'P' + str((gene2 + 1))
        distance += distance_map[point1][point2]
    return distance,


# function to make the crossover
def crossover(p1, p2):
    firsPoint = random.randint(0, len(p1) - 1)
    secondPoint = random.randint(0, len(p2) - 1)

    k = min(firsPoint, secondPoint)
    j = max(firsPoint, secondPoint) + 1

    child1 = p1[k:j]
    child2 = p2[k:j]

    temp1 = [city for city in p2 if city not in child1]
    temp2 = [city for city in p1 if city not in child2]

    child1 += temp1
    child2 += temp2

    p1[0:len(child1)] = child1[0:len(child1)]
    p2[0:len(child2)] = child1[0:len(child2)]

    return p1, p2


# function to print the results starting in P1
def p_print(best_path, duration):
    indexP1 = best_path.index(0)
    best_dist = best_path.fitness.values
    best_path = best_path[indexP1:] + best_path[:indexP1]

    best_route = ['P' + str(i + 1) for i in best_path]
    print()
    print("******************************** BEST RESULT ********************************")
    print("The best route is: ")
    print(best_route)
    print('Final distance:   {0:.2f}'.format(best_dist[0]))
    print('Found in:   {0:.2f} seconds'.format(duration))


# function to register metrics
def register_metrics():
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", np.min)
    stats.register("max", np.max)
    return stats


def main():
    random.seed(64)
    # input lists (change on the two functions below)

    cities_list15 = ['P' + str(i) for i in range(1, 16)]
    cities_list25 = ['P' + str(i) for i in range(1, 26)]
    cities_list65 = ['P' + str(i) for i in range(1, 66)]
    cities_list100 = [cols for cols in distance_map.columns]

    # checking if the list is valid
    if not is_valid_route(cities_list15):
        print("Route not valid, has duplicate cities!!")
        return

    toolbox = create_toolbox(cities_list15)

    # create an initial population of POP individuals
    pop = toolbox.population(n=POP)
    stats = register_metrics()

    start_time = time.time()
    # run the algorithm
    algorithms.eaSimple(pop, toolbox, CXPB, MUTPB, GEN, stats=stats, verbose=PRINT)
    end_time = time.time()
    duration = end_time - start_time

    # get the best individual
    best_ind = tools.selBest(pop, 1)[0]

    # print the best result
    p_print(best_ind, duration)
    return


if __name__ == "__main__":
    main()
