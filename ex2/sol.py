import random

n=30
def read_preferences(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
    preferences = [tuple(map(int, line.strip().split())) for line in lines]
    return preferences[:n], preferences[n:]  # First 30 are men, next 30 are women

def generate_individual(size):
    individuals = list(range(1, size+1))
    random.shuffle(individuals)
    sol=[(i+1, individuals[i]) for i in range(size)]
    random.shuffle(sol)
    return sol

def generate_initial_population(size, num_individuals):
    population = []
    for _ in range(num_individuals):
        sol=generate_individual(size)
        population.append(sol)
    return population

# calculate fitness score of a solution
def calculate_fitness(solution, men_preferences, women_preferences):
    score = 0
    for man, woman in solution:
        man_pref = n - men_preferences[man-1].index(woman) # from 0 to n-1
        woman_pref = n - women_preferences[woman-1].index(man) # from 0 to n-1
        score += man_pref + woman_pref
    return score / (n*n*2) # normalize score to be between 0 and 1

def select_parents(population, fitness_scores):
    total_fitness = sum(fitness_scores)
    probabilities = [f/total_fitness for f in fitness_scores]
    parents = random.choices(population, probabilities, k=2)
    return parents

def crossover(parent1, parent2):
    size = len(parent1)
    point = random.randint(1, size - 1)
    child = parent1[:point] + parent2[point:]
    seen_women = set()
    seen_men = set()
    child= [ list(x) for x in child]
    for man, woman in child[:point]:
        seen_women.add(woman)
        seen_men.add(man)  
    for i, [man, woman] in enumerate(child[point:]):
        if man in seen_men :
            child[i+point][0] = random.choice(list(set(range(1, size + 1))-seen_men))  

            seen_men.add(child[i+point][0])
        if woman in seen_women:
            child[i+point][1] = random.choice(list(set(range(1, size + 1))-seen_women))
            seen_women.add(child[i+point][1])
        return tuple(tuple(x) for x in child)


def get_key_by_value(dictionary, value):
    for key, val in dictionary.items():
        if val == value:
            return key
    return None  # return None if the value doesn't exist in the dictionary

def mutate(solution, mutation_rate=0.05):
    solution= [ list(x) for x in solution]
    for i in range(len(solution)):
        if random.random() < mutation_rate:
            j = random.randint(0, len(solution) - 1)
            solution[i][0], solution[j][0] = solution[j][0], solution[i][0]

    return tuple(tuple(x) for x in solution)

def genetic_algorithm(men_preferences, women_preferences, num_generations=180, population_size=100):
    population = generate_initial_population(n, population_size)
    for _ in range(num_generations):
        fitnesses={}
        for solution in population:
            fitnesses[tuple(solution)] = calculate_fitness(solution, men_preferences, women_preferences)
        while len(fitnesses) != population_size:
            individual=generate_individual(n)
            fitnesses[tuple(individual)] = calculate_fitness(individual, men_preferences, women_preferences)
        new_population = []
        best_fitness_scores = sorted(fitnesses.values(), reverse=True)[:int(population_size*0.05)]
        best_solutions= [get_key_by_value(fitnesses,item) for item in best_fitness_scores]
        # elitism
        new_population.extend(best_solutions)
        for _ in range(population_size-int(population_size*0.05)):
            fitnesses_list = []
            for element in fitnesses:
                fitnesses_list.append(fitnesses[element])   
            parent1, parent2 = select_parents(population ,fitnesses_list)
            child1 = crossover(parent1, parent2)
            child1 = mutate(child1)
            new_population.append(child1)

        population = new_population
 

    best_solution = max(population, key=lambda x: calculate_fitness(x, men_preferences, women_preferences))
    best_fitness = calculate_fitness(best_solution,men_preferences, women_preferences)
    return best_solution, best_fitness

men_preferences, women_preferences = read_preferences('GA_input.txt')

best_match,best_fitness  = genetic_algorithm(men_preferences, women_preferences)
print("best solution:" , best_match)
print("best fitness: ", best_fitness)