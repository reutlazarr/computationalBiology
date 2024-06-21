import random
n=30
def read_preferences(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
    preferences = [list(map(int, line.strip().split())) for line in lines]
    
    return preferences[:n], preferences[n:]  # First 30 are men, next 30 are women

def generate_initial_population(size, num_individuals):
    population = []
    for _ in range(num_individuals):
        individuals = list(range(1, size+1))
        random.shuffle(individuals)
        sol=[[i+1, individuals[i]] for i in range(size)]
        random.shuffle(sol)
        population.append(sol)
    return population

# calculate fitness score of a solution
def calculate_fitness(solution, men_preferences, women_preferences):
    score = 0
    for man, woman in solution:

        man_pref = n - men_preferences[man-1].index(woman)
        woman_pref = n - women_preferences[woman-1].index(man)

        score += man_pref + woman_pref
    return score

def select_parents(population, fitness_scores):
    total_fitness = sum(fitness_scores)
    probabilities = [f/total_fitness for f in fitness_scores]
    parents = random.choices(population, probabilities, k=2)
    return parents

def crossover(parent1, parent2):
    point = random.randint(1, len(parent1) - 1)
    child = parent1[:point] + parent2[point:]
    
    seenWoman = set()
    seenMan = set()
    for man, woman in child:
        if woman in seenWoman:
            available_women = set(range(1, n+1)) - seenWoman
            woman = random.choice(list(available_women))
        seenWoman.add(woman)
        
        if man in seenMan:
            available_men = set(range(1, n+1)) - seenMan
            man = random.choice(list(available_men))
        seenMan.add(man)
    
    return [[man, woman] for man, woman in zip(seenMan, seenWoman)]


def mutate(solution, mutation_rate=0.05):
    for i in range(len(solution)):
        if random.random() < mutation_rate:
            j = random.randint(0, len(solution) - 1)
            solution[i][0], solution[j][0] = solution[j][0], solution[i][0]
    return solution

def genetic_algorithm(men_preferences, women_preferences, num_generations=10, population_size=50):
    population = generate_initial_population(n, population_size)
    for _ in range(num_generations):
        new_population = []
        fitness_scores = [calculate_fitness(solution, men_preferences, women_preferences) for solution in population]
        print('fitness_scores: ', fitness_scores[0])
        for _ in range(population_size-int(population_size*0.05)):
            parent1, parent2 = select_parents(population,fitness_scores)
            child1 = crossover(parent1, parent2)
            child1 = mutate(child1)
            new_population.extend([child1])
        best_solutions = sorted(population, key=lambda x: fitness_scores[population.index(x)])[:int(population_size*0.05)]
        print('best_solutions: ', best_solutions[0])
        new_population.extend(best_solutions)
        population = new_population
 

    best_solution = max(population, key=lambda x: calculate_fitness(x, men_preferences, women_preferences))
    return best_solution




men_preferences, women_preferences = read_preferences('GA_input.txt')

best_match = genetic_algorithm(men_preferences, women_preferences)
print("Best Match:", best_match)
print("Fitness Score:", calculate_fitness(best_match,men_preferences, women_preferences ))



