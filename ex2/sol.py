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

        man_pref = n - men_preferences[man-1].index(woman) # from 1 to n
        woman_pref = n - women_preferences[woman-1].index(man) # from 1 to n

        score += man_pref + woman_pref
    return (score -2 ) / (n*n*2 -2) # normalize score to be between 0 and 1

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
        return child



def mutate(solution, mutation_rate=0.05):
    for i in range(len(solution)):
        if random.random() < mutation_rate:
            j = random.randint(0, len(solution) - 1)
            solution[i][0], solution[j][0] = solution[j][0], solution[i][0]
    return solution

def genetic_algorithm(men_preferences, women_preferences, num_generations=100, population_size=180):
    population = generate_initial_population(n, population_size)
    for _ in range(num_generations):
        new_population = []
        fitness_scores = [calculate_fitness(solution, men_preferences, women_preferences) for solution in population]
        print('fitness_scores: ', sorted(fitness_scores, reverse=True)[:1])
        # elitism
        best_solutions = sorted(population, key=lambda x: fitness_scores[population.index(x)], reverse=True)[:int(population_size*0.05)]
        new_population.extend(best_solutions)
       

        # print('len!!! ', len(best_solutions))
        # for i in range(5):
            # print('best_solutions: ', best_solutions[i])
            # print('best_solutions: ', calculate_fitness(best_solutions[i],men_preferences, women_preferences))
      
        for _ in range(population_size-int(population_size*0.05)):
            parent1, parent2 = select_parents(population,fitness_scores)
            child1 = crossover(parent1, parent2)
            child1 = mutate(child1)
            new_population.extend([child1])
        # print('best_solutions: ', best_solutions[0])

        population = new_population
 

    best_solution = max(population, key=lambda x: calculate_fitness(x, men_preferences, women_preferences))
    return best_solution




men_preferences, women_preferences = read_preferences('GA_input.txt')

best_match = genetic_algorithm(men_preferences, women_preferences)
print("Best Match:", best_match)
print("Fitness Score:", calculate_fitness(best_match,men_preferences, women_preferences ))



