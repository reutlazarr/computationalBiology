def test_crossover():
    parent1 = [(1, 2), (2, 3), (3, 1)]
    parent2 = [(1, 3), (2, 1), (3, 2)]
    
    child = crossover(parent1, parent2)
    
    # Check if the child contains all individuals from both parents
    assert set(child) == set(parent1 + parent2)
    
    # Check if the child has the same length as the parents
    assert len(child) == len(parent1)
    
    # Check if the child has unique individuals
    assert len(set(child)) == len(child)
    
    # Check if the child has the same number of each individual as the parents
    for individual in child:
        assert child.count(individual) == parent1.count(individual) + parent2.count(individual)
    
    print("crossover function test passed!")

test_crossover()