# This file acts as proof that regardless of class distribution
# the chances of choosing said class are simply 1/n where n
# is the number of classes.

# It acts as proof that my 30% accuracy in my regular classification
# is, indeed, better than guessing.

from random import randint

def histo_choice(population_map, total_population):
    selection = randint(1, total_population)

    for class_identifier, population in population_map.items():
        if selection <= population:
            return class_identifier
        else:
            selection -= population
            continue

# This class acts as the container for the distributions
# Numbers can, obviously, be modified and will present the same results
class_distribution = {
    'not_cat': 40,
    'cat_fat_1': 12,
    'cat_fat_2': 12,
    'cat_fat_3': 12,
    'cat_fat_4': 12,
    'cat_fat_5': 12
}

random_guess = {
    'not_cat': 1,
    'cat_fat_1': 1,
    'cat_fat_2': 1,
    'cat_fat_3': 1,
    'cat_fat_4': 1,
    'cat_fat_5': 1
}

number_guesses = 100000
correct_guesses = 0

for _ in range(0, 100000):
    chosen_class = histo_choice(random_guess, 6)
    actual_class = histo_choice(class_distribution, 100)

    if chosen_class == actual_class:
        correct_guesses += 1

print(float(correct_guesses) / float(number_guesses))
