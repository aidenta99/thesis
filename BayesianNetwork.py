import numpy as np

def update_distribution(dist, indices):
    '''
    Update the probabilities of schools that already appeared in previous choices to zero,
    to avoid duplicates of choices.
    Divide probability weight equally over the rest of the distribution.
    '''
    probability_sum_schools_already_in_previous_choices = sum(dist[index] for index in indices)
    for index, probability in enumerate(dist):
        if index in indices:
            dist[index] = 0.0
        else:
            dist[index] = probability + probability_sum_schools_already_in_previous_choices/(len(dist) - len(indices))
    return dist




