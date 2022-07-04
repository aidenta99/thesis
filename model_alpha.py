import pandas as pd
import numpy as np

from utils import encoding, get_school_list, preprocessing, decoding

# Generate school choice probability distributions per education type
# Inputs: encoded df, schools, education types
# Output: a dictionary in which key = education type, and value = corresponding school choice probability matrix

def generate_probability_distributions(df, n_schools, edu_types):
    dists_per_edu_type = {}
    for edu_type in edu_types:
        sub_df = df[df['Basisschool advies'] == edu_type]
        # df.dropna(axis=1, how='all', inplace=True)
        choice_cols = [c for c in list(df.columns) if 'Voorkeur' in c]
        dists = []

        for c in choice_cols:
            dist_per_choice_col = []
            for s in range(n_schools):
                p = len(sub_df[sub_df[c] == s]) / len(sub_df)
                dist_per_choice_col.append(p)
            dists.append(dist_per_choice_col)
        dists_per_edu_type[edu_type] = dists
    return dists_per_edu_type

# Update probability distribution of a school choice variable, given the values of the previous school choice variables
def update_distribution(dist, prev_choices):
    # Input: probability distribution dist, list of (encoded) schools previously chosen
    # Output: updated vector p
    # Algorithm: if the school is already in list prev_choices, change their corresponding probability p_ij to 0, 
    # then dividing their sum of probability equally over the rest of the distribution.
    if len(prev_choices) == 0 or sum(dist) == 0:
        return dist
    else:
        chosen_school_p_sum = sum([dist[i] for i in prev_choices])
        for i in range(len(dist)):
            if i in prev_choices:
                dist[i] = 0.0
            else:
                dist[i] = dist[i] + chosen_school_p_sum/(len(dist) - len(prev_choices))
        # normalize p
        dist = np.asarray(dist).astype('float64')
        dist = dist / np.sum(dist)
        return list(dist)

# Generate one simulation, corresponding with a single application of a student
def model_alpha_single_simulation(choice_col_dists, n_schools, edu_type):
    # choice_col_dists: a list of probability distributions

    simulation_dict = {}
    simulation_dict['Basisschool advies'] = edu_type
    chosen_schools = []
    previous_choice = -1
    num_choice_cols = len(choice_col_dists)

    for i in range(num_choice_cols):
        # If previous choice is the last one, aka encoded value of null (aka 182), 
        # then current choice is also (encoded) null
        if previous_choice == n_schools:
            current_choice = n_schools
        else:
            updated_dist = update_distribution(choice_col_dists[i], chosen_schools)
            current_choice = np.random.choice(range(n_schools), 1, p=updated_dist)[0]
            chosen_schools.append(current_choice)
        simulation_dict['Voorkeur {}'.format(i)] = current_choice
        previous_choice = current_choice
    
    return simulation_dict

# A function to generate a synthetic data of the same length with the input dataset, using random choice
# Input: application data (with engineered null values), school list (including 'None' value)
# Output: synthetic data
# Assumption: application df has the same columns with application_21, i.e., 'Basisschool advies', 'Voorkeur i'

def model_alpha_synthetic_generator(choices_df, schools_df):

    schools = get_school_list(schools_df)

    choices_df = preprocessing(choices_df)
    encoded_choices_df = encoding(choices_df, schools)

    # Get 6 education types
    edu_types = encoded_choices_df['Basisschool advies'].unique().tolist()

    # Calculate probability matrices
    dists_per_edu_type = generate_probability_distributions(encoded_choices_df, len(schools), edu_types)

    simulations = []
    for e in edu_types:
        count = len(encoded_choices_df[encoded_choices_df['Basisschool advies'] == e])
        dists = dists_per_edu_type.get(e)
        simulations_per_edu_type = []

    # Generate a synthetic dataset with the same length of the real one
        for i in range(count):
            simulation = model_alpha_single_simulation(dists, len(schools), e)
            simulations_per_edu_type.append(simulation)
        simulations += simulations_per_edu_type
    
    synthetic_df = pd.DataFrame(simulations)
    synthetic_df = decoding(synthetic_df, schools)

    return synthetic_df