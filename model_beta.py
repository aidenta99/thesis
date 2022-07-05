import pandas as pd
import numpy as np
from PrivBayes import greedy_bayes, construct_noisy_conditional_distributions
from utils import preprocessing, encoding, get_school_list, decoding
from numpy import random
from pandas import DataFrame

def get_sampling_order(bn):
    order = [bn[0][1][0]]
    for child, _ in bn:
        order.append(child)
    return order

def bayesian_network_synthetic_generator(n, bn, conditional_probabilities):
    bn_root_attr = bn[0][1][0]
    root_attr_dist = conditional_probabilities[bn_root_attr]
    synthetic_df = DataFrame(columns=get_sampling_order(bn))
    synthetic_df[bn_root_attr] = random.choice(len(root_attr_dist), size=n, p=root_attr_dist)

    for child, parents in bn:
        child_conditional_distributions = conditional_probabilities[child]
        for parents_instance in child_conditional_distributions.keys():
            dist = child_conditional_distributions[parents_instance]
            parents_instance = list(eval(parents_instance))

            # Resolve the error that probabilities do not sum up to 1
            dist = np.asarray(dist).astype('float64')
            dist = dist / np.sum(dist)

            filter_condition = ''
            for parent, value in zip(parents, parents_instance):
                filter_condition += f"(synthetic_df['{parent}']=={value})&"

            filter_condition = eval(filter_condition[:-1])
            
            size = synthetic_df[filter_condition].shape[0]
            if size:
                synthetic_df.loc[filter_condition, child] = random.choice(len(dist), size=size, p=dist)

    synthetic_df[synthetic_df.columns] = synthetic_df[synthetic_df.columns].astype(int)
    return synthetic_df

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

def simulate_remaining_choices(choice_col_dists, n_schools, edu_type, chosen_schools, k):
    # choice_col_dists: a list of probability distributions

    simulation_dict = {}
    previous_choice = chosen_schools[-1]
    num_choice_cols = len(choice_col_dists)

    for i in range(k, num_choice_cols):
        # If previous choice is the last one, aka encoded value of null (aka 182), 
        # then current choice is also (encoded) null
        if previous_choice == n_schools-1:
            current_choice = n_schools-1
        else:
            updated_dist = update_distribution(choice_col_dists[i], chosen_schools)
            current_choice = np.random.choice(range(n_schools), 1, p=updated_dist)[0]
            chosen_schools.append(current_choice)
        simulation_dict['Voorkeur {}'.format(i+1)] = current_choice
        previous_choice = current_choice
    
    return simulation_dict

def model_beta_synthetic_generator(choices_df, schools_df, epsilon=0.1, k=5):
    schools = get_school_list(schools_df)

    choices_df = preprocessing(choices_df)
    encoded_choices_df = encoding(choices_df, schools)

    # Get 6 education types
    edu_types = encoded_choices_df['Basisschool advies'].unique().tolist()

    # Calculate probability matrices
    dists_per_edu_type = generate_probability_distributions(encoded_choices_df, len(schools), edu_types)
    
    simulations = []

    for e in edu_types:
        dists = dists_per_edu_type.get(e)

        subset = encoded_choices_df[encoded_choices_df['Basisschool advies'] == e]
        chosen_cols = ['Voorkeur {}'.format(i) for i in range(1, k+1)]
        first_choices_subset = subset[chosen_cols]

        bn = greedy_bayes(first_choices_subset, k=0, epsilon=0.1 / 2, seed=0)
        conditional_probabilities = construct_noisy_conditional_distributions(bn, first_choices_subset, epsilon/2)

        synthetic_first_k_choice_cols = bayesian_network_synthetic_generator(len(subset), bn, conditional_probabilities)
        first_k_choice_cols_dicts = synthetic_first_k_choice_cols.to_dict('records')
    
        # synthetic_first_k_choices['Basisschool advies'] = e

        subset_rows = []
        for i in range(len(subset)):
            first_k_choices = first_k_choice_cols_dicts[i]
            remaining_choices_per_row = simulate_remaining_choices(dists, len(schools), e, list(first_k_choices.values()), k)
            row = {**first_k_choices, **remaining_choices_per_row}
            row['Basisschool advies'] = e
            subset_rows.append(row)
        
        simulations += subset_rows
    
    synthetic_df = pd.DataFrame(simulations)
    cols = list(encoded_choices_df.columns)
    synthetic_df = synthetic_df[cols]
    synthetic_df = decoding(synthetic_df, schools)

    return synthetic_df
