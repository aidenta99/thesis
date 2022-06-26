import pandas as pd
import numpy as np

# Generate school choice probability matrices
# Inputs: application_df, schools, education types
# Output: a dictionary in which key = education type, and value = corresponding school choice probability matrix

def generate_probability_matrix(application_df, schools: list, edu_types: str):
    p_matrix_dict = {}
    for edu_type in edu_types:
        df = application_df[application_df['Basisschool advies'] == edu_type]
        # Remove choice column that has no student fill in
        df.dropna(axis=1, how='all', inplace=True)
        choice_cols = [c for c in list(df.columns) if 'Voorkeur' in c]
        p_matrix = []

        for j in choice_cols:
            p_vector = []
            for i in schools:
                perc = len(df[df[j] == i]) / (df[j].notna().sum())
                p_vector.append(perc)
            p_matrix.append(p_vector)
        p_matrix_dict[edu_type] = p_matrix
    return p_matrix_dict

# Update probability vector of a choice, knowing the schools previously chosen
def update_p_vector(p, L):
    # Input: probability vector p, list of schools previously chosen with their indexes
    # Output: updated vector p
    # Algorithm: if the school is already in list L, change their corresponding probability p_ij to 0, then dividing their sum 
    # of probability equally over the rest of the vector element
    if len(L) == 0 or sum(p) == 0:
        return p
    else:
        chosen_school_index = [s[0] for s in L]
        chosen_school_p_sum = sum([p[i] for i in chosen_school_index])
        for i in range(len(p)):
            if i in chosen_school_index:
                p[i] = 0.0
            else:
                p[i] = p[i] + chosen_school_p_sum/(len(p) - len(L))
        # normalize p
        p = np.asarray(p).astype('float64')
        p = p / np.sum(p)
        return list(p)

# Generate one simulation, corresponding with a single application of a student
def baseline_single_simulation(schools: list, p_vector_list: list, edu_type: str):
    # schools: list of schools (string), e.g., ['SvPO Amsterdam - vwo', 'Amsterdams BN - v.a. vmbo-b', ...]
    # p_vector_list: a list of probability vectors, each vector corresponding to each ordered choice made by a student

    simulation_dict = {}
    simulation_dict['Basisschool advies'] = edu_type
    chosen_schools = []
    previous_choice = ''

    for i, p_vector in enumerate(p_vector_list):
        if previous_choice == 'None':
            current_choice = 'None'
        else:
            updated_p_vector = update_p_vector(p_vector, chosen_schools)
            current_choice = np.random.choice(schools, 1, p=updated_p_vector)[0]
            chosen_schools.append((schools.index(current_choice), current_choice))
        simulation_dict['Voorkeur {}'.format(i)] = current_choice
        previous_choice = current_choice
    
    return simulation_dict

# A function to generate a synthetic data of the same length with the input dataset, using random choice
# Input: application data (with engineered null values), school list (including 'None' value)
# Output: synthetic data
# Assumption: application df has the same columns with application_21, i.e., 'Basisschool advies', 'Voorkeur i'

def baseline_synthetic_school_choice(application_df, school_df):

    # Add 'None' into school list
    schools = list(school_df['Key'])
    schools.append('None')

    # Drop choice columns that are completely empty
    application_df.dropna(axis=1, how='all', inplace=True)

    # Fill in null values in choice columns by engineered 'None' string values, convenient to synthetic data simulation
    application_df.fillna("None", inplace=True)

    # Get 6 education types
    edu_types = application_df['Basisschool advies'].unique().tolist()

    # Calculate probability matrices
    p_matrices = generate_probability_matrix(application_df, schools, edu_types)

    simulations = []
    for e in edu_types:
        count = len(application_df[application_df['Basisschool advies'] == e])
        p_matrix = p_matrices.get(e)
        simulations_per_edu_type = []

    # Generate a synthetic dataset with the same length of the real one
        for i in range(count):
            simulation = baseline_single_simulation(schools, p_matrix, e)
            simulations_per_edu_type.append(simulation)
        simulations += simulations_per_edu_type
    baseline_simulation_df = pd.DataFrame(simulations)

    return baseline_simulation_df

# Bayesian network