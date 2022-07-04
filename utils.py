import json
import random
import numpy as np
from pandas import Series, DataFrame
from sklearn.metrics import mutual_info_score, normalized_mutual_info_score

def set_random_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)

# Functions used for greedy Bayesian algorithms

def mutual_information(labels_x: Series, labels_y: DataFrame):
    """Mutual information of distributions in format of Series or DataFrame.

    Parameters
    ----------
    labels_x : Series
    labels_y : DataFrame
    """
    if labels_y.shape[1] == 1:
        labels_y = labels_y.iloc[:, 0]
    else:
        labels_y = labels_y.apply(lambda x: ' '.join(x.values), axis=1)

    return mutual_info_score(labels_x, labels_y)


def pairwise_attributes_mutual_information(dataset):
    """Compute normalized mutual information for all pairwise attributes. Return a DataFrame."""
    sorted_columns = sorted(dataset.columns)
    mi_df = DataFrame(columns=sorted_columns, index=sorted_columns, dtype=float)
    for row in mi_df.columns:
        for col in mi_df.columns:
            mi_df.loc[row, col] = normalized_mutual_info_score(dataset[row].astype(str),
                                                               dataset[col].astype(str),
                                                               average_method='arithmetic')
    return mi_df


def normalize_given_distribution(frequencies):
    distribution = np.array(frequencies, dtype=float)
    distribution = distribution.clip(0)  # replace negative values with 0
    summation = distribution.sum()
    if summation > 0:
        if np.isinf(summation):
            return normalize_given_distribution(np.isinf(distribution))
        else:
            return distribution / summation
    else:
        return np.full_like(distribution, 1 / distribution.size)


# Preprocessing choice data
def preprocessing(df):
    # Drop rows that are 'Niet geplaatst' (not complete)
    df.drop(df[df['Geplaatst op'] == 'Niet geplaatst'].index, inplace=True)
    df.reset_index(inplace=True, drop=True)

    # Drop columns with only null values
    df.dropna(axis=1, how='all', inplace=True)

    # Get choice columns, keep 'Basisschool advies' and choice columns
    choice_cols = [c for c in list(df.columns) if 'Voorkeur' in c]
    selected_cols =  ['Basisschool advies'] + choice_cols
    df = df[selected_cols]

    return df

# Get the list of schools from school dataframe, and add "NA" for null values
def get_school_list(df):
    schools = list(df['Key'])
    schools.append('NA')
    return schools

# Encoding choice data, using indices of school list
def encoding(df, schools):

    # Get school indices from the school list dataframe
    school_indices_dict = dict(zip(schools, range(len(schools))))

    encoded_null = school_indices_dict['NA']

    choice_cols = [c for c in list(df.columns) if 'Voorkeur' in c]

    for choice_col in choice_cols:
        df[choice_col] = df[choice_col].map(school_indices_dict)
        df[choice_col].fillna(encoded_null, inplace=True)
        df[choice_col] = df[choice_col].astype(int, copy=False)
    
    return df

def decoding(df, schools):
    # Input: encoded dataframe, decoding it into the original format
    school_indices_dict = dict(zip(range(len(schools)), schools))
    choice_cols = [c for c in list(df.columns) if 'Voorkeur' in c]
    for choice_col in choice_cols:
        df[choice_col] = df[choice_col].map(school_indices_dict)
        df[choice_col] = df[choice_col].astype(str, copy=False)

    return df


# Display BN

def display_bayesian_network(bn):
    length = 0
    for child, _ in bn:
        if len(child) > length:
            length = len(child)

    print('Constructed Bayesian network:')
    for child, parents in bn:
        print("    {0:{width}} has parents {1}.".format(child, parents, width=length))


# Save dataset description

def save_dataset_description_to_file(data_description, file_name):
    with open(file_name, 'w') as outfile:
        json.dump(data_description, outfile, indent=4)

def display_dataset_description(data_description):
    print(json.dumps(data_description, indent=4))