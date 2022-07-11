# Thesis

## Simulating realistic school choice data using Bayesian networks

In this project we attempt to synthesize realistic versions of Amsterdam School Choice data. 

The school choice data consists of anonymized students, their assigned lottery number, school type (Advies), their preference list of schools, and the school they were finally assigned to by the Deferred Acceptance algorithm.

In this notebook we fit a Bayesian model over the original data and attempt to synthesize new data. 

In order for the generated data to be consistent it must follow the following constraints:

1. Each student has at least one school choice
2. Each student has at most 22 school choices
3. Students do not pick schools that do not fit their Advies
4. Students may not pick the same school twice

## Running the notebook

Using conda, import the needed libraries using

```
conda create --name <envname> --file requirements.txt
```

Now run the Jupyter notebook

```
jupyter notebook --allow-root
```

