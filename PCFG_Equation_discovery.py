from nltk.grammar import Nonterminal
import pandas as pd
import numpy as np
import nltk
from nltk.parse.generate import generate
import itertools
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from nltk.corpus import treebank
from nltk import treetransforms
from nltk import induce_pcfg
from nltk.parse import pchart
from pcfg import PCFG

example_grammar = PCFG.fromstring("""
  E -> E Plus Term [0.4] | Term [0.6]
  Term -> 'const' [0.4] | 'const' Times 'v' [0.4] | 'const' Times LT Times LT [0.05] | 'const' Times 'v' Times LT [0.05] | 'const' Times LT [0.1]
  LT -> 'np.sin(' 'const' Times 'v' Plus 'const )' [1.0]
  Plus -> '+' [1.0]
  Times -> '*' [1.0]
  """)

def equation_list(grammar, d):
    sentences = []
    for s in grammar.generate(84):
        print(s)
        sentences.append(''.join(s))
    return sentences

def eq_reader(eq):
    hasC = False
    eq_list = eq.split()
    const_count = 0
    for i, param in enumerate(eq_list):
        if param == 'const':
            eq_list[i] = '{const' + str(const_count) + '}'
            const_count += 1
    return (' '.join(eq_list))

def find_char_indexes(s, ch):
    return [i for i, ltr in enumerate(s) if ltr == ch]

def create_lambda(equation_string):
    def func(X, *params):
        vals_map = {"const"+str(i):params[i] for i in range(len(params))}
        actual_equation_string = equation_string.format(**vals_map)
        return eval(actual_equation_string)
    return func

def param_fitter(eq_list, target, df):
    # Split the dataframe into train and test set, based on the target parameter
    df_train = df[df['DATE']  < '1999-03-01'].fillna(method='ffill')
    df_val = df[df['DATE'] >= '1999-03-01'].fillna(method='ffill')
    df_val = df_val[df_val['DATE'] < '2010-03-01'].fillna(method='ffill')
    df_test = df[df['DATE'] >= '2010-03-01'].fillna(method='ffill')
    X_train = df_train.drop(columns=[target, 'DATE'])
    X_val = df_val.drop(columns=[target, 'DATE'])
    X_test = df_test.drop(columns=[target, 'DATE'])
    y_train = df_train[target]
    y_val = df_val[target]
    y_test = df_test[target]

    top_10_models = dict()
    for e in eq_list:
        if e.count('v')>0: # Only interested in equations with 1 or more input 
                           # variable (x=1 isn't a great forecasting model)
            equation_string = eq_reader(e)
            # Find all positions where input variables will need to be inserted
            v_indices = find_char_indexes(equation_string, 'v') 
            # Get a list of column indices (e.g for 3 inputs, x_cols=[0,1,2]
            x_cols = [x for x in range(len(X_train.columns))]
            
            # Returns all possible permutations for the equation of inserting input variables
            for perm in itertools.product(x_cols,repeat=len(v_indices)):
                equation_string = eq_reader(e)
                for i in range(len(perm)): 
                    v_indices = find_char_indexes(equation_string, 'v')
                    equation_string = equation_string[:v_indices[0]] + 'X[:,' + str(perm[i]) + ']' + equation_string[v_indices[0]+1:]

                print(equation_string)
                #run the curve fit
                func = create_lambda(equation_string)
                # We need to pass prior assumptions for each parameter - we pass a list of 1's
                priors = [1 for c in range(equation_string.count('{'))] 
                popt, pcov = curve_fit(func, X_train.values, y_train.values, p0=priors, maxfev=1000000)
                # Only uncomment if not fitting many equations - may run out 
                # of memory to display otherwise:
                #plt.rcParams["figure.figsize"] = (20,20)
                #plt.plot(df['DATE'],  y.values, 'b-', label='data')
                #plt.plot(df['DATE'], func(X.values, *popt), 'r-')
                
                # Find the mse for the fitted model for the 
                mse = np.mean((y_val.values-func(X_val.values, *popt))**2)
                if len(top_10_models) < 10:
                    top_10_models[mse] = (equation_string, popt)
                elif mse < max(top_10_models):
                    del top_10_models[max(top_10_models)]
                    top_10_models[mse] = (equation_string, popt)
                    # max_key = max(top_10_models) # max(top_10_models, key=top_10_models.get)
                    
                plt.show()
                print("Mean Squared Error: " + str(mse))
    return top_10_models

# Fetch dataset and discard unnecessary values
df = pd.read_csv("USMacroDataset2021.csv")
df.drop(df.tail(4).index,inplace=True)
df['DATE'] = pd.to_datetime(df['DATE'])

results = dict()
# Generate equations for each variable in dataset
for col in df.drop(columns='DATE').columns:
    ################ Incressing the depth parameter for equation_list() will result in a large increase in the number of equations to fit ##############################
    eq_list = equation_list(example_grammar, 84)
    top_10 = param_fitter(eq_list, col, df)
    results[col] = top_10
    print("Top models by MSE for " + col + ": " + str(top_10))
    print('\n')
    print('\n')
    print('\n')
print('\n')
print(results)
