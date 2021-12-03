import pandas as pd
import numpy as np
import nltk
from nltk.parse.generate import generate
import itertools
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import collections
from collections import OrderedDict

''' THIS VERSION OF THE EQUATION DISCOVERY TOOL USES AN INFORMED SEARCH WITH A BACKWARDS STEP POSSIBLE IN ORDER TO AVOID A LOCAL MINIMUM '''

example_grammar = nltk.CFG.fromstring("""
  E -> E Plus Term | Term
  Term -> 'const' | 'const' Times 'v' | 'const' Times LT Times LT | 'const' Times v Times LT | 'const' Times LT
  LT -> 'np.sin(' 'const' Times 'v' Plus 'const )'
  Plus -> '+'
  Times -> '*'
  """)

additional_terms = [
    ' + const',
    ' + const * v',
    ' + const * np.sin( const * v + const ) * np.sin( const * v + const )',
    ' + const * v * np.sin( const * v + const )',
    ' + const * np.sin( const * v + const )',
]

def equation_list(grammar, d):
    sentences = []
    for s in generate(grammar, depth=d):
        sentences.append(' '.join(s))
    return sentences

def eq_reader(eq):
    '''
    Replaces any instances of const with {constX} where X is the position of the constants in the equation.
    
    parameter(s) = eq - string that is an equation
    returns = joined up equation list joined on spaces
    '''
    eq_list = eq.split()
    const_count = 0
    for i, param in enumerate(eq_list):
        if param == 'const':
            eq_list[i] = '{const' + str(const_count) + '}'
            const_count += 1
    return (' '.join(eq_list))

def find_char_indexes(s, ch):
    '''
    Finds the index of characters in a string.
    
    parameter(s): s - string
                  ch - character to search for
    returns: the index of the character in the string
    '''
    return [i for i, ltr in enumerate(s) if ltr == ch]

def create_lambda(equation_string):
    '''
    Finds the values of a variable (column) in the CSV so it can fit them.

    parameter(s): equation_string - string of the equation
    returns: func
    '''
    def func(X, *params):
        vals_map = {"const"+str(i):params[i] for i in range(len(params))}
        actual_equation_string = equation_string.format(**vals_map)
        return eval(actual_equation_string)
    return func

def param_fitter(eq_list, target, df):
    '''
    Creates test and train dataframes,
    Creates X-test and X-train by dropping the target and date columns of the dataframe,
    Repeats dropping of columns for the test dataframes,
    Initialises the top_10_models dictionary in order to store the best performing models by MSE,
    Loops over all the equations in eqn list,
        If it contains a 'v' (variable) in the equation string, enter the if statement
        Form equation string using the equation reader statement (Make any 'const' '{constX}')
        FOR LOOP
            Get all the 'v' indexes in the equation string
            Get list of column indexes
            Adds in permutation of variables where the 'v's are in the equation string (for loop)
            Run the equations through curve fit (linear regression)
            Calculate the MSE value for the equation and store in the top_10_models dictionary if it's a good enough MSE score
    
    parameter(s): eq_list - list of equation strings
                  target - the target variable being modelled
                  df - the datafram which uses forward fill for missing data (in quarterly data)
    returns: top_10_models - the dictionary containing the top 10 models { MSE : (Equation string post processing, constant values) }

    '''

    # Split the dataframe into train and test set, based on the target parameter
    df_train = df[df['DATE']  < '2015-01-01'].fillna(method='ffill')
    df_test = df[df['DATE'] >= '2015-01-01'].fillna(method='ffill')
    X_train = df_train.drop(columns=[target, 'DATE'])
    X_test = df_test.drop(columns=[target, 'DATE'])
    y_train = df_train[target]
    y_test = df_test[target]
    
    top_10_models = dict()
    # Remove any equations generated that have already been modelled
    if unvisited_eqns:
        for eqn_tuple in unvisited_eqns:
            if eqn_tuple[0] in eq_list:
                eq_list.remove(eqn_tuple[0])
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
                try:
                    popt, pcov = curve_fit(func, X_train.values, y_train.values, p0=priors, maxfev=10000)

                    # Only uncomment if not fitting many equations - may run out 
                    # of memory to display otherwise:
                    # plt.rcParams["figure.figsize"] = (20,20)
                    # plt.plot(df['DATE'],  df_test.values, 'b-', label='data')
                    # plt.plot(df['DATE'], func(df_test.values, *popt), 'r-')
                    
                    # Find the mse for the fitted model for the 
                    mse = np.mean((y_test.values-func(X_test.values, *popt))**2)
                    if len(top_10_models) < 10:
                        top_10_models[mse] = (equation_string, popt, e)
                    elif mse < max(top_10_models):
                        del top_10_models[max(top_10_models)]
                        top_10_models[mse] = (equation_string, popt, e)
                        # max_key = max(top_10_models) # max(top_10_models, key=top_10_models.get)

                    # Add the equation and the best mse score to the unvisited_eqns list (eqn, mse)
                    if unvisited_eqns:
                        if (e == unvisited_eqns[-1][0]) and (mse < unvisited_eqns[-1][1]):
                            del unvisited_eqns[-1]
                            unvisited_eqns.append((e, mse))
                        else:
                            unvisited_eqns.append((e, mse))
                    else:
                        unvisited_eqns.append((e, mse))
                    plt.show()
                    print("Mean Squared Error: " + str(mse))
                except RuntimeError:
                    print("Bad equation, couldn't find a fit")
    return top_10_models

def graph_deepen():
    unvisited_eqns.sort(key = lambda x: x[1])
    visited_eqns.append(unvisited_eqns.pop(0))
    revised_eqn_list = add_terms(visited_eqns[-1][0])
    return revised_eqn_list

def add_terms(eqn):
    new_eqn_list = []
    for ad_term in additional_terms:
        new_eqn_list.append(eqn + ad_term)
    print(new_eqn_list)
    return new_eqn_list

# Fetch dataset and discard unnecessary values
df = pd.read_csv("USMacroDataset2021.csv")
df.drop(df.tail(4).index,inplace=True)
df['DATE'] = pd.to_datetime(df['DATE'])

results = dict()
# Generate equations for each variable in dataset
for col in df.drop(columns='DATE').columns:
    ################ Incressing the depth parameter for equation_list() will result in a large increase in the number of equations to fit ##############################
    # Create visited and unvisited equation queues
    visited_eqns = []
    unvisited_eqns = []
    eq_list = equation_list(example_grammar, 5)
    top_10_1 = param_fitter(eq_list, col, df)
    top_10_2 = param_fitter(graph_deepen(), col, df)
    top_10_3 = param_fitter(graph_deepen(), col, df)
    # top_10_4 = param_fitter(graph_deepen(), col, df) 
    # top_10_5 = param_fitter(graph_deepen(), col, df) 
    results[col] = top_10_1
    results[col] = top_10_2
    results[col] = top_10_3
    # results[col] = top_10_4
    # results[col] = top_10_5
    print("Top models by MSE for " + col + ": " + str(top_10_3))
    print('\n')
    print('\n')
    print('\n')
print('\n')
print(results)

# CPI
# DEPTH 2
# 24.234950305510058
# ('{const0} + {const1} * X[:,1] + {const2} * X[:,0] + {const3} * np.sin( {const4} * X[:,0] + {const5} ) * np.sin( {const6} * X[:,1] + {const7} )', 
# array([-3.18875471e+01,  1.97398084e+00,  1.50591290e-02,  2.84114051e+02, 9.99866808e-01,  3.26019470e+00,  3.92281032e-03,  6.25463798e+00]), 
# 'const + const * v + const * v + const * np.sin( const * v + const ) * np.sin( const * v + const )')

# DEPTH 3
# 16.811307593380192
# ('{const0} + {const1} * X[:,1] + {const2} * X[:,1] + {const3} * np.sin( {const4} * X[:,1] + {const5} ) * np.sin( {const6} * X[:,1] + {const7} ) + {const8} 
# * X[:,0] * np.sin( {const9} * X[:,1] + {const10} )', 
# array([-2.16121787e+01, -4.98976880e+06,  4.98976895e+06, -1.70172544e+00, 1.42974356e+06,  8.53248912e+05, -1.42978783e+06, -8.52944982e+05,
# -2.06182870e-02, -1.47610733e-02,  5.55034863e+00]), 
# 'const + const * v + const * v + const * np.sin( const * v + const ) * np.sin( const * v + const ) + const * v * np.sin( const * v + const )')

# GDP
# DEPTH 2
# 179791.38921228296
# ('{const0} * X[:,0] + {const1} * X[:,0] + {const2} * np.sin( {const3} * X[:,1] + {const4} ) * np.sin( {const5} * X[:,1] + {const6} )', 
# array([ -151.63388761,   222.5383063 ,  1346.79560375,  5546.14461102, -2507.51211156, -5512.02913636,  2486.55585323]), 
# 'const * v + const * v + const * np.sin( const * v + const ) * np.sin( const * v + const )')

# DEPTH 3
# 208628.5046557647
# ('{const0} * X[:,0] + {const1} * X[:,0] + {const2} * np.sin( {const3} * X[:,0] + {const4} ) * np.sin( {const5} * X[:,0] + {const6} ) + {const7} * X[:,0] 
# * np.sin( {const8} * X[:,0] + {const9} )', 
# array([-5.96680782e+04,  5.97427183e+04, -1.87924438e+02, -4.14765892e+03, -2.47767683e+06,  4.15044061e+03,  2.47743333e+06, -3.85053181e-01, 1.01867803e+00, -1.37379113e+00]), 
# 'const * v + const * v + const * np.sin( const * v + const ) * np.sin( const * v + const ) + const * v * np.sin( const * v + const )')

# UNEMP
# DEPTH 2
# 4.0108419831524875
# ('{const0} + {const1} * np.sin( {const2} * X[:,0] + {const3} ) + {const4} * X[:,0] * np.sin( {const5} * X[:,1] + {const6} )', 
# array([ 5.81720562e+00, -4.72641208e-01,  1.00229082e+00,  4.47764719e-01, 2.07413660e-03,  1.00038764e+00, -3.37942793e+00]), 
# 'const + const * np.sin( const * v + const ) + const * v * np.sin( const * v + const )')

# DEPTH 3
# 3.7981580051719788
# ('{const0} + {const1} * np.sin( {const2} * X[:,1] + {const3} ) + {const4} * X[:,0] * np.sin( {const5} * X[:,1] + {const6} ) + {const7}', 
# array([-2.82098648e+05,  3.08500722e-01,  9.99177126e-01,  5.07599625e+00, 2.06601603e-03,  1.00031495e+00, -2.81735411e+00,  2.82104486e+05]), 
# 'const + const * np.sin( const * v + const ) + const * v * np.sin( const * v + const ) + const')
