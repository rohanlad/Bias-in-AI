import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import statistics
import copy

# Load in the data set
try:
    df = pd.read_csv('german_credit.csv')
except:
    print('Unable to locate the data set. Ensure that the german_credit.csv file exists in the current directory.')
    exit()

# Drop columns that we don't want
del df['Purpose']
del df['Guarantors']
del df['Type of apartment']
del df['Telephone']
del df['Foreign Worker']

# Our 2 demographic groups will be 'Old' and 'Young' where 'Old' is
# defined as those with age >= 25, and 'Young' everyone else
df_young = df[df['Age (years)'] < 25]
df_old = df[df['Age (years)'] >= 25]
young_indexes = df_young.index
old_indexes = df_old.index

# ****************************************************
# INITIAL ANALYSIS OF DATA SET:
# ****************************************************
def initial_analysis():

    print('\nThe size of the "young" group is: ' + str(df_young.shape[0]) + '\n')
    print('Mean values of each feature:' + '\n' + str(df_young.mean()) + '\n')
    print('Variance of each feature:' + '\n' + str(df_young.var()) + '\n')

    print('\nThe size of the "old" group is: ' + str(df_old.shape[0]) + '\n')
    print('Mean values of each feature:' + '\n' + str(df_old.mean()) + '\n')
    print('Variance of each feature:' + '\n' + str(df_old.var()) + '\n')

    return

# ****************************************************
# CONVENTIONAL IMPLEMENTATION - UNDIVERSE TESTING SET:
# ****************************************************
def conventional_undiverse():

    stratified = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train, test in stratified.split(df, df['Creditability']):
        strat_train = df.reindex(train)
        strat_test = df.reindex(test)

    X_train = strat_train.drop('Creditability', axis=1)
    X_test = strat_test.drop('Creditability', axis=1)
    Y_train = strat_train['Creditability']
    Y_test = strat_test['Creditability']

    clf = LogisticRegression(C=0.95, max_iter=50000, random_state=42, class_weight='balanced').fit(X_train, Y_train)
    Y_pred = clf.predict(X_test)
    
    young_good_outcomes = 0
    old_good_outcomes = 0
    young_total_outcomes = 0
    old_total_outcomes = 0
    for j in Y_test.index:
        if j in young_indexes:
            young_total_outcomes += 1
            if Y_pred[Y_test.index.get_loc(j)] == 1:
                young_good_outcomes += 1
        elif j in old_indexes:
            old_total_outcomes += 1
            if Y_pred[Y_test.index.get_loc(j)] == 1:
                old_good_outcomes += 1

    print('----------------------------------------------------------')
    print('Results for Undiverse Testing Set:')
    print('Disparate Impact: ' + str((young_good_outcomes/young_total_outcomes)/(old_good_outcomes/old_total_outcomes)))
    print('Zemel Fairness: ' + str(1 - ((old_good_outcomes/old_total_outcomes) - (young_good_outcomes/young_total_outcomes))))
    print('Accuracy: ' + str(accuracy_score(Y_test, Y_pred)))
    print('----------------------------------------------------------')
    
    return

# ****************************************************
# CONVENTIONAL IMPLEMENTATION - DIVERSE TESTING SET:
# ****************************************************
def conventional_diverse():

    stratified_o = StratifiedShuffleSplit(n_splits=1, test_size=100, random_state=42)
    for train_o, test_o in stratified_o.split(df_old, df_old['Creditability']):
        for k in range (0, len(train_o)):
            train_o[k] = (df_old.iloc[[train_o[k]]].index[0])
        for m in range (0, len(test_o)):
            test_o[m] = (df_old.iloc[[test_o[m]]].index[0])
        strat_train_o = df_old.reindex(train_o)
        strat_test_o = df_old.reindex(test_o)

    X_train_o = strat_train_o.drop('Creditability', axis=1)
    X_test_o = strat_test_o.drop('Creditability', axis=1)
    Y_train_o = strat_train_o['Creditability']
    Y_test_o = strat_test_o['Creditability']

    stratified_y = StratifiedShuffleSplit(n_splits=1, test_size=100, random_state=42)
    for train_y, test_y in stratified_y.split(df_young, df_young['Creditability']):
        for k in range (0, len(train_y)):
            train_y[k] = (df_young.iloc[[train_y[k]]].index[0])
        for m in range (0, len(test_y)):
            test_y[m] = (df_young.iloc[[test_y[m]]].index[0])
        strat_train_y = df_young.reindex(train_y)
        strat_test_y = df_young.reindex(test_y)

    X2_train = strat_train_y.drop('Creditability', axis=1)
    X2_test = strat_test_y.drop('Creditability', axis=1)
    Y2_train = strat_train_y['Creditability']
    Y2_test = strat_test_y['Creditability']

    X_train_c = pd.concat([X_train_o, X2_train])
    Y_train_c = pd.concat([Y_train_o, Y2_train])
    X_test_c = pd.concat([X_test_o, X2_test])
    Y_test_c = pd.concat([Y_test_o, Y2_test])

    clf = LogisticRegression(C=0.95, max_iter=50000, random_state=42, class_weight='balanced').fit(X_train_c, Y_train_c)
    Y_pred = clf.predict(X_test_c)

    young_good_outcomes = 0
    old_good_outcomes = 0
    young_total_outcomes = 0
    old_total_outcomes = 0
    for j in Y_test_c.index:
        if j in young_indexes:
            young_total_outcomes += 1
            if Y_pred[Y_test_c.index.get_loc(j)] == 1:
                young_good_outcomes += 1
        elif j in old_indexes:
            old_total_outcomes += 1
            if Y_pred[Y_test_c.index.get_loc(j)] == 1:
                old_good_outcomes += 1

    print('----------------------------------------------------------')
    print('Results for Diverse Testing Set:')
    print('Disparate Impact: ' + str((young_good_outcomes/young_total_outcomes)/(old_good_outcomes/old_total_outcomes)))
    print('Zemel Fairness: ' + str(1 - ((old_good_outcomes/old_total_outcomes) - (young_good_outcomes/young_total_outcomes))))
    print('Accuracy: ' + str(accuracy_score(Y_test_c, Y_pred)))
    print('----------------------------------------------------------')

    return

# ************************************************************
#     REPAIR FUNCTION & HELPER FUNCTIONS:
# ************************************************************
def modified_median(sequence):
    if (len(sequence) % 2) == 0:
        sequence.sort()
        return (sequence[int(len(sequence)/2) - 1])
    else:
        return statistics.median(sequence)

def unique_value_data_structures(columns):
    sorted_lists = {}
    index_lookups = {}
    for column in columns:
        sorted_list = []
        for c in column:
            if c not in sorted_list:
                sorted_list.append(c)
        sorted_list = sorted(sorted_list)
        sorted_lists[columns[column]] = sorted_list 
        index_lookup = {}
        for val in sorted_list:
            index_lookup[val] = sorted_list.index(val)
        index_lookups[columns[column]] = index_lookup
    return sorted_lists, index_lookups

def repair(requested_repair_type, columns, all_stratified_combinations, num_quantiles, sorted_lists, index_lookups, lambda_val):
    df_repaired = copy.deepcopy(df)
    for column in columns:
        group_offsets = {'young':-1,'old':-1}
        i = 0
        while i < num_quantiles:
            i += 1
            median_values_at_quantile = []
            entries_at_quantile = []
            for key in all_stratified_combinations:
                start = min((int(round(group_offsets[key] + 1))), (all_stratified_combinations[key]-1))
                end = min(int(round((min(((group_offsets[key] + 1)), (all_stratified_combinations[key]-1))) + (all_stratified_combinations[key]/num_quantiles) - 1)), (all_stratified_combinations[key]-1))
                group_offsets[key] = min(((min(((group_offsets[key] + 1)), (all_stratified_combinations[key]-1))) + (all_stratified_combinations[key]/num_quantiles) - 1), (all_stratified_combinations[key]-1))
                if key == 'young':
                    values = df_young.sort_values(by=[column])[start:(end+1)][column].values
                    median_values_at_quantile.append(modified_median(values))
                    for e in (df_young.sort_values(by=[column])[start:(end+1)].index):
                        entries_at_quantile.append(e)
                elif key == 'old':
                    values = df_old.sort_values(by=[column])[start:(end+1)][column].values
                    median_values_at_quantile.append(modified_median(values))
                    for e in (df_old.sort_values(by=[column])[start:(end+1)].index):
                        entries_at_quantile.append(e)
            target_value = modified_median(median_values_at_quantile)
            position_of_target = index_lookups[column][target_value]
            for entry in entries_at_quantile:
                value = df[column].values[entry]
                if requested_repair_type == 'combinatorial':
                    position_of_original_value = index_lookups[column][value]
                    distance = position_of_target - position_of_original_value
                    distance_to_repair = int(round(distance * lambda_val))
                    index_of_repair_value = position_of_original_value + distance_to_repair
                    repair_value = sorted_lists[column][index_of_repair_value]
                else:
                    repair_value = (1 - lambda_val)*value + (lambda_val*target_value)
                df_repaired.at[entry, column] = repair_value
    del df_repaired['Sex & Marital Status']
    del df_repaired['Age (years)']           
    return df_repaired


column_values = {}
column_names = []
for column in df:
    if column not in ['Age (years)', 'Sex & Marital Status', 'Creditability']:
        column_names.append(column)
        column_values[tuple(df[column].values)] = column

sorted_lists, index_lookups = unique_value_data_structures(column_values)
num_quantiles = min(df_young.shape[0], df_old.shape[0])

combinatorial_accuracy = []
combinatorial_young_accuracy = []
combinatorial_old_accuracy = []
combinatorial_utility = []
combinatorial_zemel = []
combinatorial_disparate_impact = []
geometric_accuracy = []
geometric_young_accuracy = []
geometric_old_accuracy = []
geometric_utility = []
geometric_zemel = []
geometric_disparate_impact = []
lambda_range = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]

def run_repair_iteratively():
    for l in lambda_range:
        for method in ['combinatorial', 'geometric']:
            # Repair call
            df_repaired = repair(method, column_names, {'young':df_young.shape[0], 'old':df_old.shape[0]}, num_quantiles, sorted_lists, index_lookups, l)

            stratified = StratifiedShuffleSplit(n_splits=1, test_size=0.33, random_state=42)
            for train, test in stratified.split(df_repaired, df_repaired['Creditability']):
                strat_train = df_repaired.reindex(train)
                strat_test = df_repaired.reindex(test)

            X_train = strat_train.drop('Creditability', axis=1)
            X_test = strat_test.drop('Creditability', axis=1)
            Y_train = strat_train['Creditability']
            Y_test = strat_test['Creditability']

            clf = LogisticRegression(C=0.95, max_iter=50000, random_state=42, class_weight='balanced').fit(X_train, Y_train)
            Y_pred = clf.predict(X_test)

            young_good_outcomes = 0
            old_good_outcomes = 0
            young_total_outcomes = 0
            old_total_outcomes = 0
            young_correct_outcomes = 0
            old_correct_outcomes = 0
            for j in Y_test.index:
                if j in young_indexes:
                    young_total_outcomes += 1
                    if Y_pred[Y_test.index.get_loc(j)] == 1:
                        young_good_outcomes += 1
                    if Y_pred[Y_test.index.get_loc(j)] == Y_test[j]:
                        young_correct_outcomes += 1
                elif j in old_indexes:
                    old_total_outcomes += 1
                    if Y_pred[Y_test.index.get_loc(j)] == 1:
                        old_good_outcomes += 1
                    if Y_pred[Y_test.index.get_loc(j)] == Y_test[j]:
                        old_correct_outcomes += 1
            
            if method == 'combinatorial':
                combinatorial_accuracy.append(accuracy_score(Y_test, Y_pred))
                combinatorial_young_accuracy.append(young_correct_outcomes/young_total_outcomes)
                combinatorial_old_accuracy.append(old_correct_outcomes/old_total_outcomes)
                combinatorial_utility.append(balanced_accuracy_score(Y_test, Y_pred))
                combinatorial_disparate_impact.append((young_good_outcomes/young_total_outcomes)/(old_good_outcomes/old_total_outcomes))
                combinatorial_zemel.append(1-((old_good_outcomes/old_total_outcomes) - (young_good_outcomes/young_total_outcomes)))
            elif method == 'geometric':
                geometric_accuracy.append(accuracy_score(Y_test, Y_pred))
                geometric_young_accuracy.append(young_correct_outcomes/young_total_outcomes)
                geometric_old_accuracy.append(old_correct_outcomes/old_total_outcomes)
                geometric_utility.append(balanced_accuracy_score(Y_test, Y_pred))
                geometric_disparate_impact.append((young_good_outcomes/young_total_outcomes)/(old_good_outcomes/old_total_outcomes))
                geometric_zemel.append(1-((old_good_outcomes/old_total_outcomes) - (young_good_outcomes/young_total_outcomes)))
    return

# ****************************************************
# FUNCTIONS TO PLOT GRAPHS:
# ****************************************************

def plot_young_old_accuracy_graph():
    x1 = lambda_range
    y1 = combinatorial_young_accuracy
    plt.plot(x1, y1, label = "Young (Comb.)")
    x2 = lambda_range
    y2 = combinatorial_old_accuracy
    plt.plot(x2, y2, label = "Old (Comb.)")
    x3 = lambda_range
    y3 = geometric_young_accuracy
    plt.plot(x3, y3, label = "Young (Geom.)")
    x4 = lambda_range
    y4 = geometric_old_accuracy
    plt.plot(x4, y4, label = "Old (Geom.)")
    plt.xlabel('Lambda')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
    return


def plot_general_accuracy_graph():
    x1 = lambda_range
    y1 = combinatorial_accuracy
    plt.plot(x1, y1, label = "Accuracy (Comb.)")
    x2 = lambda_range
    y2 = combinatorial_utility
    plt.plot(x2, y2, label = "Utility (Comb.)")
    x3 = lambda_range
    y3 = geometric_accuracy
    plt.plot(x3, y3, label = "Accuracy (Geom.)")
    x4 = lambda_range
    y4 = geometric_utility
    plt.plot(x4, y4, label = "Utility (Geom.)")
    plt.xlabel('Lambda')
    plt.ylabel('Performance')
    plt.legend()
    plt.show()
    return

def plot_fairness_graph():
    x1 = combinatorial_zemel
    y1 = lambda_range
    plt.scatter(x1, y1, label = "Zemel (Comb.)")
    x2 = combinatorial_disparate_impact
    y2 = lambda_range
    plt.scatter(x2, y2, label = "DI (Comb.)")
    x3 = geometric_zemel
    y3 = lambda_range
    plt.scatter(x3, y3, label = "Zemel (Geom.)")
    x4 = geometric_disparate_impact
    y4 = lambda_range
    plt.scatter(x4, y4, label = "DI (Geom.)")
    plt.xlabel('Fairness')
    plt.ylabel('Lambda')
    plt.legend()
    plt.show()
    return

# ****************************************************
# FUNCTION CALLS:
# ****************************************************
initial_analysis()
conventional_undiverse()
conventional_diverse()
print('Running fairness implementation... Please wait a few moments, the graphs will appear shortly.')
run_repair_iteratively()
plot_young_old_accuracy_graph()
plot_general_accuracy_graph()
plot_fairness_graph()