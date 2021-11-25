if __name__ == "__main__":
    from coordinate_descent_algo import train  # type: ignore
else:
    from .coordinate_descent_algo import train

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils import load_dataset, problem

def init_lambda(df_train_target, df_train_observations):

    target_mean = df_train_target.mean()
    ls_lambda = []

    for j in range(len(df_train_observations.columns)):
        rolling_x_ik = 0
        for i in range(len(df_train_observations)):

            rolling_x_ik = rolling_x_ik + df_train_observations.iloc[i][j] * (df_train_target[i] - target_mean)

        ls_lambda.append(2*abs(rolling_x_ik))

    return max(ls_lambda)

def mse(actual, pred):
    actual, pred = np.array(actual), np.array(pred)
    return np.square(np.subtract(actual,pred)).mean()

def se(actual, pred):
    actual, pred = np.array(actual), np.array(pred)
    return np.square(np.subtract(actual, pred))



@problem.tag("hw2-A", start_line=3)
def main():
    # df_train and df_test are pandas dataframes.
    # Make sure you split them into observations and targets

    df_train, df_test = load_dataset("crime")

    #calculate lambda max
    df_train_target = df_train['ViolentCrimesPerPop']
    df_train_observations = df_train.drop('ViolentCrimesPerPop', 1)

    df_test_target = df_test['ViolentCrimesPerPop']
    df_test_observations = df_test.drop('ViolentCrimesPerPop', 1)

    #Get all columns
    all_col = df_train_observations.columns.values.tolist()

    lambda_init = init_lambda(df_train_target, df_train_observations)

    non_zero_ls = []
    lambda_ls = []
    train_mse_ls = []
    test_mse_ls = []

    #plots for regularization paths
    Regularization_plot_ls = ["agePct12t29", "pctWSocSec", "pctUrban", "agePct65up", "householdsize"]
    Regularization_indexes = [all_col.index(item) for item in Regularization_plot_ls]
    Regularization_plot_results = []




    while lambda_init > 0.01:

        print('Select Lambda is ' + str(lambda_init))

        weight, bias = train(df_train_observations.to_numpy(), df_train_target.to_numpy(), lambda_init)
        count_nonzero = np.count_nonzero(weight)
        non_zero_ls.append(count_nonzero)
        lambda_ls.append(lambda_init)

        lambda_init = lambda_init/2

        df_train_model = df_train_observations.dot(weight.T)
        df_test_model = df_test_observations.dot(weight.T)

        train_mse = mse(df_train_model, df_train_target)
        test_mse = mse(df_test_model, df_test_target)

        train_mse_ls.append(train_mse)
        test_mse_ls.append(test_mse)


        #store regularization paths

        reg_ls = []
        for i in Regularization_indexes:
            beta = df_train_model[i]
            reg_ls.append(beta)

        Regularization_plot_results.append(reg_ls)



    mse_df = pd.DataFrame({'train_mse': train_mse_ls, 'test_mse':test_mse_ls})
    df = pd.DataFrame({'lambda': lambda_ls, 'count_non_zero':non_zero_ls})
    reg_df = pd.DataFrame(Regularization_plot_results, columns=Regularization_plot_ls)

    #print(df)
    #print(reg_df)

    #mse.to_csv('mse.csv')
    #df.to_csv('model_debug.csv')
    #reg_df.to_csv('Reg_df.csv')


    #plt.plot(df)
    #plt.plot(reg_df)

    weight_30, bias_30 = train(df_train_observations.to_numpy(), df_train_target.to_numpy(), 30)

    print("Finished")

if __name__ == "__main__":
    main()
