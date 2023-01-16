from sklearn.svm import SVC 
import argparse, itertools
from sklearn import metrics
from multiprocessing import Pool
import data_utils 
import matplotlib.pyplot as plt
import pandas as pd 


parser = argparse.ArgumentParser()

parser.add_argument("--start_gamma", default = -2, type=int)
parser.add_argument("--end_gamma", default = 4, type=int)
parser.add_argument("--step_gamma", default = 1, type=int)

parser.add_argument("--start_c", default = 2, type=int)
parser.add_argument("--end_c", default = 4, type=int)
parser.add_argument("--step_c", default = 1, type=int)

parser.add_argument("--model_name", type=str)
parser.add_argument("--output_dir",type=str)

parser.add_argument("--data_dir",type=str)
parser.add_argument("--rbf",type=str)
parser.add_argument("--fig_title", type=str)
parser.add_argument("--png_path", type=str)
parser.add_argument("--csv_path", type=str)
parser.add_argument("--txt_path", type=str)

args = parser.parse_args()

x_train, x_test, y_train, y_test = data_utils.prepared_data(args)


def cmpt_proba(params):

    gamma, c = params['gamma'], params['c']
    hashable_params = (gamma,c)

    model = SVC(gamma=gamma, C=c, class_weight='balanced', probability=True)
    model.fit(x_train, y_train) 
    y_pred_proba = model.predict_proba(x_test)
    
    return hashable_params, [pred[1] for pred in y_pred_proba]

if __name__ == "__main__":
    
    assert args.model_name == 'svc'

    result = {}

    GAMMAS = [10**e for e in range(args.start_gamma,args.end_gamma,args.step_gamma)]
    CS = [10**e for e in range(args.start_c,args.end_c,args.step_c)]

    params_grid = [{"gamma": item[0], "c": item[1]} 
                                            for item in itertools.product(GAMMAS, CS)]
    print(f'Number of models to train : {len(params_grid)}')

    if len(params_grid) == 1: save = True  
    else: save = False

    with Pool() as pool:

        results = pool.map(cmpt_proba, params_grid)

    
    for hashable_params, y_pred in results:

        fpr, tpr, _ = metrics.roc_curve(y_test, y_pred)
        auc_score = metrics.auc(fpr, tpr)

        

        result[hashable_params] = auc_score
        
        if save: 

            txt = f'Saving for gamma = {hashable_params[0]}, c = {hashable_params[1]}'
            print(txt)
            with open(args.txt_path, 'w') as file:
                file.write(txt)
            
            df_pred = pd.DataFrame(columns = ['y_test', 'y_pred'])
            df_pred['y_test'] = y_test
            df_pred['y_pred'] = y_pred
            df_pred.to_csv(args.csv_path)
    

    plt.rcParams["figure.figsize"] = (30,20)
    plt.rcParams.update({'font.size': 16})
    for gamma in GAMMAS:

        this_result = []

        for params in params_grid:
            hashable_params = (params["gamma"], params["c"])

            if params["gamma"] == gamma:

                this_result.append(result[hashable_params])
            
        plt.plot(CS, this_result, 'o-', label=f"Gamma = {gamma}")
    
    

    
    plt.xlabel('CS')
    plt.ylabel('AUC SCORE')
    plt.legend(loc=4, prop={'size': 6})
    plt.title(args.fig_title)
    if not save: plt.savefig(args.png_path, dpi = 600)
    plt.show()
    



