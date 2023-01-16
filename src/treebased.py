
from sklearn import metrics
import itertools, os, argparse
from multiprocessing import Pool
import matplotlib.pyplot as plt
from sklearn.svm import SVC 
from sklearn.ensemble import RandomForestClassifier 
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
import data_utils
import pandas as pd 



parser = argparse.ArgumentParser()

parser.add_argument("--start_depth",default = 4, type=int)
parser.add_argument("--end_depth",default = 10, type=int)
parser.add_argument("--step_depth",default = 1, type=int)

parser.add_argument("--start_n_es", default = 10, type=int)
parser.add_argument("--end_n_es", default = 180, type=int)
parser.add_argument("--step_n_es", default = 5, type=int)


parser.add_argument("--learning_rate",type=float)

parser.add_argument("--model_name", type=str)
parser.add_argument("--output_dir",type=str)

parser.add_argument("--data_dir",type=str)
parser.add_argument("--rbf",type=int)
parser.add_argument("--fig_title", type=str)
parser.add_argument("--png_path", type=str)
parser.add_argument("--csv_path", type=str)
parser.add_argument("--txt_path", type=str)


args = parser.parse_args()

x_train, x_test, y_train, y_test = data_utils.prepared_data(args)

def get_model(model_name, params):

    n_estimators, max_depth, eta = params["n_estimators"], params["max_depth"], params["eta"]
   
    if model_name == 'catboost': 
        
        return CatBoostClassifier(loss_function="Logloss", num_trees=n_estimators,depth=max_depth, eta=eta, use_best_model=True,
                                verbose=False,eval_metric="AUC") 

    elif model_name == 'xgboost': 
        
        return XGBClassifier(eval_metric='mlogloss', n_estimators = n_estimators, max_depth = max_depth, eta=eta)
        
    elif model_name == 'rf': 

        return RandomForestClassifier(n_estimators = n_estimators, max_depth=max_depth) 
    
    else: 

        NotImplemented('This model is not implemented yet!')

def cmpt_proba(params):

    hashable_params = (params["n_estimators"], params["max_depth"])
    model = get_model(args.model_name, params)
    if args.model_name == 'catboost':

        model.fit(x_train , y_train, eval_set = (x_test, y_test))
    else:
        model.fit(x_train , y_train)
    y_pred_proba= model.predict_proba(x_test)

    return hashable_params, [pred[1] for pred in y_pred_proba]


if __name__ == "__main__":

    
    N_ESTIMATORS = range(args.start_n_es,args.end_n_es,args.step_n_es)
    MAX_DEPTHS = list(range(args.start_depth,args.end_depth, args.step_depth))
    ETA = [args.learning_rate]

    if args.model_name == 'rf':  MAX_DEPTHS.append(None)


    result = {}
    params_grid = [{"n_estimators": item[0], "max_depth": item[1], "eta":item[2]} 
                                                                 for item in itertools.product(N_ESTIMATORS, MAX_DEPTHS, ETA)]

    print(f'Number of models to train : {len(params_grid)}')

    if len(params_grid) == 1: save = True  
    else: save = False 

    with Pool() as pool:

        results = pool.map(cmpt_proba, params_grid)

    # import pdb; pdb.set_trace()
    for hashable_params, y_pred in results:

       
        fpr, tpr, _ = metrics.roc_curve(y_test, y_pred)
        auc_score = metrics.auc(fpr, tpr)

        result[hashable_params] = auc_score

        if save: 

            txt = f'Saving for n_estimators = {hashable_params[0]}, max_depth = {hashable_params[1]}'
            print(txt)
            with open(args.txt_path, 'w') as file:
                file.write(txt)
           
            df_pred = pd.DataFrame(columns = ['y_test', 'y_pred'])
            df_pred['y_test'] = y_test
            df_pred['y_pred'] = y_pred
            df_pred.to_csv(args.csv_path)
    

    plt.rcParams["figure.figsize"] = (30,20)
    plt.rcParams.update({'font.size': 16})

    for max_depth in MAX_DEPTHS:

        this_result = []

        for params in params_grid:
            hashable_params = (params["n_estimators"], params["max_depth"])

            if params["max_depth"] == max_depth:

                this_result.append(result[hashable_params])
            
        plt.plot(N_ESTIMATORS, this_result, 'o-', label=f"DEPTH = {max_depth}")
    

    plt.xlabel('N_ESTIMATORS')
    plt.ylabel('AUC SCORE')
    plt.legend(loc=2, prop={'size': 6})
    plt.title(args.fig_title)
    if not save: plt.savefig(args.png_path, dpi = 900)
    plt.show()

