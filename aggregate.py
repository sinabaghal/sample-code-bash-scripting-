import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn import metrics
import more_itertools



def compute_auc(models):

    aggregate = {}
    for model in models:

        df = pd.DataFrame(columns=models)
        csv_path = f'./outputs/{model}/csv/{model}.csv'
        df = pd.read_csv(csv_path)
        aggregate[model] = df['y_pred'].values
       

    df_agg = pd.DataFrame(aggregate)
    df_agg['mean_col'] = df_agg.mean(axis=1)
    df_agg['actual'] = df['y_test'].values

    fpr, tpr, _ = metrics.roc_curve(df_agg.actual.values.tolist(), df_agg.mean_col.values.tolist())
    auc_score = metrics.auc(fpr, tpr)

    return auc_score

my_models = ['xgboost', 'catboost', 'rf', 'mlp', 'svc']

best_auc = 0
with open("aggregation_result.txt", "w") as text_file:
    
    for these_models in more_itertools.powerset(my_models):

        if len(these_models) == 0 : continue

        auc_score = compute_auc(these_models)
        
        if len(these_models) == 1: text_file.write("AUC SCORE: "+str(auc_score)); text_file.write("     ++++++ MODEL: "+str(these_models));text_file.write('\n')
        if auc_score > best_auc:

            best_auc = auc_score
            best_model = these_models


    text_file.write("AUC SCORE: "+str(best_model))
    text_file.write("     ++++++ MODELS: "+str(best_auc))





    


