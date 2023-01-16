import pandas as pd 
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import argparse
from sklearn.manifold import TSNE
import numpy as np 



COLS_TO_DROP = [ "CustomerId", 'BranchId']
CATEGORICAL_COLS = ['City', 'Gender', 'PrefContact', 'CurrencyCode', 'PrefLanguage']

COLS_TO_CONVERT = CATEGORICAL_COLS


class SplitData:

    def __init__(self, x_train=None, y_train=None, x_test=None, y_test=None):

        self.x_train, self.y_train, self.x_test, self.y_test = x_train, y_train, x_test, y_test
        self.y_train_proba, self.y_test_proba = None, None 
    
    
        
class DataLoader:

    
    def __init__(self, args):

        self.cols_to_drop = COLS_TO_DROP
        self.df_train = pd.read_csv(args.data_dir+"train.csv")
        self.df_test = pd.read_csv(args.data_dir+"test.csv")

    def split(self):

        y_train = self.df_train["Exited"]
        x_train = self.df_train.drop(columns= ['Exited'] + self.cols_to_drop) 

        y_true = self.df_test["Exited"]
        x_test = self.df_test.drop(columns=["Exited"] + self.cols_to_drop) 

        self.split_data = SplitData(x_train, y_train, x_test, y_true)

    def ohe(self):

        x_train, y_train, x_test, y_test = \
            self.split_data.x_train, self.split_data.y_train, self.split_data.x_test, self.split_data.y_test

        for col in ['City', 'Gender', 'PrefContact', 'CurrencyCode', 'PrefLanguage']:
            for df in x_train, x_test:
                
                ohe = OneHotEncoder(sparse=False, dtype=int, drop='first')
                df_t = ohe.fit_transform(df[[col]])
                df[ohe.categories_[0][1:]] = df_t

                
                df.drop(columns = [col], inplace=True)


        self.split_data = SplitData(x_train, y_train, x_test, y_test)

    

    def scale(self):

        scaler = StandardScaler()

        cols = ['CreditScore', 'Balance', 'EstimatedSalary']

        scaled_features = scaler.fit_transform(self.split_data.x_train[cols].values)
        scaled_features_df = pd.DataFrame(scaled_features, index=self.split_data.x_train.index, columns=cols)
        self.split_data.x_train[cols] = scaled_features_df[cols]


        scaled_features = scaler.transform(self.split_data.x_test[cols].values)
        scaled_features_df = pd.DataFrame(scaled_features, index=self.split_data.x_test.index, columns=cols)
        self.split_data.x_test[cols] = scaled_features_df[cols]
    

def cmpt_rbf(args):

    
    

    data = DataLoader(args)
    data.split()
    data.ohe()
    data.scale()
    split_data = data.split_data

    df_tot = pd.concat([split_data.x_train,split_data.x_test ])
    xemb = TSNE(n_components=2).fit_transform(df_tot)

    xemb_train = xemb[:split_data.x_train.shape[0],]
    xemb_test = xemb[split_data.x_train.shape[0]:,]

    np.save(args.data_dir+"xemb_train.npy", xemb_train)
    np.save(args.data_dir+"xemb_test.npy", xemb_test)
   
def prepared_data(args):

    data = DataLoader(args)
    data.split() ; data.ohe() ; data.scale()
    split_data = data.split_data 

    if args.rbf == 1: 

        print('Loading RBF dataset!')

        rbf_train, rbf_test = args.data_dir+"xemb_train.npy", args.data_dir+"xemb_test.npy"
        x_train, x_test = np.load(rbf_train), np.load(rbf_test)

    else: 
        x_train, x_test = split_data.x_train, split_data.x_test 

    y_train, y_test = split_data.y_train, split_data.y_test.to_numpy()
    return x_train, x_test, y_train, y_test 

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",type=str)
    args = parser.parse_args()
    cmpt_rbf(args)

    