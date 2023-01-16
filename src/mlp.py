import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt 
from sklearn import metrics 
import argparse
import data_utils 
import pandas as pd 


class TrainData(Dataset):
    
    def __init__(self, x_data, y_data):
        self.x_data = x_data
        self.y_data = y_data
        
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]
        
    def __len__ (self):
        return len(self.x_data)


class TestData(Dataset):
    
    def __init__(self, x_data, y_data):
        self.x_data = x_data
        self.y_data = y_data
        
    def __getitem__(self, index):
        return self.x_data[index]
        
    def __len__ (self):
        return len(self.x_data)


class BinaryClassification(nn.Module):

    def __init__(self, in_feature, n_hidden):
        super(BinaryClassification, self).__init__()
        
        self.layer_1 = nn.Linear(in_feature, n_hidden) 
        
        self.layer_out = nn.Linear(n_hidden, 1) 
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.7)
        
        
    def forward(self, inputs):

        x = self.relu(self.layer_1(inputs))
        x = self.dropout(x)
        x = self.layer_out(x)
        
        return x


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--start_nhidden", default = 2, type=int)
    parser.add_argument("--end_nhidden", default = 10, type=int)
    parser.add_argument("--step_nhidden", default = 1, type=int)

    parser.add_argument("--learning_rate", default = 0.03, type=float)
    parser.add_argument("--batch_size", default = 64, type=int)
    parser.add_argument("--epochs", default = 100, type=int)
    parser.add_argument("--rbf",type=int)
    
    
    parser.add_argument("--output_dir",type=str)
    parser.add_argument("--data_dir",type=str)
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--fig_title", type=str)
    parser.add_argument("--png_path", type=str)
    parser.add_argument("--csv_path", type=str)
    

    args = parser.parse_args()
    #print(args)
    

    x_train, x_test, y_train, y_test = data_utils.prepared_data(args)
    # import pdb; pdb.set_trace()

    if args.rbf == 0:
        x_train_nn, x_test_nn, y_train_nn, y_test_nn = x_train.values, x_test.values, y_train.values, y_test
    else:
        x_train_nn, x_test_nn, y_train_nn, y_test_nn = x_train, x_test, y_train, y_test

    in_features = x_train.shape[1]

    
    train_data = TrainData(torch.FloatTensor(x_train_nn), torch.FloatTensor(y_train_nn))
    test_data = TestData(torch.FloatTensor(x_test_nn), torch.FloatTensor(y_test_nn))

    train_loader = DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_data, batch_size=1)
    

    def eval(model):

        model.eval()
        y_pred = []
        with torch.no_grad():

            for x_batch in test_loader:
                x_batch = x_batch.to(device)
                y_test_pred = model(x_batch)

                y_test_pred = torch.sigmoid(y_test_pred)
                y_pred.append(y_test_pred.tolist()[0][0])
                
        return y_pred


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    plt.rcParams["figure.figsize"] = (30,20)
    plt.rcParams.update({'font.size': 16})

    def run_nn(n_hidden):

        model = BinaryClassification(in_features, n_hidden)

        model.to(device)
        criterion = torch.nn.BCEWithLogitsLoss ()
        optimizer = optim.SGD(model.parameters(), lr=args.learning_rate)
        model.train()

        auc_scores = []
        for _ in range(1, args.epochs+1):
            
            y_train_ = []
            y_pred_ = []
            for x_batch, y_batch in train_loader:

                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                
                y_pred = model(x_batch)

                y_pred_.append(y_pred)
                y_train_.append(y_pred) 
                loss = criterion(y_pred, y_batch.unsqueeze(1))

                loss.backward()
                optimizer.step()

            y_pred = eval(model)
            fpr, tpr, _ = metrics.roc_curve(y_test, y_pred)
            auc_score = metrics.auc(fpr, tpr)
            auc_scores.append(auc_score) 

        print(f'Done with {n_hidden}')
        plt.plot(range(1, args.epochs+1), auc_scores, 'o-', label = f'# HIDDEN NODES = {n_hidden}' )
        return y_pred    

        

    for n_hidden in range(args.start_nhidden,args.end_nhidden,args.step_nhidden):

        y_pred = run_nn(n_hidden)

        if args.end_nhidden-1 == n_hidden: 

            print(f'Saving for n_hidden = {n_hidden}')
            df_pred = pd.DataFrame(columns = ['y_test', 'y_pred'])
            df_pred['y_test'] = y_test
            df_pred['y_pred'] = y_pred
            df_pred.to_csv(args.csv_path)
    

    

    plt.xlabel('EPOCHS')
    plt.ylabel('AUC SCORE')
    plt.legend(loc=4, prop={'size': 6})
    plt.title(args.fig_title)
    plt.savefig(args.png_path, dpi = 600)
    plt.show()





