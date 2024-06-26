import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import os
import re
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
import copy

np.random.seed(42)

data_directories = "data/raw/RNN"

all_csvs = []
for folder in os.listdir(data_directories):
    if folder in ['bending1', 'bending2']:
        continue
    folder_csvs = data_directories+'/'+folder
    for data_csv in os.listdir(folder_csvs):
        if data_csv == 'dataset8.csv' and folder == 'sitting':
            # this dataset only has 479 instances
            # it is possible to use it, but would require padding logic
            continue
        loaded_data = pd.read_csv(f"{data_directories}/{folder}/{data_csv}", skiprows=4)
        #print(f"{folder}/{data_csv} ------ {loaded_data.shape}")
        
        csv_id = re.findall(r'\d+', data_csv)[0]
        loaded_data['id'] = csv_id
        loaded_data['all_id'] = f"{folder}_{csv_id}"
        loaded_data['activity'] = folder
        all_csvs.append(loaded_data)

all_data = pd.concat(all_csvs)

raw_model_features = ['avg_rss12', 'var_rss12', 'avg_rss13', 'var_rss13', 'avg_rss23', 'var_rss23']
all_data.columns = ['timestamp', 'avg_rss12', 'var_rss12', 'avg_rss13', 'var_rss13', 'avg_rss23', 'var_rss23', 'id', 'all_id', 'activity']


# choose ids to use for test
ids_for_test = np.random.choice(all_data['id'].unique(), size=4, replace=False)

d_train =  all_data[~all_data['id'].isin(ids_for_test)]
d_test = all_data[all_data['id'].isin(ids_for_test)]

class NumericalNormalizer:
    def __init__(self, fields: list):
        self.metrics = {}
        self.fields = fields

    def fit(self, df: pd.DataFrame ) -> list:
        means = df[self.fields].mean()
        std = df[self.fields].std()
        for field in self.fields:
            field_mean = means[field]
            field_stddev = std[field]
            self.metrics[field] = {'mean': field_mean, 'std': field_stddev}

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        # Transform to zero-mean and unit variance.
        for field in self.fields:
            f_mean = self.metrics[field]['mean']
            f_stddev = self.metrics[field]['std']
            # OUTLIER CLIPPING to [avg-3*std, avg+3*avg]
            df[field] = df[field].apply(lambda x: f_mean - 3 * f_stddev if x < f_mean - 3 * f_stddev else x)
            df[field] = df[field].apply(lambda x: f_mean + 3 * f_stddev if x > f_mean + 3 * f_stddev else x)
            if f_stddev > 1e-5:
                df[f'p_{field}_normalized'] = df[field].apply(lambda x: ((x - f_mean)/f_stddev))
            else:
                df[f'p_{field}_normalized'] = df[field].apply(lambda x: x * 0)
        return df
    

#all features are numerical
normalizor = NumericalNormalizer(raw_model_features)
normalizor.fit(d_train)
d_train_normalized = normalizor.transform(d_train)
d_test_normalized = normalizor.transform(d_test)


model_features = [f"p_{x}_normalized" for x in raw_model_features]
time_feat = 'timestamp'
label_feat = 'activity'
sequence_id_feat = 'all_id'

plot_feats = {
    'p_avg_rss12_normalized': "Mean Chest <-> Right Ankle",
    'p_var_rss12_normalized': "STD Chest <-> Right Ankle",
    'p_avg_rss13_normalized': "Mean Chest <-> Left Ankle",
    'p_var_rss13_normalized': "STD Chest <-> Left Ankle",
    'p_avg_rss23_normalized': "Mean Right Ankle <-> Left Ankle",
    'p_var_rss23_normalized': "STD Right Ankle <-> Left Ankle",
    
}

# possible activities ['cycling', 'lying', 'sitting', 'standing', 'walking']
#Select the activity to predict
chosen_activity = 'cycling'

d_train_normalized['label'] = d_train_normalized['activity'].apply(lambda x: int(x == chosen_activity))
d_test_normalized['label'] = d_test_normalized['activity'].apply(lambda x: int(x == chosen_activity))


def df_to_Tensor(df, model_feats, label_feat, group_by_feat, timestamp_Feat):
    sequence_length = len(df[timestamp_Feat].unique())
    
    data_tensor = np.zeros((len(df[group_by_feat].unique()), sequence_length, len(model_feats)))
    labels_tensor = np.zeros((len(df[group_by_feat].unique()), 1))
    
    for i, name in enumerate(df[group_by_feat].unique()):
        name_data = df[df[group_by_feat] == name]
        sorted_data = name_data.sort_values(timestamp_Feat)
        
        data_x = sorted_data[model_feats].values
        labels = sorted_data[label_feat].values
        assert labels.sum() == 0 or labels.sum() == len(labels)
        data_tensor[i, :, :] = data_x
        labels_tensor[i, :] = labels[0]
    data_tensor = torch.from_numpy(data_tensor).type(torch.FloatTensor)
    labels_tensor = torch.from_numpy(labels_tensor).type(torch.FloatTensor)
    
    return data_tensor, labels_tensor

train_data, train_labels = df_to_Tensor(d_train_normalized, model_features, 'label', sequence_id_feat, time_feat)
test_data, test_labels = df_to_Tensor(d_test_normalized, model_features, 'label', sequence_id_feat, time_feat)

if not os.path.exists('data/processed/RNN'):
    os.makedirs('data/processed/RNN')

torch.save(train_data, 'data/processed/RNN/train_data.pth')
torch.save(train_labels, 'data/processed/RNN/train_labels.pth')
torch.save(test_data, 'data/processed/RNN/test_data.pth')
torch.save(test_labels, 'data/processed/RNN/test_labels.pth')

class ExplainedRNN(nn.Module):
    def __init__(self,
                 input_size: int,
                 cfg: dict,
                 ):
        super(ExplainedRNN, self).__init__()
        self.hidden_dim = cfg.get('hidden_dim', 32)
        torch.manual_seed(cfg.get('random_seed', 42))

        self.recurrent_block = nn.GRU(
            input_size=input_size,
            hidden_size=self.hidden_dim,
            batch_first=True,
            num_layers=2,
            )
        
        self.classifier_block = nn.Linear(self.hidden_dim, 1)
        self.output_activation_func = nn.Sigmoid()

    def forward(self,
                x: torch.Tensor,
                hidden_states: tuple = None,
                ):

        if hidden_states is None:
            output, hidden = self.recurrent_block(x)
        else:
            output, hidden = self.recurrent_block(x, hidden_states)

        # -1 on hidden, to select the last layer of the stacked gru
        assert torch.equal(output[:,-1,:], hidden[-1, :, :])
        
        y = self.classifier_block(hidden[-1, :, :])
        y = self.output_activation_func(y)
        return y, hidden
    

model = ExplainedRNN(len(model_features), {})
loss_function = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

learning_rate = 0.005
EPOCHS = 8

for epoch in tqdm.trange(EPOCHS):
    train_data_local = copy.deepcopy(train_data)
    train_labels_local = copy.deepcopy(train_labels)
    
    y_pred, hidden_states = model(train_data_local)
    train_loss = loss_function(y_pred, train_labels_local)

    optimizer.zero_grad()
    train_loss.backward()
    optimizer.step()
    with torch.no_grad():
        test_data_local = copy.deepcopy(test_data)
        test_labels_local = copy.deepcopy(test_labels)
        test_preds, _ = model(test_data_local)
        test_loss = loss_function(test_preds, test_labels_local)
        print(f"Train loss: {train_loss.item()} --- Test loss {test_loss.item()} ")

torch.save(model.state_dict(), 'models/RNN.pth')
