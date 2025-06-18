import torch
import torch.nn as nn


class My_dataset(Dataset):
    def __init__(self, df, text_column):
        self.vecs = df[text_column].values
        self.cat = df['category_encoded'].values

    def __getitem__(self, index):
        return self.vecs[index], self.cat[index]

    def __len__(self):
        return len(self.vecs)


class My_model(nn.Module):
    def __init__(self, size_in, size_hide, size_out):
        super().__init__()
        self.linear_1 = nn.Linear(size_in, size_hide)
        self.bn = nn.BatchNorm1d(size_hide)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.dp = nn.Dropout(0.8)
        self.linear_2 = nn.Linear(size_hide, size_out)

    def forward(self, x):
        x = x.to(torch.float32)
        x = self.linear_1(x)
        x = self.bn(x)
        x = self.tanh(x)
        x = self.dp(x)
        x = self.linear_2(x)
        return x

    def predict(self, input, drop_punctuation, normilise_text, vectorizer, device='cpu'):
        clear_text = drop_punctuation(input, symbols_to_avoid)
        tokens_text = normilise_text(clear_text, morph)
        text_vector = vectorizer.transform([tokens_text]).toarray()[0]

        output = self.forward(torch.tensor([text_vector])).detach().numpy()

        category_id = np.argmax(output)
        return category_encoded_map_reversed[category_id]


def init_my_model(device, size_in, categories_num):
    model = My_model(size_in=size_in,
                     size_hide=categories_num,
                     size_out=categories_num).to(device)

    model.train()
    return model

def init_my_categories_weights_map(df):
    count_categories = dict(df['category'].value_counts())
    count_categories_max = max(count_categories.values())

    categories_weights_map = dict()
    for category in count_categories.keys():
        categories_weights_map[category] = count_categories_max / count_categories[category]

        return categories_weights_map


def init_my_optimizer(model):
    return torch.optim.Adam(model.parameters(), lr=0.01)


def init_my_criterion(df, device):
    categories_weights_map = init_my_categories_weights_map(df)
    return torch.nn.CrossEntropyLoss(weight=torch.tensor(list(categories_weights_map.values())).float().to(device))
