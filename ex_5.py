import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from gcommand_loader import *


# The model a convolutional neural network.
class Model(nn.Module):
    def __init__(self, epoch=15, lr=0.0001):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=20, kernel_size=5, stride=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(in_channels=20, out_channels=35, kernel_size=5, stride=1)
        self.dropout = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(28490, 100)
        self.bn = nn.BatchNorm1d(100)
        self.fc2 = nn.Linear(100, 30)
        self.epoch = epoch
        self.lr = lr

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.bn(self.fc1(x)))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# get the file names from the directory
def get_file_names(spects):
    names = []
    for spect in spects:
        split = spect[0].split('\\')
        name = split[-1]
        names.append(name)
    return names


# train the model with a validation set.
def train_with_validation(model, train_loader, validation_loader):
    optimizer = optim.Adam(model.parameters(), lr=model.lr)
    for e in range(model.epoch):
        model.train()
        print("epoch")
        print(e)
        current_loss = 0
        correct = 0
        total = 0
        for batch_idx, (data, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            for idx, i in enumerate(output):
                if torch.argmax(i) == labels[idx]:
                    correct += 1
                total += 1
            entropy_loss = nn.CrossEntropyLoss()
            loss = entropy_loss(output, labels)
            loss.backward()
            optimizer.step()
            current_loss += loss.item()
        train_curr_acc = round(correct / total, 3)
        train_curr_loss = round(current_loss / len(train_loader), 3)
        print("Current train accuracy {} ".format(train_curr_acc))
        print("Current train loss {} ".format(train_curr_loss))
        with torch.no_grad():
            valid_curr_acc, valid_curr_loss = test_validation_set(model, validation_loader)
        print("Current validation accuracy {} ".format(valid_curr_acc))
        print("Current validation loss {} ".format(valid_curr_loss))



# test the validation set for one epoch return the accuracy and loss.
def test_validation_set(model, validation_loader):
    with torch.no_grad():
        model.eval()
        test_loss = 0
        correct = 0
        total = 0
        for data, labels in validation_loader:
            output = model(data)
            entropy_loss = nn.CrossEntropyLoss()
            loss = entropy_loss((output), (labels))
            test_loss += loss.item()
            for idx, i in enumerate(output):
                if torch.argmax(i) == labels[idx]:
                    correct += 1
                total += 1
    accuracy = round(correct / total, 3)
    loss = round(test_loss / len(validation_loader), 3)
    return accuracy, loss


# get the model's predictions for a given test set.
def get_results_from_model(model, test_loader):
    predictions = []
    with torch.no_grad():
        model.eval()
        for data, _ in test_loader:
            output = model(data)
            for idx, i in enumerate(output):
                predictions.append(torch.argmax(i).item())
    predictions = np.asarray(predictions).astype(int)
    return predictions


# write the model's results to the file.
def write_results_to_file(file_name, names, results, dict):
    f = open(file_name, "w")
    last_line = results.shape[0] - 1
    for idx, res in enumerate(results):
        f.write(f"{names[idx]},{dict[res]}")
        if idx != last_line:
            f.write("\n")
    f.close()


def main():
    test_data = GCommandLoader('./test')
    train_data = GCommandLoader('./train')
    valid_data = GCommandLoader('./valid')
    class_to_idx = train_data.class_to_idx
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    file_names = get_file_names(test_data.spects)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True,
                                               num_workers=4, pin_memory=True, sampler=None)
    validation_loader = torch.utils.data.DataLoader(valid_data, batch_size=64, shuffle=False,
                                                    num_workers=4, pin_memory=True, sampler=None)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=False,
                                              num_workers=4, pin_memory=True, sampler=None)
    model = Model()
    train_with_validation(model, train_loader, validation_loader)
    results = get_results_from_model(model, test_loader)
    write_results_to_file("test_y", file_names, results, idx_to_class)


if __name__ == '__main__':
    main()
