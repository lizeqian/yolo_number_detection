import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.autograd as autograd

class LSTMLayer(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, out_dim, lstm_layer, batch):
        super(LSTMLayer, self).__init__()
        self.hidden_dim = hidden_dim
        self.batch = batch
        self.num_layers = lstm_layer
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, lstm_layer, dropout=0.5, batch_first = True)
        self.hidden = self.init_hidden()
        self.final_linear = nn.Linear(hidden_dim, out_dim)
        self.sm = nn.Softmax()

    def init_hidden(self):
        return (autograd.Variable(torch.zeros(self.num_layers, self.batch, self.hidden_dim)),
                autograd.Variable(torch.zeros(self.num_layers, self.batch, self.hidden_dim)))

    def forward(self, inputs):
#        inputs = inputs.contiguous().view(inputs.size()[0], self.batch, inputs.size()[1])
        lstm_out, self.hidden = self.lstm(inputs, self.hidden)
        outs = lstm_out[:,-1, :]
        outs = self.final_linear(outs)
        outs = torch.sigmoid(outs)
        #outs = self.sm(outs)
        return outs
