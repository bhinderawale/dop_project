import torch
import torch.nn as nn
import torch.nn.functional as F
from Truncated_Cauchy_self_attention_block import Truncated_Cauchy_self_attention_block

class SATVN(nn.Module):
    """
    Class for the Self-Attentive Time-Variant model (SATVN model)
    """
    def __init__(self, input_dim, d_model, hidden_dim, h, M, output_dim, in_seq_length, out_seq_length, device):
        """
        :param input_dim: Dimension of the inputs
        :param d_model: Dimension of query matrix
        :param hidden_dim: Number of hidden units
        :param h: Number of heads of Truncated Cauchy self-attention
        :param M: Number of encoder layers of SATVN model
        :param output_dim: Dimension of the outputs
        :param in_seq_length: Length of the input sequence
        :param out_seq_length: Length of the output sequence
        """
        super(SATVN, self).__init__()

        self.input_dim = input_dim
        self.d_model=d_model
        self.hidden_dim = hidden_dim
        self.h = h
        self.M = M
        self.output_dim = output_dim
        self.in_seq_length = in_seq_length
        self.out_seq_length = out_seq_length
        self.device = device
        # Initialise layers
        hidden_layer1 = [nn.Linear(input_dim, hidden_dim)]
        for i in range(out_seq_length - 1):
            hidden_layer1.append(nn.Linear(input_dim + hidden_dim + output_dim, hidden_dim))
        self.hidden_layer1 = nn.ModuleList(hidden_layer1)
        self.hidden_layer2 = nn.ModuleList(
            [Truncated_Cauchy_self_attention_block(hidden_dim, d_model, hidden_dim, hidden_dim, d_model, d_model, h, M, attention_size=12, dropout=0, chunk_mode=None,
                         pe='regular', is_discriminator=False) for i in range(out_seq_length)])
        self.generator = nn.ModuleList([nn.Linear(hidden_dim, output_dim) for i in range(out_seq_length)])


    def forward(self, input, target, is_training=False):
        """
        Forward propagation of the SATVN model
        :param input: Input data in the form [n_samples, input_seq_length]
        :param target: Target data in the form [output_seq_length, n_samples, output_dim]
        :param is_training: If true, use target data for training, else use the previous output.
        :return: outputs: Forecast outputs in the form [out_seq_length, n_samples, input_dim]
        """
        # Initialise outputs
        outputs = torch.zeros((self.out_seq_length, input.shape[0], self.output_dim)).to(self.device)
        # First input
        next_cell_input = input
        #print("input device:", input.device)
        #print("target device:", target.device)
        #print("outputs device:", outputs.device)
        for i in range(self.out_seq_length):
            B, T, D = next_cell_input.shape
            #print("next_cell_input:", next_cell_input.shape)
            hidden = F.relu(self.hidden_layer1[i](next_cell_input)) #Format the dataset into the form [batch_size,L] from the form [batch_size,in_seq_length]   
            #print("after hidden_layer1:", hidden.shape)
            hidden = self.hidden_layer2[i](hidden)
            # Calculate the output （Linear layer of Truncated Cauchy self-attention block described in the paper）
            output = self.generator[i](hidden[:, -1, :])    
            outputs[i,:,:] = output
            # Prepare the next input
            if is_training:
                next_part = target[i]
            else:
                next_part = output
            if next_part.dim() == 2:
                next_part = next_part.unsqueeze(1).repeat(1, T, 1)
            next_cell_input = torch.cat((input, hidden, next_part), dim=2)    
        return outputs




