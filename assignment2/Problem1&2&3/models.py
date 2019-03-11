"""
script for problem 1 : building an rnn from scratch using Pytorch
"""

import torch 
import torch.nn as nn

import numpy as np
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable
import matplotlib.pyplot as plt


# NOTE ==============================================
#
# Fill in code for every method which has a TODO
#
# Your implementation should use the contract (inputs
# and outputs) given for each model, because that is 
# what the main script expects. If you modify the contract, 
# you must justify that choice, note it in your report, and notify the TAs 
# so that we run the correct code.
#
# You may modify the internals of the RNN and GRU classes
# as much as you like, except you must keep the methods
# in each (init_weights_uniform, init_hidden, and forward)
# Using nn.Module and "forward" tells torch which 
# parameters are involved in the forward pass, so that it
# can correctly (automatically) set up the backward pass.
#
# You should not modify the interals of the Transformer
# except where indicated to implement the multi-head
# attention. 


#A helper function for producing N identical layers (each with their own parameters).
def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

# Problem 1
class RNN(nn.Module): # Implement a stacked vanilla RNN with Tanh nonlinearities.
    def __init__(self, emb_size, hidden_size, seq_len, batch_size, vocab_size, num_layers, dp_keep_prob):
        """
        emb_size:     The number of units in the input embeddings
        hidden_size:  The number of hidden units per layer
        seq_len:      The length of the input sequences
        vocab_size:   The number of tokens in the vocabulary (10,000 for Penn TreeBank)
        num_layers:   The depth of the stack (i.e. the number of hidden layers at 
                      each time-step)
        dp_keep_prob: The probability of *not* dropping out units in the 
                      non-recurrent connections.
                      Do not apply dropout on recurrent connections.
        """
        super(RNN, self).__init__()

        # TODO ========================
        # Initialization of the parameters of the recurrent and fc layers. 
        # Your implementation should support any number of stacked hidden layers 
        # (specified by num_layers), use an input embedding layer, and include fully
        # connected layers with dropout after each recurrent layer.
        # Note: you may use pytorch's nn.Linear, nn.Dropout, and nn.Embedding 
        # modules, but not recurrent modules.
        #
        # To create a variable number of parameter tensors and/or nn.Modules 
        # (for the stacked hidden layer), you may need to use nn.ModuleList or the 
        # provided clones function (as opposed to a regular python list), in order 
        # for Pytorch to recognize these parameters as belonging to this nn.Module 
        # and compute their gradients automatically. You're not obligated to use the
        # provided clones function.

        self.emb_size = emb_size 
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.dp_keep_prob = dp_keep_prob

        ###Create modules###

        #embeddings layer
        self.embeddings = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.emb_size)

        #Apply dropout to embeddings and to the non recurrent hidden connexions
        self.dropout = nn.Dropout(dp_keep_prob)
        
        #For only the first layer
        self.f_layer = nn.Sequential(
                            #(hidden_size * emb_size + hidden_size)
                            nn.Linear(self.emb_size + self.hidden_size, self.hidden_size, bias=True),
                            nn.Tanh()
                            )

        """
        #Create module
        module = nn.Sequential(
                            #FC + dropout
                            nn.Linear(self.hidden_size, self.hidden_size, bias=True),
                            nn.Dropout(dp_keep_prob),
                            #(hidden_size * hidden_size + hidden_size) for hidden layers
                            nn.Linear(self.hidden_size + self.hidden_size, self.hidden_size, bias=True), 
                            #nn.Tanh(),
                            nn.Tanh(),
                            )
        """

        #FC + dropout
        #self.fc = nn.Sequential(
                        #nn.Linear(self.emb_size, self.hidden_size, bias=True),
        #                nn.Dropout(dp_keep_prob)
        #    )

        #For the rest of the layers
        module = nn.Sequential(
                            #Sum the output of the last hidden output (in same timestep) with the previous hidden state
                            nn.Linear(self.hidden_size + self.hidden_size, self.hidden_size, bias=True), 
                            nn.Tanh(),
                            )
        
        #Create self.num_layers - 1 (excluding the first layer) hidden layers for 1 time step
        self.rec = clones(module, self.num_layers-1)

        #Output layer (logits)
        self.logit = nn.Linear(self.hidden_size, self.vocab_size, bias=True)

        #Uniform init of weights and biases of model
        self.init_weights()



    def init_weights(self):
        # TODO ========================
        # Initialize the embedding and output weights uniformly in the range [-0.1, 0.1]
        # and output biases to 0 (in place). The embeddings should not use a bias vector.
        # Initialize all other (i.e. recurrent and linear) weights AND biases uniformly 
        # in the range [-k, k] where k is the square root of 1/hidden_size

        #k is the square root of 1/hidden_size
        k = 1 / self.hidden_size

        #Weights for embeddings
        #We = (vacab_size, emb_size)
        We = torch.Tensor(self.vocab_size, self.emb_size).uniform_(-0.1, 0.1)

        #Wx = (hidden_size, emb_size) for the first layer
        Wx = torch.Tensor(self.hidden_size, self.emb_size).uniform_(-k, k)
        
        #Whh = (hidden_size, hidden_size) for the other layers
        Whh = torch.Tensor(self.hidden_size, self.hidden_size).uniform_(-k, k)

        #Wh = (hidden_size, hidden_size)
        Wh = torch.Tensor(self.hidden_size, self.hidden_size).uniform_(-k, k)

        #Wy = (vocab_size, hidden_size) 
        Wy = torch.Tensor(self.vocab_size,self.hidden_size).uniform_(-0.1, 0.1)

        #Combine the two weights Wx and Wh in combined = [Wh, Wx] in the case of the first layer
        i_combined = torch.cat((Wh, Wx), 1)

        #Combine the two weights Whh and Wh in combined = [Wh, Whh] in the case of the other layers
        h_combined = torch.cat((Wh, Whh), 1)

        #Bias
        by = torch.zeros(self.vocab_size)
        bh = torch.zeros(self.hidden_size)

        #Fill the linear layers with init weights and biases
        #To avoid problems of grads
        with torch.no_grad():

            #Embeddings (doesn't have bias)
            self.embeddings.weight.copy_(We)

            #First layer
            self.f_layer[0].weight.copy_(i_combined)
            self.f_layer[0].bias.copy_(bh)

            #The rest of the layers in timestep
            #If we have many layers 
            if(self.num_layers > 1):

                for layer in self.rec:
                    layer[0].weight.copy_(h_combined)
                    layer[0].bias.copy_(bh)

            #Last output layer that outputs logits
            self.logit.weight.copy_(Wy)
            self.logit.bias.copy_(by)
            
    
    def init_hidden(self):
        # TODO ========================
        # initialize the hidden states to zero
        """
        This is used for the first mini-batch in an epoch, only.
        """
        h0 = torch.zeros(self.num_layers, self.batch_size, self.hidden_size)

        return h0 # a parameter tensor of shape (self.num_layers, self.batch_size, self.hidden_size)

    def forward(self, inputs, hidden):
        # TODO ========================
        # Compute the forward pass, using a nested python for loops.
        # The outer for loop should iterate over timesteps, and the 
        # inner for loop should iterate over hidden layers of the stack. 
        # 
        # Within these for loops, use the parameter tensors and/or nn.modules you 
        # created in __init__ to compute the recurrent updates according to the 
        # equations provided in the .tex of the assignment.
        #
        # Note that those equations are for a single hidden-layer RNN, not a stacked
        # RNN. For a stacked RNN, the hidden states of the l-th layer are used as 
        # inputs to to the {l+1}-st layer (taking the place of the input sequence).

        """
        Arguments:
            - inputs: A mini-batch of input sequences, composed of integers that 
                        represent the index of the current token(s) in the vocabulary.
                            shape: (seq_len, batch_size)
            - hidden: The initial hidden states for every layer of the stacked RNN.
                            shape: (num_layers, batch_size, hidden_size)
        
        Returns:
            - Logits for the softmax over output tokens at every time-step.
                  **Do NOT apply softmax to the outputs!**
                  Pytorch's CrossEntropyLoss function (applied in ptb-lm.py) does 
                  this computation implicitly.
                        shape: (seq_len, batch_size, vocab_size)
            - The final hidden states for every layer of the stacked RNN.
                  These will be used as the initial hidden states for all the 
                  mini-batches in an epoch, except for the first, where the return 
                  value of self.init_hidden will be used.
                  See the repackage_hiddens function in ptb-lm.py for more details, 
                  if you are curious.
                        shape: (num_layers, batch_size, hidden_size)
        """

        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        logits = torch.zeros([self.seq_len, self.batch_size, self.vocab_size], device=device)

        #Loop over the timesteps
        for step in range(self.seq_len):

            ##Go through the first layer first

            #take embeddings of current timestep
            #(batch_size)
            inp = inputs[step]

            #Take initial hidden state for first layer
            h = hidden[0]

            # Lookup word embeddings
            #(batch_size, enb_size)
            emb = self.embeddings(inp)

            #Apply dropout to embeddings
            # OR NOT ?
            emb = self.dropout(emb)

            #Combine embeddings and last hidden state
            combined = torch.cat((h, emb), 1)

            out = self.f_layer(combined)

            #save final hidden state for next timestep
            hidden[0] = out

            #Loop over the rest of the layers
            for layer in range(self.num_layers-1):

                #Apply dropout to non recurrent connexion
                out = self.dropout(out)
                
                #Take the initial hidden state of the current layer
                h = hidden[layer+1]

                #Combine hidden state of l-th layer and current hidden state
                combined = torch.cat((h, out), 1)

                out = self.rec[layer](combined)

                #save final hidden state for next layer
                hidden[layer+1] = out

            #last layer to calculate the logits
            #(batch_size, vocab_size)
            logits[step] = self.logit(out)
                      
        return logits.view(self.seq_len, self.batch_size, self.vocab_size), hidden


    def generate(self, input, hidden, generated_seq_len):
        # TODO ========================
        # Compute the forward pass, as in the self.forward method (above).
        # You'll probably want to copy substantial portions of that code here.
        # 
        # We "seed" the generation by providing the first inputs.
        # Subsequent inputs are generated by sampling from the output distribution, 
        # as described in the tex (Problem 5.3)
        # Unlike for self.forward, you WILL need to apply the softmax activation 
        # function here in order to compute the parameters of the categorical 
        # distributions to be sampled from at each time-step.

        """
        Arguments:
            - input: A mini-batch of input tokens (NOT sequences!)
                            shape: (batch_size)
            - hidden: The initial hidden states for every layer of the stacked RNN.
                            shape: (num_layers, batch_size, hidden_size)
            - generated_seq_len: The length of the sequence to generate.
                           Note that this can be different than the length used 
                           for training (self.seq_len)
        Returns:
            - Sampled sequences of tokens
                        shape: (generated_seq_len, batch_size)
        """
       
        return samples


# Problem 2
class GRU(nn.Module): # Implement a stacked GRU RNN
  """
  Follow the same instructions as for RNN (above), but use the equations for 
  GRU, not Vanilla RNN.
  """
  def __init__(self, emb_size, hidden_size, seq_len, batch_size, vocab_size, num_layers, dp_keep_prob):
    super(GRU, self).__init__()

    # TODO ========================

  def init_weights_uniform(self):
    # TODO ========================
    pass

  def init_hidden(self):
    # TODO ========================
    return # a parameter tensor of shape (self.num_layers, self.batch_size, self.hidden_size)

  def forward(self, inputs, hidden):
    # TODO ========================
    return logits.view(self.seq_len, self.batch_size, self.vocab_size), hidden

  def generate(self, input, hidden, generated_seq_len):
    # TODO ========================
    return samples


# Problem 3
##############################################################################
#
# Code for the Transformer model
#
##############################################################################

"""
Implement the MultiHeadedAttention module of the transformer architecture.
All other necessary modules have already been implemented for you.

We're building a transfomer architecture for next-step prediction tasks, and 
applying it to sequential language modelling. We use a binary "mask" to specify 
which time-steps the model can use for the current prediction.
This ensures that the model only attends to previous time-steps.

The model first encodes inputs using the concatenation of a learned WordEmbedding 
and a (in our case, hard-coded) PositionalEncoding.
The word embedding maps a word's one-hot encoding into a dense real vector.
The positional encoding 'tags' each element of an input sequence with a code that 
identifies it's position (i.e. time-step).

These encodings of the inputs are then transformed repeatedly using multiple
copies of a TransformerBlock.
This block consists of an application of MultiHeadedAttention, followed by a 
standard MLP; the MLP applies *the same* mapping at every position.
Both the attention and the MLP are applied with Resnet-style skip connections, 
and layer normalization.

The complete model consists of the embeddings, the stacked transformer blocks, 
and a linear layer followed by a softmax.
"""

#This code has been modified from an open-source project, by David Krueger.
#The original license is included below:
#MIT License
#
#Copyright (c) 2018 Alexander Rush
#
#Permission is hereby granted, free of charge, to any person obtaining a copy
#of this software and associated documentation files (the "Software"), to deal
#in the Software without restriction, including without limitation the rights
#to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#copies of the Software, and to permit persons to whom the Software is
#furnished to do so, subject to the following conditions:
#
#The above copyright notice and this permission notice shall be included in all
#copies or substantial portions of the Software.
#
#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#SOFTWARE.



#----------------------------------------------------------------------------------

#For one single attention head
class AttentionHead(nn.Module):
    """A single attention head"""
    def __init__(self, inp, d_k, dropout):
        
        super().__init__()

        self.dropout = nn.Dropout(dropout)

        #Assuming that the queries, keys, and values have the same nb of units
        #affine transformations for queries, keys, and values to get the matrices 

        #d_k = n_units / n_heads
        #inp = n_units/size_hidden from previous attention block or embeddings

        self.query = nn.Linear(inp, d_k, bias=False)
            
        self.key = nn.Linear(inp, d_k, bias=False)

        self.value = nn.Linear(inp, d_k, bias=False)

 
    def forward(self, queries, keys, values, mask=None):
        ##For a single attention
        #d_k is dim of keys
        Q = self.query(queries) # (Batch, Seq, d_k)
        K = self.key(keys) # (Batch, Seq, d_k)
        V = self.value(values) # (Batch, Seq, d_k)

        #Compute attention score for the head
        z = self.ScaledDotProductAttention(Q, K, V, mask)
        
        return z #(batch_size, seq_len, d_k) and #(batch_size, seq_len, self.units) if we are doing multiheads


    def ScaledDotProductAttention(self, Q, K, V, mask):
        # get dim of key
        d_k = K.size(-1)

        #Should equal dim of queries also 
        assert Q.size(-1) == d_k
        
        #we get an attention score between each position in the sequence with current word
        #batch matrix-matrix product, (b,n,m)*(b,m,p)=(b,n,p)
        #(Batch, Seq, d_k) * (Batch, d_k, Seq) = (batch, Seq, Seq)
        attn = torch.bmm(Q, torch.transpose(K, 1, 2)) #(batch, Seq, Seq)
 
        #scale the dot products by d_k for numerical stability (more stable gradients)
        attn = attn / math.sqrt(d_k)

        #Apply softmax
        attn = torch.exp(attn)

        #fill attention weights with 0s where padded
        #Which is the opposite of what we want to do
        #if mask is not None: 
        #    attn = attn.masked_fill(mask, 0)

        #Cast to float tensor from byte tensor to perform multiplication
        mask = mask.type(torch.FloatTensor).cuda()

        #Apply mask to attention values
        attn = attn * mask

        #Normalize where row values add up to 1
        attn = attn / attn.sum(-1, keepdim=True)

        #print (attn)

        #For numerical stability issues
        attn = attn - (10**9) * (1 - mask) 
        #print (attn)

        #Apply dropout to attention output
        attn = self.dropout(attn)
        
        #Multiply value matrix V with attention scores: 
        #Keep value of words (having high score) we want to focus on and get rid of irrelevant ones (having low score)
        #And Sum up through matrix multiplication
        #Each row corresponds to a single query
        output = torch.bmm(attn, V) 
        
        return output #(Batch, Seq, n_k)



# TODO: implement this class
class MultiHeadedAttention(nn.Module):
    def __init__(self, n_heads, n_units, dropout=0.1):
        """
        n_heads: the number of attention heads
        n_units: the number of output units
        dropout: probability of DROPPING units
        """
        super(MultiHeadedAttention, self).__init__()

        
        # This sets the size of the keys, values, and queries (self.d_k) to all 
        # be equal to the number of output units divided by the number of heads.
        #d_k = dim of key
        self.d_k = n_units // n_heads

        #This requires the number of n_heads to evenly divide n_units.
        #NOTE: nb of n_units (hidden_size) must be a multiple of 16 (n_heads) 
        assert n_units % n_heads == 0
        #n_units represent total of units for all the heads
        
        #n_units = d_k * heads
        self.n_units = n_units 
        self.n_heads = n_heads

        # TODO: create/initialize any necessary parameters or layers
        # Note: the only Pytorch modules you are allowed to use are nn.Linear 
        # and nn.Dropout

        #Create the attention heads
        self.attn_heads = nn.ModuleList([
            AttentionHead(self.n_units, self.d_k, dropout) for _ in range(self.n_heads)
        ])
        #input dim = n_units/size_hidden from previous attention block and outpul dim = n_units
        self.projection = nn.Linear(self.n_units, self.n_units) 

        
    def forward(self, query, key, value, mask=None):
        # TODO: implement the masked multi-head attention.
        # query, key, and value all have size: (batch_size, seq_len, self.n_units, self.d_k)
        # mask has size: (batch_size, seq_len, seq_len)
        # As described in the .tex, apply input masking to the softmax 
        # generating the "attention values" (i.e. A_i in the .tex)
        # Also apply dropout to the attention values.


        #Loop over heads
        z = [attn(query, key, value, mask=mask) #(batch_size, seq_len, self.n_units)
             for i, attn in enumerate(self.attn_heads)]
         
        # concatenate all attention heads and perform 
        z = torch.cat(z, dim=2) # (Batch, Seq, n_k * n_heads)

        z = self.projection(z) # (Batch, Seq, self.n_units)

        return z #(batch_size, seq_len, self.n_units)



#----------------------------------------------------------------------------------
# The encodings of elements of the input sequence

class WordEmbedding(nn.Module):
    def __init__(self, n_units, vocab):
        super(WordEmbedding, self).__init__()
        self.lut = nn.Embedding(vocab, n_units)
        self.n_units = n_units

    def forward(self, x):
        #print (x)
        return self.lut(x) * math.sqrt(self.n_units)


class PositionalEncoding(nn.Module):
    def __init__(self, n_units, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, n_units)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, n_units, 2).float() *
                             -(math.log(10000.0) / n_units))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], 
                         requires_grad=False)
        return self.dropout(x)



#----------------------------------------------------------------------------------
# The TransformerBlock and the full Transformer


class TransformerBlock(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(TransformerBlock, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(ResidualSkipConnectionWithLayerNorm(size, dropout), 2)
 
    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask)) # apply the self-attention
        return self.sublayer[1](x, self.feed_forward) # apply the position-wise MLP


class TransformerStack(nn.Module):
    """
    This will be called on the TransformerBlock (above) to create a stack.
    """
    def __init__(self, layer, n_blocks): # layer will be TransformerBlock (below)
        super(TransformerStack, self).__init__()
        self.layers = clones(layer, n_blocks)
        self.norm = LayerNorm(layer.size)
        
    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class FullTransformer(nn.Module):
    def __init__(self, transformer_stack, embedding, n_units, vocab_size):
        super(FullTransformer, self).__init__()
        self.transformer_stack = transformer_stack
        self.embedding = embedding
        self.output_layer = nn.Linear(n_units, vocab_size)
        
    def forward(self, input_sequence, mask):
        embeddings = self.embedding(input_sequence)
        return F.log_softmax(self.output_layer(self.transformer_stack(embeddings, mask)), dim=-1)


def make_model(vocab_size, n_blocks=6, 
               n_units=512, n_heads=16, dropout=0.1):
    "Helper: Construct a model from hyperparameters."
    c = copy.deepcopy
    attn = MultiHeadedAttention(n_heads, n_units)
    ff = MLP(n_units, dropout)
    position = PositionalEncoding(n_units, dropout)
    model = FullTransformer(
        transformer_stack=TransformerStack(TransformerBlock(n_units, c(attn), c(ff), dropout), n_blocks),
        embedding=nn.Sequential(WordEmbedding(n_units, vocab_size), c(position)),
        n_units=n_units,
        vocab_size=vocab_size
        )
    
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model


#----------------------------------------------------------------------------------
# Data processing

def subsequent_mask(size):
    """ helper function for creating the masks. """
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0

class Batch:
    "Object for holding a batch of data with mask during training."
    def __init__(self, x, pad=0):
        self.data = x
        self.mask = self.make_mask(self.data, pad)
    
    @staticmethod
    def make_mask(data, pad):
        "Create a mask to hide future words."
        mask = (data != pad).unsqueeze(-2)
        mask = mask & Variable(
            subsequent_mask(data.size(-1)).type_as(mask.data))
        return mask


#----------------------------------------------------------------------------------
# Some standard modules

class LayerNorm(nn.Module):
    "layer normalization, as in: https://arxiv.org/abs/1607.06450"
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class ResidualSkipConnectionWithLayerNorm(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(ResidualSkipConnectionWithLayerNorm, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))


class MLP(nn.Module):
    """
    This is just an MLP with 1 hidden layer
    """
    def __init__(self, n_units, dropout=0.1):
        super(MLP, self).__init__()
        self.w_1 = nn.Linear(n_units, 2048)
        self.w_2 = nn.Linear(2048, n_units)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))