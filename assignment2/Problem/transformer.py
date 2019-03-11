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

