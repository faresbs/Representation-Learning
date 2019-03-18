#For one single attention head
class AttentionHead(nn.Module):
	"""A single attention head"""
	def __init__(self, inp, d_k, dropout):
		
		super(AttentionHead, self).__init__()

		self.dropout = nn.Dropout(dropout)

		#Assuming that the queries, keys, and values have the same dim = d_k
		 

		#d_k = n_units / n_heads
		#inp = n_units/size_hidden from previous attention block or embeddings

		#Affine transformations for queries, keys, and values to get the matrices
		self.query = nn.Linear(inp, d_k, bias=True)	
		self.key = nn.Linear(inp, d_k, bias=True)
		self.value = nn.Linear(inp, d_k, bias=True)

		self.softmax = nn.Softmax()

		# TODO: create/initialize any necessary parameters or layers
        # Initialize all weights and biases uniformly in the range [-k, k],
        # where k is the square root of 1/n_units.
        # Note: the only Pytorch modules you are allowed to use are nn.Linear 
        # and nn.Dropout

        #Init the weights and biases

        #k is the square root of 1/n_units in the case of multihead or n_k for 1 head
		k = np.sqrt(1 / d_k) 

		#Weights of W, K and V matrices
		Wq = torch.Tensor(d_k, inp).uniform_(-k, k)
		Wk = torch.Tensor(d_k, inp).uniform_(-k, k)
		Wv = torch.Tensor(d_k, inp).uniform_(-k, k)

		#biases of W, K and V matrices
		bq = torch.Tensor(d_k).uniform_(-k, k)
		bk = torch.Tensor(d_k).uniform_(-k, k)
		bv = torch.Tensor(d_k).uniform_(-k, k)

		#Fill the linear layers with init weights and biases
		#To avoid problems of grads
		with torch.no_grad():

			self.query.weight.copy_(Wq)
			self.query.bias.copy_(bq)

			self.key.weight.copy_(Wk)
			self.key.bias.copy_(bk)
			
			self.value.weight.copy_(Wv)
			self.value.bias.copy_(bv)

		
				
 
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
		attn = attn / torch.sqrt(d_k)

		#Apply softmax
		#attn = torch.exp(attn)

		#fill attention weights with 0s where padded
		#Which is the opposite of what we want to do
		#if mask is not None: 
		#    attn = attn.masked_fill(mask, 0)

		#Cast to float tensor from byte tensor to perform multiplication
		mask = mask.type(torch.FloatTensor).cuda()

		#Apply mask to attention values
		attn = attn * mask

		#For numerical stability issues
		attn = attn - (10**9) * (1 - mask) 
        
        #Apply softmax
		#attn = torch.exp(attn)

		#Normalize where row values add up to 1
		#attn = attn / attn.sum(-1, keepdim=True)

		#Apply softmax
		attn = self.softmax(attn)

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
		self.projection = nn.Linear(self.n_units, self.n_units, bias=True) 

		##Init the weights and biases for the projection layer
        #k is the square root of 1/n_units
		k = np.sqrt(1 / self.n_units) 

		#Weights of projection layer
		W = torch.Tensor(self.n_units, self.n_units).uniform_(-k, k)

		#bias of projection layer
		b = torch.Tensor(self.n_units).uniform_(-k, k)

		#Fill the linear layers with init weights and biases
		#To avoid problems of grads
		with torch.no_grad():

			self.projection.weight.copy_(W)
			self.projection.bias.copy_(b)

		
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

def clones(module, N):
    """
    A helper function for producing N identical layers (each with their own parameters).
    
    inputs: 
        module: a pytorch nn.module
        N (int): the number of copies of that module to return

    returns:
        a ModuleList with the copies of the module (the ModuleList is itself also a module)
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

# TODO: implement this class
class MultiHeadedAttention(nn.Module):
    def __init__(self, n_heads, n_units, dropout=0.1):
        """
        n_heads: the number of attention heads
        n_units: the number of input and output units
        dropout: probability of DROPPING units
        """
        super(MultiHeadedAttention, self).__init__()
        # This sets the size of the keys, values, and queries (self.d_k) to all 
        # be equal to the number of output units divided by the number of heads.
        self.d_k = n_units // n_heads
        # This requires the number of n_heads to evenly divide n_units.
        assert n_units % n_heads == 0
        self.n_units = n_units 

        # TODO: create/initialize any necessary parameters or layers
        # Initialize all weights and biases uniformly in the range [-k, k],
        # where k is the square root of 1/n_units.
        # Note: the only Pytorch modules you are allowed to use are nn.Linear 
        # and nn.Dropout
        # ETA: you can also use softmax
        # ETA: you can use the "clones" function we provide.

        #Affine transformations for queries, keys, and values to get the matrices
		self.query = nn.Linear(inp, n_units, bias=True)	
		self.key = nn.Linear(inp, n_units, bias=True)
		self.value = nn.Linear(inp, n_units, bias=True)

		self.softmax = F.softmax()

		# TODO: create/initialize any necessary parameters or layers
        # Initialize all weights and biases uniformly in the range [-k, k],
        # where k is the square root of 1/n_units.
        # Note: the only Pytorch modules you are allowed to use are nn.Linear 
        # and nn.Dropout

        #Init the weights and biases

        #k is the square root of 1/n_units
		k = np.sqrt(1 / n_units) 

		#Weights of W, K and V matrices
		Wq = torch.Tensor(d_k, inp).uniform_(-k, k)
		Wk = torch.Tensor(d_k, inp).uniform_(-k, k)
		Wv = torch.Tensor(d_k, inp).uniform_(-k, k)

		#biases of W, K and V matrices
		bq = torch.Tensor(d_k).uniform_(-k, k)
		bk = torch.Tensor(d_k).uniform_(-k, k)
		bv = torch.Tensor(d_k).uniform_(-k, k)

		#Fill the linear layers with init weights and biases
		#To avoid problems of grads
		with torch.no_grad():

			self.query.weight.copy_(Wq)
			self.query.bias.copy_(bq)

			self.key.weight.copy_(Wk)
			self.key.bias.copy_(bk)
			
			self.value.weight.copy_(Wv)
			self.value.bias.copy_(bv)


		#input dim = n_units/size_hidden from previous attention block and outpul dim = n_units
		self.projection = nn.Linear(self.n_units, self.n_units, bias=True) 

		##Init the weights and biases for the projection layer
        #k is the square root of 1/n_units
		k = np.sqrt(1 / self.n_units) 

		#Weights of projection layer
		W = torch.Tensor(self.n_units, self.n_units).uniform_(-k, k)

		#bias of projection layer
		b = torch.Tensor(self.n_units).uniform_(-k, k)

		#Fill the linear layers with init weights and biases
		#To avoid problems of grads
		with torch.no_grad():

			self.projection.weight.copy_(W)
			self.projection.bias.copy_(b)
        



    def forward(self, query, key, value, mask=None):
        # TODO: implement the masked multi-head attention.
        # query, key, and value correspond to Q, K, and V in the latex, and 
        # they all have size: (batch_size, seq_len, self.n_units)
        # mask has size: (batch_size, seq_len, seq_len)
        # As described in the .tex, apply input masking to the softmax 
        # generating the "attention values" (i.e. A_i in the .tex)
        # Also apply dropout to the attention values.

        ##For a single attention
		#d_k is dim of keys
		Q = self.query(query) # (Batch, Seq, self.n_units)
		K = self.key(key) # (Batch, Seq, self.n_units)
		V = self.value(value) # (Batch, Seq, self.n_units)

		#Compute attention score for the head
		z = self.ScaledDotProductAttention(Q, K, V, mask)
		
		return z #size: (batch_size, seq_len, self.n_units)


	def ScaledDotProductAttention(self, Q, K, V, mask):

		if torch.cuda.is_available():
			device = torch.device("cuda")
		else:
			device = torch.device("cpu")

		# get dim of key
		self.n_units = K.size(-1)

		#Should equal dim of queries also 
		assert Q.size(-1) == self.n_units
		
		#we get an attention score between each position in the sequence with current word
		#batch matrix-matrix product, (b,n,m)*(b,m,p)=(b,n,p)
		#(Batch, Seq, self.n_units) * (Batch, self.n_units, Seq) = (batch, Seq, Seq)
		attn = torch.bmm(Q, torch.transpose(K, 1, 2)) #(batch, Seq, Seq)
 
		#scale the dot products by d_k for numerical stability (more stable gradients)
		attn = attn / torch.sqrt(d_k)

		#Apply softmax
		#attn = torch.exp(attn)

		#fill attention weights with 0s where padded
		#Which is the opposite of what we want to do
		#if mask is not None: 
		#    attn = attn.masked_fill(mask, 0)

		#Cast to float tensor from byte tensor to perform multiplication
		mask = mask.type(torch.FloatTensor).cuda()

		#Apply mask to attention values
		attn = attn * mask

		#For numerical stability issues
		#attn = attn - (10**9) * (1 - mask) 
        
        #Apply softmax
		#attn = torch.exp(attn)

		#Normalize where row values add up to 1
		#attn = attn / attn.sum(-1, keepdim=True)

		#Apply softmax
		attn = self.softmax(attn)

		#print (attn)

		#Apply dropout to attention output
		attn = self.dropout(attn)
		
		#Multiply value matrix V with attention scores: 
		#Keep value of words (having high score) we want to focus on and get rid of irrelevant ones (having low score)
		#And Sum up through matrix multiplication
		#Each row corresponds to a single query
		output = torch.bmm(attn, V) 
		
		return output #(Batch, Seq, n_k)

        