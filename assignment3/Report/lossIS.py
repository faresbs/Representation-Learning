##Evaluate VAE using importance sampling 
#Q2.1

def loss_IS(model, true_x, z):

	#Loop over the elements i of batch
	M = true_x.shape[0]

	#Save logp(x)
	logp_x = np.zeros([M])

	#Get mean and std from encoder
	#2 Vectors of 100
	mu, logvar = model.encode(true_x.to(device))
	std = torch.exp(0.5*logvar)

	K = 200

	#Loop over tha batch
	for i in range(M):
		#z_ik
		samples = z[i,:,:]

		#Compute the reconstructed x's from sampled z's
		x = model.decode(samples.to(device))	

		#Compute the p(x_i|z_ik) of x sampled from z_ik
		#Bernoulli dist = Apply BCE
		#Output an array of losses

		true_xi = true_x[i,:,:].view(-1, 784)
		x = x.view(-1, 784)
		
		#P(x/z) is shape (200, 784)
		p_xz = -(true_xi * torch.log(x) + (1.0-true_xi) * torch.log(1.0-x))
		
		#Multiply the independent probabilities (784)	
		#Apply logsumexp to avoid small image prob
		#P(x/z) is shape (200, 784)
		p_xz = logsumexp(p_xz.cpu().numpy(), axis=1)

	
		##q(z_ik/x_i) follows a normal dist
		#q_z = mgd(samples, mu, std)
		s = std[i, :].view([std.shape[1]])
		m = mu[i, :].view([std.shape[1]])

		q_z = multivariate_normal.pdf(samples.cpu().numpy(),mean=m.cpu().numpy(), cov=np.diag(s.cpu().numpy()**2))

		
		##p(z_ik) follows a normal dist with mean 0/variance 1
		#Normally distributed with loc=0 and scale=1
		std_1 = torch.ones(samples.shape[1])
		mu_0 = torch.zeros(samples.shape[1])	

		p_z = multivariate_normal.pdf(samples.cpu().numpy(),mean=mu_0.cpu().numpy(), cov=np.diag(std_1.cpu().numpy()**2))

		#Multiply the probablities
		
		#logp_x[i] = np.log((1.0/K) * np.sum(np.exp(np.log(p_xz) + np.log(p_z) - np.log(q_z))))
		#logp_x[i] = np.log((1.0/K) * np.sum(np.exp(p_xz.cpu().numpy() + np.log(p_z) - np.log(q_z))))
		logp_x[i] = np.log((1.0/K) * np.sum((p_xz * p_z)/q_z))

	return logp_x