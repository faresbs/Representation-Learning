def calculate_fid_score(sample_feature_iterator,
                        testset_feature_iterator):
    gen_features = np.array([])
    test_features = np.array([])
    
    print("Extracting the features ...")
    #sample_feature_iterator is a generator of a minibatch of features images, 
    #that is the last conv2d layer of the classifier of 512 features
    
    #For generated images
    gen_size=0
    for i in sample_feature_iterator: #iterate over minibatch images
        #Now let's get the activation of the images
        gen_features = np.vstack([gen_features,i.reshape(1,512)])  \
        if gen_features.size else i.reshape(1,512)
        gen_size+=1
        if gen_size==1000: break
    gen_features = gen_features.T
    
    #For test images
    test_size=0
    for i in testset_feature_iterator: #iterate over test images
        test_features= np.vstack([test_features,i.reshape(1,512)])  
        test_size+=1
        if test_size==1000: break
    test_features = test_features.T
    
    print("Estimating the mean ...")
    #Estimating the mean of the generated images
    mu_gen = np.mean(gen_features,axis=1).reshape(512,1)
    
    #Estimating the mean of the test images
    mu_test = np.mean(test_features,axis=1).reshape(512,1)
    
    print("Estimating the variance ...")
    # We use the unbiased variance estimate which
    #is given by (X-mu)(X-mu)^T/(n-1)
    gen_centered  = gen_features - mu_gen
    test_centered = test_features - mu_test
    
    sigma_gen = np.matmul(gen_centered, gen_centered.T) / (gen_size - 1)
    sigma_test = np.matmul(test_centered, test_centered.T) / (test_size - 1)
    
    print("Calculating the sqrt of cov matrices product ...")
    # The sqrt of a matrix A needs A to be symetric, but if A, and B 
    # are sysmetric A.B is not symeyric necessarly. 
    # To solve that we use this trick:
    # sqrt(sigma1 sigma2) = sqrt(A sigma2 A), where A = sqrt(sigma1)
    # the covariance matrix are by definition symetric
    
    # to prevent negative values in the cov product
    eps = np.eye(512) * 1e-5
    
    root_sigma_gen = linalg.sqrtm(sigma_gen + eps)
    sigmas_prod = np.matmul(root_sigma_gen,np.matmul(sigma_test, root_sigma_gen))
    # given np.matmul(root_sigma_gen,np.matmul(sigma_test, root_sigma_gen)) is symetric:
    root_sigmas_prod = linalg.sqrtm(sigmas_prod + eps)
    
    print("Calculating the FID score ...")
    # Calculating the trace
    trace = np.trace(sigma_test + sigma_gen - 2.0 * root_sigmas_prod)
    
    # Calculate the squared norm between means
    squared_norm = np.sum((mu_test - mu_gen)**2)

    # Calculate the fid score
    fid = squared_norm + trace

    return fid