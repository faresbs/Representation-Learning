import argparse
import os
import torchvision
import torchvision.transforms as transforms
import torch
import classify_svhn
from classify_svhn import Classifier
import numpy as np
from scipy import linalg



SVHN_PATH = "svhn"
PROCESS_BATCH_SIZE = 32


def get_sample_loader(path, batch_size):
    """
    Loads data from `[path]/samples`

    - Ensure that path contains only one directory
      (This is due ot how the ImageFolder dataset loader
       works)
    - Ensure that ALL of your images are 32 x 32.
      The transform in this function will rescale it to
      32 x 32 if this is not the case.

    Returns an iterator over the tensors of the images
    of dimension (batch_size, 3, 32, 32)
    """
    data = torchvision.datasets.ImageFolder(
        path,
        transform=transforms.Compose([
            transforms.Resize((32, 32), interpolation=2),
            classify_svhn.image_transform
        ])
    )
    data_loader = torch.utils.data.DataLoader(
        data,
        batch_size=batch_size,
        num_workers=2,
    )
    return data_loader


def get_test_loader(batch_size):
    """
    Downloads (if it doesn't already exist) SVHN test into
    [pwd]/svhn.

    Returns an iterator over the tensors of the images
    of dimension (batch_size, 3, 32, 32)
    """
    testset = torchvision.datasets.SVHN(
        SVHN_PATH, split='test',
        download=True,
        transform=classify_svhn.image_transform
    )
    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=batch_size,
    )
    return testloader


def extract_features(classifier, data_loader):
    """
    Iterator of features for each image.
    """
    with torch.no_grad():
        for x, _ in data_loader:
            #print(x.shape)
            #print(x)
            h = classifier.extract_features(x).numpy()
            #print("h.shape=",h.shape)
            for i in range(h.shape[0]):
                yield h[i]


def calculate_fid_score(sample_feature_iterator,
                        testset_feature_iterator):
    """
    To be implemented by you!
    """
    #Extract the representation features of the generated images
    #from the given model
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
        test_features= np.vstack([test_features,i.reshape(1,512)])  if test_features.size else i.reshape(1,512)
        test_size+=1
        if test_size==1000: break
    test_features = test_features.T
    
    print("Estimating the mean ...")
    #Estimating the mean of the generated images
    mu_gen = np.mean(gen_features,axis=1).reshape(512,1)
    
    #Estimating the mean of the test images
    mu_test = np.mean(test_features,axis=1).reshape(512,1)
    
    print("Estimating the variance ...")
    # We use the unbiased variance estimate which is given by (X-mu)(X-mu)^T/(n-1)
    gen_centered  = gen_features - mu_gen
    test_centered = test_features - mu_test
    
    sigma_gen = np.matmul(gen_centered, gen_centered.T) / (gen_size - 1)
    sigma_test = np.matmul(test_centered, test_centered.T) / (test_size - 1)
    
    print("Calculating the sqrt of cov matrices product ...")
    # The sqrt of a matrix A needs A to be symetric, but if A, and B are sysmetric
    # A.B is not symeyric necessarly. To solve that we use this trick:
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
    
    
    #raise NotImplementedError(
    #    "TO BE IMPLEMENTED."
    #    "Part of Assignment 3 Quantitative Evaluations"
    #)
    
    return fid


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Score a directory of images with the FID score.')
    parser.add_argument('--model', type=str, default="svhn_classifier.pt",
                        help='Path to feature extraction model.')
    parser.add_argument('directory', default="images", type=str,
                        help='Path to image directory')
    args = parser.parse_args()

    quit = False
    if not os.path.isfile(args.model):
        print("Model file " + args.model + " does not exist.")
        quit = True
    if not os.path.isdir(args.directory):
        print("Directory " + args.directory + " does not exist.")
        quit = True
    if quit:
        exit()
    print("Test")
    classifier = torch.load(args.model, map_location='cpu')
    classifier.eval()

    sample_loader = get_sample_loader(args.directory,
                                      PROCESS_BATCH_SIZE)
    sample_f = extract_features(classifier, sample_loader)

    test_loader = get_test_loader(PROCESS_BATCH_SIZE)
    test_f = extract_features(classifier, test_loader)

    fid_score = calculate_fid_score(sample_f, test_f)
    print("FID score:", fid_score)
