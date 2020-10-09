'''
The underlying functions derive a conditional multivariate normal distribution (function 1)
and generate a random sample from the conditional distribution.
Both functions are a one-to-one Python implementation of the existing R modules, found here:
https://rdrr.io/cran/condMVNorm/src/R/condMVNorm.r
'''

# Function 1: Estimate a conditional multivariate normal distribution
def conmvn(dependent_indx, given_indx, X_given, mean, covmat, check_covmat = True):
    '''
    Finds the conditional mean and conditional variance covariance matrix
    '''
    # Conditions
    if check_covmat == True:
        assert np.allclose(covmat, covmat.T, 1e-05, 1e-08),\
            'provided variance covariance matrix needs to be symmetric'
        assert np.any(np.linalg.eig(covmat < 1e-8)[0]),\
            'provided variance covariance matrix needs to be positive-definite'
    if str(type(covmat)).endswith("frame.DataFrame"):
        covmat = covmat.values
    # Decompose covariance matrix
    try:
        sigma_sbsb = covmat[dependent_indx][:, dependent_indx]
    except:
        sigma_sbsb = covmat[dependent_indx, dependent_indx]
    sigma_sbs = covmat[dependent_indx][:, given_indx]
    try:
        sigma_ss = covmat[given_indx][:, given_indx]
    except:
        sigma_ss = covmat[given_indx,given_indx]
    try: 
        sigma_sbsSigmassInv = np.dot(sigma_sbs,np.linalg.inv(sigma_ss))
    except:
        sigma_sbsSigmassInv = sigma_sbs / sigma_ss
    try:
        conMu = mean[dependent_indx] + np.dot(sigma_sbsSigmassInv, (X_given - mean[given_indx]))
    except:
        conMu = mean[dependent_indx] + np.dot(sigma_sbsSigmassInv, (X_given - mean[given_indx]))
    conVar = sigma_sbsb - np.dot(sigma_sbsSigmassInv, sigma_sbs.T)

    return conMu, conVar;

# Function 2: Random sample generator
def rg_condmvn(n, dependent_indx, given_indx, X_given, mean, covmat, check_covmat = True):
    '''
    Random number generator from a conditional multivariate normal distribution
    '''
    conMu, conVar = conmvn(dependent_indx, given_indx, X_given, mean, covmat, check_covmat = True)
    result = np.random.multivariate_normal(conMu, conVar, n)

    return result