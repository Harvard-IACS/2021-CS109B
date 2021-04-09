#solution
class Analytical_KL:
    def __init__(self, Lambda):
        self.Lambda = Lambda
    def __call__(self, mu, logvariance):
        lossval = np.sum(-0.5*(1 + logvariance - mu**2 - logvariance),axis=-1)
        return lossval 
