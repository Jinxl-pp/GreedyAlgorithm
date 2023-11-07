import numpy as np
from quadrature import Quadrature

## =====================================
## deterministic samples
## numpy    


class QuasiMonteCarloQuadrature():
    
    def __init__(self, device):
        """ The Quasi-Monte-Carlo sampling information on rectangle domains in any dimension.
            Generates a deterministic set of samples by using the Halton sequence. If you have 
            installed scipy package, try calling: 
            
                        qmc = QuasiMonteCarloQuadrature(device)
                        samples = qmc.n_rectangle_samples(n_rectangle, number_of_samples).
                        
            Otherwise there is another version (much slower when large amount):
            
                        samples = qmc._n_rectangle_samples(n_rectangle, number_of_samples).
            
        Args:
                device: cuda/cpu
        """     

        self.device = device
        self.qtype = "QMC"
        
    def _prime(self, index):
        
        prime = np.array([2,   3,   5,   7,   11,  13,  17,  19,  23,  29,  31,  37,  41,  43, 
                          47,  53,  59,  61,  67,  71,  73,  79,  83,  89,  97,  101, 103, 107, 
                          109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 
                          191, 193, 197, 199, 211, 223, 227, 229, 233, 239, 241, 251, 257, 263,
                          269, 271, 277, 281, 283, 293, 307, 311, 313, 317, 331, 337, 347, 349, 
                          353, 359, 367, 373, 379, 383, 389, 397, 401, 409, 419, 421, 431, 433, 
                          439, 443, 449, 457, 461, 463, 467, 479, 487, 491, 499, 503, 509, 521, 
                          523, 541, 547, 557, 563, 569, 571, 577, 587, 593, 599, 601, 607, 613, 
                          617, 619, 631, 641, 643, 647, 653, 659, 661, 673, 677, 683, 691, 701, 
                          709, 719, 727, 733, 739, 743, 751, 757, 761, 769, 773, 787, 797, 809, 
                          811, 821, 823, 827, 829, 839, 853, 857, 859, 863, 877, 881, 883, 887, 
                          907, 911, 919, 929, 937, 941, 947, 953, 967, 971, 977, 983, 991, 997])
        
        assert index <= prime.size, "Only support primes within 1000."
        return prime[index]
        
        
    def _get_standard_halton(self, number_of_samples, prime_base):
        """ Halton sequence on the standard domain [0,1].
        
        Args:
                number_of_samples: the length of Halton sequence
                prime_base: the prime number by which [0,1] will be divided
        """
        
        sequence  = np.zeros(number_of_samples)
        numer_of_bits = int(1 + np.ceil(np.log(number_of_samples)/np.log(prime_base)))
        
        frac_base = prime_base ** (-(np.linspace(1,numer_of_bits,numer_of_bits)))
        working_base = np.zeros(numer_of_bits)
        
        for i in range(number_of_samples):
            j = 0
            condition = True
            while condition:
                working_base[j] += 1
                if working_base[j] < prime_base:
                    condition = False
                else:
                    working_base[j] = 0
                    j += 1 
            sequence[i] = np.dot(working_base, frac_base)
            
        return sequence
        
        
    def _n_rectangle_samples(self, n_rectangle, number_of_samples):
        """ The Quasi-Monte-Carlo method supports generating samples in any dimension
            but only in domains with rectangle shapes. Standard sequence of samples will
            be generated in [0,1]^n with n the dimensionality. However, scaling process
            is allowed in this function.

        Args:
            n_rectangle (np.array): rectangle in any dimension
            number_of_samples (int): total number of samples
        """
        
        # dimensionality 
        dim = n_rectangle.shape[0]
        
        # get scaled qmc sequence 
        lengths = np.zeros(dim)
        quadpts = np.zeros((number_of_samples, dim))
        for i in range(dim):
            prime = self._prime(i)
            length = n_rectangle[i][1] - n_rectangle[i][0]
            sequence = self._get_standard_halton(number_of_samples, prime)
            quadpts[...,i] = n_rectangle[i][0] + length * sequence
            lengths[i] = length
        
        # get volumn (measure) of domain
        measure = lengths.prod()
        weights = measure * np.ones((quadpts.shape[0],1)).astype(np.float64) 
        h = np.array([1 / number_of_samples])
        
        return Quadrature(self.device, quadpts, weights, h)      
            
    
    def n_rectangle_samples(self, n_rectangle, number_of_samples):
        
        # needs scipy.stats.qmc
        from scipy.stats import qmc 
        
        # dimensionality
        dim = n_rectangle.shape[0]
        
        # get scaled qmc sequence 
        sampler = qmc.Halton(d=dim, scramble=False, seed=10)
        sample = sampler.random(n=number_of_samples)
        quadpts = qmc.scale(sample, n_rectangle[...,0], n_rectangle[...,1])
        
        # get volumn (measure) of domain
        lengths = np.zeros(dim)
        for i in range(dim):
            length = n_rectangle[i][1] - n_rectangle[i][0]
            lengths[i] = length
        measure = lengths.prod()
        weights = measure * np.ones((quadpts.shape[0],1)).astype(np.float64) 
        h = np.array([1 / number_of_samples])
        
        return Quadrature(self.device, quadpts, weights, h)      
        