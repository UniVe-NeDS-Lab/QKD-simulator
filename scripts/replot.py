from scipy.stats import beta , gamma, lognorm
import seaborn as sns
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '..')
from qber import FSOQKD

      

def gen_rate(n, kind='urban'):
    """ returns the SKR in mb/s """
    if kind == 'urban':
        s, loc, scale = 0.04522353283931704, -27619.843260475463,\
            30045.119473174826
        rv = lognorm(s,loc=loc, scale=scale)
        
        # these are the numbers for uncapped length
        #z, loc, scale = 2.3545759677799762, -347.41797231322744, \
        #    1959.3001069474171
        #rv = gamma(z,loc=loc, scale=scale)
        #rv = gamma.rvs(z,loc=loc, scale=scale, size=10000)
    elif kind == 'suburban':
        z, loc, scale = 1.2126718606095204, 0.9714686346219052, \
            1201.7735807830677
        #a, b, loc, scale = 1.0192923828462457, 749.7202469556448, \
        #    0.9926676075144574, 1359383.7402280043
        #rv = beta(a,b,loc=loc, scale=scale)
        rv = gamma(z,loc=loc, scale=scale)
    else:
        z, loc, scale = 1.1266204361283103, 0.9969766163389133,\
            1112.0829398842093
        # z, loc, scale = 1.4008467231626485, -29.12897636784828,\
        #    965.3878800954703
        # rv = gamma(z,loc=loc, scale=scale)
        rv = gamma(z,loc=loc, scale=scale)
    
    fso = FSOQKD()
    freq = 600_000_000
    for i in range(n):
        distance = rv.rvs(size=1)[0]
        yield distance, fso.get_rate(distance, freq)[3]/1_000_000

def plot_skr():
    dist = []
    rate = []
    for d, r in gen_rate(10000, kind='urban'):
        rate.append(r)
        dist.append(d)
    f, axes = plt.subplots(1,2)
    axes[0].hist(dist, density=True, bins=30)
    axes[0].set_title('histogram of link lenght')
    axes[0].set(xlabel='m')
    axes[1].hist(rate, density=True, bins=30)
    axes[1].set_title('histogram of key rate')
    axes[1].set(xlabel='mb/s')
    plt.show()
    
plot_skr()

