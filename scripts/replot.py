from scipy.stats import beta , gamma
import seaborn as sns
import matplotlib.pyplot as plt

f = plt.figure()
#sf.subfigures(1,3)

# gamma, urban
z, loc, scale = 2.3545759677799762, -347.41797231322744, 1959.3001069474171
rv = gamma.rvs(z,loc=loc, scale=scale, size=10000)
print(rv.mean())
# beta, suburban
a, b, loc, scale = 1.0192923828462457, 749.7202469556448, 0.9926676075144574, 1359383.7402280043
rv = beta.rvs(a,b,loc=loc, scale=scale, size=10000)
print(rv.mean())
# gamma, rural
z, loc, scale = 1.4008467231626485, -29.12897636784828, 965.3878800954703
rv = gamma.rvs(z,loc=loc, scale=scale, size=10000)
print(rv.mean())
plt.show()

