# position and velocity bounds
vLOW = -0.07
vHIGH = 0.07
xLOW = -1.2
xHIGH = 0.5

# no of tiles
nx = 6
nv = 6


# GUI scalings
scalex=500
scaley=500

scale=100

# default path to store policies
defaultpath = 'models/temp-model.pkl'

# default values of SARSA algorithm parameters
#defaultalpha = 0.1
defaultalpha = 0.9
#defaultnoepisodes = 1000
#defaultgamma = 1
defaultgamma = .95
defaultepsilon = 0.1
defaultinitq = 0
defaultlambda = 0.1
timethreshold = 5000
