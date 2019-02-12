import numpy as np
import fastmat as fm
from usmltransform import CudaBlockTwoLevelToeplitz
import matplotlib.pyplot as plt

C = CudaBlockTwoLevelToeplitz(
    nZ1=520,
    nZ2=520,
    nX=361,
    nY=70,
    pulseLength=142.0,
    dx=0.5e-3,
    dy=0.5e-3,
    dz=(0.5 * 5920.0)/80e6,
    centerFreq=3.4e6,
    bandWidth=2.2361e6,
    speedOfSound=5920.0,
    samplingFreq=80e6,
    foreRunLength=70.3e-3,
    beamAngle=0.26794919243
)

print(C.shape)

#arrRef = C.reference()
#print(arrRef)

x = np.random.randn(C.shape[1]).astype(np.float32)
print("norm(x): %e"% (np.linalg.norm(x)))

y = C * x
print("norm(y): %e"% (np.linalg.norm(y)))

#z = arrRef.dot(x)
#print("norm(z): %e"% (np.linalg.norm(z)))
#print("norm(z-y): %e"% (np.linalg.norm(z-y)))

