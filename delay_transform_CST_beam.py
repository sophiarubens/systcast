import numpy as np
from matplotlib import pyplot as plt
import cmasher as cmr
from scipy.fft import fftn,fftshift,ifftshift
from forecasting_pipeline import cosmo_stats
from astropy import units as u

dir="/Users/sophiarubens/Downloads/research/code/pipeline/"
cst_fname="fidu_CST_0.58_0.62_0.0002_MHz.npy"
cstbeam=np.load(dir+cst_fname)
Nxy,_,Nz=cstbeam.shape
Lxy,Lz=np.genfromtxt("Lxyz.txt")*u.Mpc
cyl_extent=np.genfromtxt("cyl_extent.txt")

# delay transform
delaytransformed=fftshift(fftn(ifftshift(
                                          cstbeam,         # not bothering with an area element or normalization since I'm mostly interested in qualitative impressions here
                                          axes=-1
                                        ),
                                axes=-1,
                                norm="backward" #,
                                # s=(Nxy,Nxy,Nz)
                               ),
                            axes=-1
                          )

# bin cylindrically
binning_holder=cosmo_stats(Lxy,Lz=Lz,
                           P_fid=delaytransformed*u.mK**2*u.Mpc**3) # might need to pass as T_pristine
binning_holder.bin_power()
binned_delay_transformed_beam=binning_holder.P_binned

# plot
plt.figure()
plt.imshow(np.abs(binned_delay_transformed_beam)**2,
           origin="lower", extent=cyl_extent, aspect=0.15,
           cmap=cmr.amber,norm="log"
           )
plt.title("delay spectrum of CST beam")
plt.xlabel("k$_\perp$ (1/Mpc)")
plt.ylabel("k$_{||}$ (1/Mpc)")
plt.colorbar()
plt.savefig("binned_delay_transformed_beam.png",dpi=300)