import numpy as np

from matplotlib import pyplot as plt
from matplotlib import gridspec
from matplotlib.colors import LogNorm,TwoSlopeNorm,SymLogNorm ,CenteredNorm 

from scipy.fft import fftshift,ifftshift,fftfreq, fftn, irfftn, set_workers
from scipy.integrate import quad
from scipy.interpolate import RectBivariateSpline as RBS
from scipy.interpolate import RegularGridInterpolator as RGI
from scipy.interpolate import griddata as gd
from scipy.signal import convolve
from scipy.stats import binned_statistic_dd

import camb

from astropy.cosmology import Planck18
from astropy.cosmology.units import littleh
from astropy import constants as const
from astropy.units import Quantity
from astropy import units as u
from py21cmsense import GaussianBeam, Observatory, Observation, PowerSpectrum

import cmasher
import inspect
import json
import pandas as pd
from pathlib import Path
import pygtc
import time
# import io

set_workers(6)

# cosmological. all are Planck18, whether they come from astropy or not
H0=Planck18.H0
h=H0/100
Omegam=Planck18.Om0
Omegamh2=Omegam*h**2
Omegab=Planck18.Ob0
Omegabh2=Omegab*h**2
Omegach2=0.12011
OmegaLambda=0.6842
Omegak=0
Omegar=1-OmegaLambda-Omegam-Omegak
ln1010AS=3.0448
AS=np.exp(ln1010AS)/10**10
ns=0.96605
w=-1
Omegamh2=Omegam*h**2
pars_fidu=    [ H0,    Omegabh2,      Omegamh2,      AS,           ns,    w] # suitable for getting matter power spec
parnames_fidu=["H_0", "Omega_b h^2", "Omega_c h^2", "10^9 * A_S", "n_s", "w"]

pars_forecast=    [ H0,    Omegabh2,      Omegach2,      w  ] # expect a 21-cm experiment to provide insight into these
parnames_forecast=["H_0", "Omega_b h^2", "Omega_c h^2", "w"]

dpar_default=1e-3*np.ones(len(pars_fidu))
dpar_default[3]*=1e-9

# physical
nu_HI_z0=1420.405751768*u.MHz
c=const.c
dif_lim_prefac=1.029

# mathematical
pi=np.pi
twopi=2.*pi
ln2=np.log(2)

# numerical
maxint=   np.iinfo(np.int64  ).max
BasicAiryHWHM=1.616339948310703178119139753683896309743121097215461023581 # intentionally preposterous number of sig figs from Mathematica
eps=1e-15
dpi_to_use=250

# CHORD
N_NS_full=24
N_EW_full=22
b_NS=8.5*u.m
b_EW=6.3*u.m
b_max_CHORD=np.sqrt((N_NS_full*b_NS)**2+(N_EW_full*b_EW)**2)*u.m
DRAO_lat=49.320791*pi/180.*u.rad # Google Maps satellite view, eyeballing what looks like the middle of the CHORD site: 49.320791, -119.621842 (bc considering drift-scan CHIME-like "pointing at zenith" mode, same as dec)
D=6.*u.m
CHORD_channel_width_MHz=0.1953125*u.MHz
def_observing_dec=pi/60.
def_offset=1.75*pi/180. # for this placeholder state where I build up the CHORD layout using rotation matrices instead of actual measurements. probably add Hans' mask at some point to punch the corners and receiver hut holes out...
def_pbw_pert_frac=1e-2
def_evol_restriction_threshold=1./30. # HERA 1/15 was made up. turn this down for a computationally less intense substitute
img_bin_tol=5 # ringing is remarkably insensitive to turning this down; you get really bad scale mismatch by turning it up... the real solution was the "need good resolution in both Fourier and configuration space" thing
def_PA_N_grid_pix=256
N_fid_beam_types=1
integration_s=10*u.s # seconds
hrs_per_night=8*u.hr # borrowed from Debanjan / 21cmSense
# N_nights=100 # also borrowed from Debanjan / 21cmSense
N_nights=1
# def_N_timesteps=int(N_nights*hrs_per_night//integration_s)
def_N_timesteps=1 # for local tests
print("def_N_timesteps=",def_N_timesteps)

# side calculations
def get_padding(n): # avoid edge effects in a convolution
    padding=n-1
    padding_lo=int(np.ceil(padding / 2))
    padding_hi=padding-padding_lo
    return padding_lo,padding_hi
def synthesized_beam_crossing_time(nu,bmax,dec=30.*u.deg): # to accumulate rotation synthesis
    synthesized_beam_width_rad=dif_lim_prefac*(c/nu)/bmax
    beam_width_deg=synthesized_beam_width_rad*180/pi
    crossing_time_hrs_no_dec=beam_width_deg/15
    crossing_time_hrs= crossing_time_hrs_no_dec*np.cos(dec.to(u.rad))
    return crossing_time_hrs
def extrapolation_warning(regime,want,have):
    print("WARNING: if extrapolation is permitted in the interpolate_P call, it will be conducted for {:15s} (want {:9.4}, have{:9.4})".format(regime,want,have))
def comoving_dist_arg(z,Omegam=Omegam,OmegaLambda=OmegaLambda): # this is 1/ E(z)
    return 1/np.sqrt(Omegam*(1+z)**3+OmegaLambda)
def comoving_distance(z=0.5,H0=H0,Omegam=Omegam,OmegaLambda=OmegaLambda):
    integral,_=quad(comoving_dist_arg,0,z,args=(Omegam,OmegaLambda,))
    return (c.value*integral)/(H0.value*1000)*u.Mpc

# typical trivial conversions
def freq2z(nu_rest,nu_obs):
    nu_obs=nu_obs.to(nu_rest.unit)
    return nu_rest.value/nu_obs.value-1.
def z2freq(nu_rest=600.*u.MHz,z=nu_HI_z0/(600*u.MHz)-1.):
    return nu_rest/(z+1)

# Fourier space
def kpar(nu_ctr=600*u.MHz,chan_width=0.1953125*u.MHz,N_chan=300,H0=H0): # not pure cosmo. relies on LoS details of survey
    prefac=1e3*twopi*H0.value*nu_HI_z0.value/c.value # 1e3 to account for units of H0/c ... assumes nu_HI_z0 and chan_width have the same units
    z_ctr=freq2z(nu_HI_z0,nu_ctr)
    Ez=1/comoving_dist_arg(z_ctr)
    zterm=Ez/((1+z_ctr)**2*chan_width.value)
    kparmax=prefac*zterm
    kparmin=kparmax/N_chan
    Delta_kpar=kparmin
    kpar_bins=np.arange(kparmin,kparmax+Delta_kpar,Delta_kpar)/u.Mpc # units by construction
    return kpar_bins # evaluating at the z of the central freq of the survey (trusting slow variation...)
def kperp(nu_ctr=600.*u.MHz,bmin=6.*u.m,bmax=500.*u.m): # not pure cosmo. relies on sky plane details of survey
    Dc=comoving_distance(freq2z(nu_HI_z0,nu_ctr)) # evaluating at the z of the central freq of the survey (rely on slow variation = not worth reevaluating at each freq, as usual)
    prefac=twopi*nu_HI_z0.value*1e6/(c.value*Dc.value)
    kperpmin=prefac*bmin.value
    kperpmax=prefac*bmax.value
    Delta_kperp=kperpmin
    kperp_bins=np.arange(kperpmin,kperpmax+Delta_kperp,Delta_kperp)/u.Mpc # units by construction
    return kperp_bins
def wedge_kpar(nu_ctr,kperp,H0=H0,nu_rest=nu_HI_z0): # for some kperps of interest, which kparallels will the interferometer smear the wedge up to?
    nu_rest=nu_rest.to(nu_ctr.unit)
    z=freq2z(nu_rest,nu_ctr)
    E=1/comoving_dist_arg(z)
    Dc=comoving_distance(z)
    prefac=(H0*Dc*E).decompose()/(c*(1+z))
    return prefac*kperp
def calc_b_HI(z):
    return 1.489 +0.460*(z-1) -0.118*(z-1)**2 +0.0678*(z-1)**3 -0.0128*(z-1)**4 +0.0009*(z-1)**5 # https://arxiv.org/abs/1804.09180 # Villaescusa-Navarro 2018. Widely accepted, but CHIME disagrees. CHIME is just one data point, but CHORD will probably be doing early science at similarly nonlinear scales
def Blackman_Harris_safe_for_FFT(N): # !!centre-origin!! apodization function
    a0,a1,a2,a3=0.35875,0.48829,0.14128,0.01168 # from the MATLAB (!) docs https://www.mathworks.com/help/signal/ref/blackmanharris.html
    n=np.arange(N)
    w= a0 \
      -a1*np.cos(twopi*n/N) \
      +a2*np.cos(4.*pi*n/N) \
      -a3*np.cos(6.*pi*n/N)
    return w
def comprehensive_slice_figure(box,                      # 3D box to plot slices of
                               norm=None,                # norm of the colour scale
                               name="placeholder.png",   # name of the output figure
                               dpi=500,                  # resolution of the output figure
                               fracs=[0,1e-5,1/3,1/2,1], # fractions along each axis at which to slice the box
                               cmap=None
                               ):
    box_shape=box.shape
    assert len(box_shape)==3, "this plotting function requires a 3D box"
    Nx,Ny,Nz=box_shape
    _,axs=plt.subplots(len(fracs),3,layout="constrained",figsize=(8,15))
    axs[0,0].set_title("x index 0/"+str(Nx-1))
    axs[0,1].set_title("y index 0/"+str(Ny-1))
    axs[0,2].set_title("z index 0/"+str(Nz-1))
    for i,frac in enumerate(fracs):
        x_idx=int(frac*Nx)
        y_idx=int(frac*Ny)
        z_idx=int(frac*Nz)
        if frac==1:
            x_idx-=1
            y_idx-=1
            z_idx-=1
        elif frac==1e-5:
            x_idx=1
            y_idx=1
            z_idx=1
        if i>0:
            axs[i,0].set_title(str(x_idx)+"/"+str(Nx-1))
            axs[i,1].set_title(str(y_idx)+"/"+str(Ny-1))
            axs[i,2].set_title(str(z_idx )+"/"+str(Nz-1))

        sl0=box[x_idx,:,:]
        img=axs[i,0].imshow(sl0.T,origin="lower",norm=norm,cmap=cmap)
        plt.colorbar(img,ax=axs[i,0])
        axs[i,0].set_xlabel("y idx")
        axs[i,0].set_ylabel("z idx")

        sl1=box[:,y_idx,:]
        img=axs[i,1].imshow(sl1.T,origin="lower",norm=norm,cmap=cmap)
        plt.colorbar(img,ax=axs[i,1])
        axs[i,1].set_xlabel("x idx")
        axs[i,1].set_ylabel("z idx")

        sl2=box[:,:,z_idx]
        img=axs[i,2].imshow(sl2.T,origin="lower",norm=norm,cmap=cmap)
        plt.colorbar(img,ax=axs[i,2])
        axs[i,2].set_xlabel("x idx")
        axs[i,2].set_ylabel("y idx")
    plt.savefig(name, dpi=dpi)
    plt.close()

# main computations
"""
this class helps compute contaminant power and cosmological parameter biases
using a Fisher-based formalism and numerical windowing for power beams with  
assorted properties and systematics.
"""

class beam_effects(object):
    def __init__(self,
                 # SCIENCE
                 # the observation
                 bmin:float=b_EW,bmax:float=b_max_CHORD,                          # max and min baselines of the array
                 nu_ctr:float=600.*u.MHz,                                         # central freq of survey
                 delta_nu:float=CHORD_channel_width_MHz,                          # channel width
                 evol_restriction_threshold:float=def_evol_restriction_threshold, # how close to coeval is close enough? \Delta z/z
                 
                 # parameters of per-antenna systematic–aware beam synthesis
                 N_pbws_pert:int=0,                 # number of beams to perturb
                 ioname:str="placeholder",          # unique identifier for saving files and figures related to the uv coverage of this scenario
                 antenna_distribution:str="random", # random, column, corner, or frame distribution of fiducial beam types?
                 array_version:str="full",          # full or pathfinder CHORD?

                 # beam config
                 CST_lo=None,CST_hi=None,                                     # low and high frequencies of the CST simulation band (GHz !!!!!!!!!!not MHz)
                 CST_deltanu=None,                                            # frequency spacing of CST simulations (MHz)
                 beam_sim_directory=None,                                     # directory to import CST simulations from 
                 beam_domain:np.ndarray=None,                                 # config space pts at which a pre–discretely sampled beam is known
                 f_mid1:str="pol1/f_",f_mid2:str="pol1/f_",                   # middle part of CST file names... should include something distinguish the two polarizations (not enforced)
                 f_tail:str="_GHz.txt",                                       # trailing part of CST file names 
                 CST_f_head_fidu:str="fiducial/",CST_f_head_syst:str="syst/", # start of CST file names for different beam types (see Memo I for terminology description)
                 pointing_errors=[0.,0.,0.],                                  # subject the real and thgt beams to pointing errors 

                 # FORECASTING
                 pars_set_cosmo:np.ndarray=pars_fidu,          # cosmo params to condition CAMB calls
                 pars_forecast:np.ndarray=pars_fidu,           # cosmo params of interest for a forecast
                 pars_forecast_names:np.ndarray=parnames_fidu, # >>>>> coming soon: support for derived parameters <<<<<
                 P_fid_for_cont_pwr=None,                      # fiducial power spectrum to use in Monte Carlo... typical choice for forecasting is CAMB (enforced default); some analyses may favour, for example, a flat spectrum
                 k_idx_for_window:int=0,                       # examine contaminant power or window functions?
                 wedge_cut:bool=False,                         # excise info from voxels inside the foreground wedge?
                 layer_foregrounds:bool=True,                  # add synchrotron foregrounds on top of cosmo + beam data?

                 # NUMERICAL 
                 N_theory_k:int=4096,                 # how many points in the cosmo power spectrum?
                 dpar=None,                           # initial guess for numerical partial derivative step size
                 init_and_box_tol:float=0.05,         # how much wider to make the config space extent of the brightness temp boxes compared to the survey box (numerical insurance factor...)
                 CAMB_tol:float=0.05,                 # same thing but for the CAMB call (if you make a sensible choice here, you will never have to extrapolate the cosmo spectrum to get info about a part of k-space you're interested in)
                 frac_tol_conv:float=0.1,             # fraction (not percent) convergence for Monte Carlo ensemble -> used to determine the number of necessary realizations
                 seed=None,                           # specify a seed if you want replicable RNG behaviour
                 ftol_deriv:float=1e-16,              # this numerical tolerance factor * the function you are trying to differentiate gives a pointwise comparison for whether the derivative computation is accurate enough with the current step size
                 maxiter:int=5,                       # maximum number of times the partial derivative computation can recurse with an updated step size estimate
                 PA_N_grid_pix:int=def_PA_N_grid_pix, # number of pixels per side of gridded uv plane
                 LoS_taper=False,image_taper=False,   # apply apodization along the line of sight or transverse directions?

                 # CONVENIENCE
                 heavy_beam_recalc:bool=True,         # save time by using pre-saved synthesized beams?
                 ):   
                
        # forecasting considerations
        self.seed=seed
        self.pars_set_cosmo=pars_set_cosmo
        self.N_pars_set_cosmo=len(pars_set_cosmo)
        self.pars_forecast=pars_forecast
        self.N_pars_forecast=len(pars_forecast)
        self.N_theory_k=N_theory_k
        self.dpar=dpar
        self.wedge_cut=wedge_cut
        nu_ctr=nu_ctr.to(u.MHz)
        self.nu_ctr=nu_ctr
        self.Deltanu=delta_nu
        self.bw=nu_ctr*evol_restriction_threshold
        self.Nchan=int(self.bw/self.Deltanu)
        self.z_ctr=freq2z(nu_HI_z0,nu_ctr)
        self.nu_lo=self.nu_ctr-self.bw/2.
        self.z_hi=freq2z(nu_HI_z0,self.nu_lo)
        self.Dc_hi=comoving_distance(self.z_hi)
        self.nu_hi=self.nu_ctr+self.bw/2.
        self.z_lo=freq2z(nu_HI_z0,self.nu_hi)
        self.Dc_lo=comoving_distance(self.z_lo)
        self.deltaz=self.z_hi-self.z_lo
        self.surv_channels=np.arange(self.nu_lo.value,self.nu_hi.value,self.Deltanu.value)*self.Deltanu.unit
        self.r0=comoving_distance(self.z_ctr)
        self.layer_foregrounds=layer_foregrounds
        self.b_NS=b_NS
        self.b_EW=b_EW
        if array_version=="full":
            N_ant=512
            self.N_NS=N_NS_full
            self.N_EW=N_EW_full
        elif array_version=="pathfinder":
            N_ant=64
            self.N_NS=N_NS_full//2
            self.N_EW=N_EW_full//2
        else:
            raise ValueError("unknown array version")
        N_ant=N_ant
        
        # cylindrically binned survey k-modes and box considerations
        kpar_surv=kpar(self.nu_ctr,self.Deltanu,self.Nchan)
        self.kpar_surv=kpar_surv
        kparmin_surv=kpar_surv[0]
        kparmax_surv=kpar_surv[-1]
        self.kpar_surv=kpar_surv
        self.kparmin_surv=kparmin_surv
        self.kparmax_surv=kparmax_surv
        self.Nkpar_surv=len(self.kpar_surv)
        self.bmin=bmin
        self.bmax=bmax
        kperp_surv=kperp(self.nu_ctr,self.bmin,self.bmax)
        kperpmin_surv=kperp_surv[0]
        kperpmax_surv=kperp_surv[-1]
        self.kperp_surv=kperp_surv
        self.kperpmin_surv=kperpmin_surv
        self.kperpmax_surv=kperpmax_surv
        self.Nkperp_surv=len(self.kperp_surv)

        self.kmin_surv=np.sqrt(kperpmin_surv**2+kparmin_surv**2)
        self.kmax_surv=np.sqrt(kperpmax_surv**2+kparmax_surv**2)

        self.Lsurv_box_xy=twopi/kperpmin_surv
        self.Nvox_box_xy=int(self.Lsurv_box_xy*kperpmax_surv/pi)
        self.Lsurv_box_z=twopi/kparmin_surv
        self.Nvox_box_z=int(self.Lsurv_box_z*kparmax_surv/pi)
        print("beam_effects: Nxy,Nz =",self.Nvox_box_xy,self.Nvox_box_z)

        self.fgfreqs=np.asarray([self.nu_lo.value,self.nu_hi.value])*self.nu_ctr.unit

        self.N_timesteps=           def_N_timesteps
        precalculated_xy_vec=self.Lsurv_box_xy*fftshift(fftfreq(def_PA_N_grid_pix))
        N_CST_types=len(CST_f_head_syst)

        if np.all(pointing_errors==[[0.,0.,0.]]):
            N_pointing_errors=[0]
        else:
            N_pointing_errors=np.arange(0,len(pointing_errors)+1)
        N_pointing_errors_max=np.max(N_pointing_errors)
            
        already_imported_fidu_CST=Path("fidu_CST_"+str(CST_lo.value)+"_"+str(CST_hi.value)+"_"+str(CST_deltanu.value)+"_MHz.npy").is_file()
        already_imported_syst_CST=Path("syst_boxes_"+ioname+".npy").is_file()
        if heavy_beam_recalc and not already_imported_fidu_CST:
            fidu=reconfigure_CST_beam(CST_lo,CST_hi,CST_deltanu,Nxy=def_PA_N_grid_pix,
                                        beam_sim_directory=beam_sim_directory,f_head=CST_f_head_fidu,
                                        f_mid1=f_mid1,f_mid2=f_mid2,f_tail=f_tail,box_outname="fidu_box_"+ioname)
            fidu.construct_CST_box()
            print("generated fidu beam box\n")
            fidu_box=fidu.box
            CST_z_vec=np.asarray(fidu.CST_z_vec)*u.Mpc # by construction = not brittle
            np.save("fidu_CST_"+str(CST_lo.value)+"_"+str(CST_hi.value)+"_"+str(CST_deltanu.value)+"_MHz.npy",fidu_box)
            np.save("z_vec"+ioname+".npy",CST_z_vec.value)
        else:
            fidu_box=  np.load("fidu_CST_"+str(CST_lo.value)+"_"+str(CST_hi.value)+"_"+str(CST_deltanu.value)+"_MHz.npy")

            ioname_base_case=ioname.replace("N_CST_types_"+str(N_CST_types),"N_CST_types_1")
            ioname_base_case=ioname_base_case.replace("N_ptg_err_"+str(N_pointing_errors_max),"N_ptg_err_0")
            CST_z_vec=np.load("z_vec"+ioname_base_case+".npy")*u.Mpc # by construction = not brittle
        N_CST_z=len(CST_z_vec)

        syst_boxes=np.zeros((N_CST_types,def_PA_N_grid_pix,def_PA_N_grid_pix,N_CST_z)) # this needs to be 4D to be forward-compatible with the new iteration strategy in synthesize_beam
        if heavy_beam_recalc and not already_imported_syst_CST: # only import the fiducial beam once
            for i,CST_f_head_syst_i in enumerate(CST_f_head_syst):
                syst=reconfigure_CST_beam(CST_lo,CST_hi,CST_deltanu,Nxy=def_PA_N_grid_pix,
                                            beam_sim_directory=beam_sim_directory,f_head=CST_f_head_syst_i,
                                            f_mid1=f_mid1,f_mid2=f_mid2,f_tail=f_tail,box_outname="syst_box_"+ioname)
                syst.construct_CST_box()
                print("generated syst beam box\n")
                syst_boxes[i,:,:,:]=syst.box
            
            np.save("syst_boxes_"+ioname+".npy",syst_boxes)
        else:
            if N_CST_types>1:
                syst_boxes=np.load("syst_boxes_"+ioname+".npy")
            else:
                syst_boxes[0,:,:,:]=fidu_box
        
        N_CST_z=len(CST_z_vec)
        beam_domain=(precalculated_xy_vec.value,precalculated_xy_vec.value,CST_z_vec.value)
        self.beam_domain=beam_domain

        CST_syst_ensemble=np.zeros((N_CST_types,N_pointing_errors_max+1,def_PA_N_grid_pix,def_PA_N_grid_pix,N_CST_z)) # shape of CST_syst_ensemble is (N_CST_types,self.Nvox_box_xy,self.Nvox_box_xy,N_CST_z) but the sub-ensembles passed to synthesize_beam have shapes  ////////replace
        CST_syst_ensemble[:,0,:,:,:]=syst_boxes # situate the pointing error–free versions

        if type(pointing_errors[0])==float:
            pointing_errors_to_loop_over=[pointing_errors]
        elif pointing_errors is not None:
            pointing_errors_to_loop_over=pointing_errors
        else:
            pointing_errors_to_loop_over=[[0.,0.,0.]]
        for i,syst_box in enumerate(syst_boxes):
            if N_pointing_errors_max>0:
                for j,pointing_error in enumerate(pointing_errors_to_loop_over):
                    repointed=repoint_beam(beam_domain,syst_box,pointing_error)
                    CST_syst_ensemble[i,j+1,:,:,:]=repointed
        print("finished repointing beams for this complexity case")
        
        CST_freqs=np.arange(CST_lo.value,CST_hi.value,CST_deltanu.value)*CST_deltanu.unit
        if heavy_beam_recalc: # redo the beam synthesis
            fidu_synthesis=synthesize_beam(array_version=array_version,N_timesteps=self.N_timesteps,
                                                N_pbws_pert=0,nu_ctr=nu_ctr,N_grid_pix=PA_N_grid_pix,
                                                distribution="random",Npix=def_PA_N_grid_pix,
                                                sub_ensemble_of_CST_beams=fidu_box,
                                                CST_xy=precalculated_xy_vec,CST_freqs=CST_freqs,
                                                supplementary_name=ioname)
            fidu_synthesis.stack_to_box()
            print("finished synthesizing fiducial CST beam")
            fidu_box_synthesized=fidu_synthesis.box
            synthesized_xy_vec=fidu_synthesis.xy_vec
            synthesized_z_vec=fidu_synthesis.z_vec
            syst_synthesis=synthesize_beam(array_version=array_version,N_timesteps=self.N_timesteps,
                                                N_pbws_pert=N_pbws_pert,nu_ctr=nu_ctr,N_grid_pix=PA_N_grid_pix,
                                                distribution=antenna_distribution,Npix=def_PA_N_grid_pix,
                                                sub_ensemble_of_CST_beams=[fidu_box,CST_syst_ensemble],
                                                CST_xy=precalculated_xy_vec,CST_freqs=CST_freqs,
                                                supplementary_name=ioname)
            syst_synthesis.stack_to_box()
            print("finished synthesizing systematic-laden CST beam")
            syst_box_synthesized=syst_synthesis.box
            weights_synthesized=syst_synthesis.weights
            Ntypes=syst_synthesis.N_total_beam_types
            
            np.save("fidu_box_synthesized_"+ioname+".npy",fidu_box_synthesized)
            np.save("syst_box_synthesized_"+ioname+".npy",syst_box_synthesized)
            np.save("xy_vec_synthesized_"+ioname+".npy",synthesized_xy_vec.value)
            np.save("z_vec_synthesized_"+ioname+".npy",synthesized_z_vec.value)
            np.save("weights_synthesized_"+ioname+".npy",weights_synthesized)
            print("saved synthesized beam")
        else: 
            fidu_box_synthesized=np.load("fidu_box_synthesized_"+ioname+".npy")
            syst_box_synthesized=np.load("syst_box_synthesized_"+ioname+".npy")
            synthesized_xy_vec=np.load("xy_vec_synthesized_"+ioname+".npy")*u.Mpc # by construction = not brittle
            synthesized_z_vec=np.load("z_vec_synthesized_"+ioname+".npy")*u.Mpc
            weights_synthesized=np.load("weights_synthesized_"+ioname+".npy")
            print("loaded synthesized beam")
        print("finished importing/constructing synthesized CST beam")
        
        self.fi_eff_primary_box=fidu_box
        weighted_sum_syst_primary=np.zeros_like(fidu_box)
        Ntypes=len(weights_synthesized) # this is super hacky and I need to streamline it
        if Ntypes>1:
            q=0
            for i in range(N_CST_types):
                for j in range(N_pointing_errors_max+1):
                    syst_box_here=CST_syst_ensemble[i,j,:,:,:]
                    if not np.allclose(syst_box_here,0):
                        weighted_sum_syst_primary+=weights_synthesized[q]*syst_box_here
                    q+=1
            self.sy_eff_primary_box=weighted_sum_syst_primary
        else:
            self.sy_eff_primary_box=np.copy(self.fi_eff_primary_box)
        
        synthesized_pbm=(synthesized_xy_vec.value,synthesized_xy_vec.value,synthesized_z_vec.value) # might need to re-unit-ify this more robustly later, but for now the main use is interpolation and I don't want to jam up scipy by putting units where they have no business being

        self.fidu=fidu_box_synthesized
        self.real=fidu_box_synthesized
        self.thgt=syst_box_synthesized

        self.pbm_for_cs=synthesized_pbm

        # groundwork-informed forecasting considerations
        self.P_fid_for_cont_pwr=P_fid_for_cont_pwr
        self.k_idx_for_window=k_idx_for_window

        # numerical protections for assorted k-ranges
        kmin_box_and_init=(1-init_and_box_tol)*self.kmin_surv
        kmax_box_and_init=(1+init_and_box_tol)*self.kmax_surv
        kmin_CAMB=(1-CAMB_tol)*kmin_box_and_init
        kmax_CAMB=(1+CAMB_tol)*kmax_box_and_init*np.sqrt(3) # factor of sqrt(3) from pythag theorem for box to make extrapolation less likely to be necessary
        ksph,self.Ptruesph=self.get_21cm_power_spec(self.pars_set_cosmo,0.1*kmin_CAMB,10*kmax_CAMB)
        self.ksph=ksph/u.Mpc # by construction
        self.Deltabox_xy=self.Lsurv_box_xy/self.Nvox_box_xy
        self.Deltabox_z= self.Lsurv_box_z/ self.Nvox_box_z
        self.LoS_taper=LoS_taper
        self.image_taper=image_taper

        # precision control for numerical derivatives
        self.ftol_deriv=ftol_deriv
        self.eps=eps
        self.maxiter=maxiter

        self.frac_tol_conv=frac_tol_conv
        
        # considerations for printing the calculated bias results
        self.pars_forecast_names=pars_forecast_names
        assert (len(pars_forecast)==len(pars_forecast_names))

        # placeholders for forecasting-relevant matrices
        self.del_P_del_pars=np.zeros((self.N_pars_forecast,self.Nkpar_surv,self.Nkperp_surv))
        self.F=None
        self.B=None

    def get_21cm_power_spec(self,pars_use:np.ndarray,minkh:float=1e-4/u.Mpc,maxkh:float=1./u.Mpc,
                            A_HI_sq=3.55, # we definitely don't have this many sig figs, but this is a plausible enough value to use for now, taken from the first row of Table 2 of the CHIME/cosmology 2026 interpretation paper, which has a frequency conveniently pretty close to the 600 MHz sim I've been doing a bunch of tests with.  this is also their no-disclaimers best-fit value in the middle of the right-hand column of pg. 2
                            ): # get matter power spec from CAMB
        """
        same patchwork model as in CHIME 2026 (https://arxiv.org/abs/2603.25680)
        but defer FoG to resample_P_fid_on_grid
        """
        N_zs=5
        z_ctr_idx=int(np.median(np.arange(1,N_zs+1)))
        z=[self.z_ctr+i for i in np.linspace(self.z_ctr/2,-self.z_ctr/2,N_zs,endpoint=True)] # matter power interpolator does better with more redshifts
        H0,ombh2,ommh2,As,ns,_=pars_use
        omch2=ommh2-ombh2
        h=H0/100.

        pars_use_internal=camb.set_params(H0=H0.value, ombh2=ombh2.value, omch2=omch2.value, ns=ns, mnu=0.06, omk=Omegak)
        pars_use_internal.Transfer.transfer_type = camb.model.Transfer_b
        pars_use_internal.InitPower.set_params(As=As,ns=ns,r=0)
        maxkh=maxkh.to(1/u.Mpc)
        minkh=minkh.to(1/u.Mpc)
        pars_use_internal.set_matter_power(redshifts=z, kmax=maxkh.value*h.value)
        results = camb.get_results(pars_use_internal)
        hub_un=False
        matter_power_interpolator=results.get_matter_power_interpolator(nonlinear=True, hubble_units=hub_un, k_hunit=False)

        if hub_un:
            lengco_units=u.Mpc/littleh
        else:
            lengco_units=u.Mpc

        k_CAMB=np.linspace(minkh.value,maxkh.value,self.N_theory_k)/lengco_units
        P_m=matter_power_interpolator.P(self.z_ctr,k_CAMB.value)*lengco_units**3

        Hz= results.hubble_parameter(z[z_ctr_idx])*u.km/u.s/u.Mpc
        Hz=Hz.to(H0.unit)
        h=h.to(u.km/u.s/u.Mpc)
        self.h=h
        Omega_HI=4e-4*(1+self.z_ctr)**0.6 # Crichton et al. 2015 fitting function; cf. eq. 5 of the CHIME/cosmology 2026 interpretation paper
        T_b_bar=191.06*(h.value*H0/Hz*Omega_HI*(1+self.z_ctr)**2) *u.mK # cf. eq. 3 of the CHIME/cosmology 2026 interpretation paper
        bias_term_sq= A_HI_sq/(1e6*Omega_HI**2) # cf. eq. 24 of the CHIME/cosmology 2026 interpretation paper
        P_21=T_b_bar**2 *bias_term_sq *P_m # * D_FoG_HI**2

        return k_CAMB,P_21
    
    def unbin_to_Pcyl(self,pars_to_use:np.ndarray,kperp_to_use:np.ndarray=None,kpar_to_use:np.ndarray=None): # interpolate a spherically binned CAMB MPS to provide MPS values for a cylindrically binned k-grid of interest (nkpar x nkperp)
        if kperp_to_use is None:
            kperp_to_use=self.kperp_surv
        if kpar_to_use is None:
            kpar_to_use=self.kpar_surv
        k,Psph_use=self.get_21cm_power_spec(pars_to_use,minkh=0.1*self.kmin_surv,maxkh=10*self.kmax_surv)
        CAMBlength=len(Psph_use)
        k=k.reshape((CAMBlength,))
        Psph_use=Psph_use.reshape((CAMBlength,))
        k_unique, unique_idx = np.unique(k, return_index=True)
        Psph_use = Psph_use[unique_idx]
        k = k_unique

        self.Psph=Psph_use
        kperp_grid,kpar_grid=np.meshgrid(kperp_to_use,kpar_to_use, indexing="ij")
        kmag_grid=np.sqrt(kpar_grid**2+kperp_grid**2)
        Nkperp_use=len(kperp_to_use)
        Nkpar_use=len(kpar_to_use)
        Nk=Nkperp_use*Nkpar_use
        kmag_grid_flat=np.reshape(kmag_grid,(Nk,),order="C")
        sort_array=np.argsort(kmag_grid_flat)
        kmag_grid_flat_sorted=kmag_grid_flat[sort_array]

        Pcyl=np.zeros(Nk)
        interpolator=RGI((k.value,),Psph_use,
                         bounds_error=False,fill_value=None)
        Pcyl[sort_array]=interpolator(kmag_grid_flat_sorted[:, None])
        Pcyl=np.reshape(Pcyl,(Nkperp_use,Nkpar_use),order="C")*Psph_use.unit

        return kpar_grid,kperp_grid,Pcyl
    
    def get_pwr_law_FG_ingredient(self,              # synchrotron and free-free FG emission are both well described by power laws. cf. Liu & Tegmark 2011 (https://arxiv.org/abs/1106.0007)
                                  Tref,nuref,        # temp and freq to which this kind of power law FG are referenced
                                  alpha,sigma_alpha, # spectral index and its spread for this kind of power law FG (cf. Liu 2011)
                                  rngseed=438): 
        
        # generate a slice of white noise
        fg=cosmo_stats(self.Lsurv_box_xy,Lz=self.Deltabox_z,
                       P_fid=self.P_flat,k_fid=self.k_for_flat,
                       Nvox=self.Nvox_box_xy,Nvoxz=1,
                       seed=rngseed,nu_ctr=self.nu_ctr) 
        fg.generate_GRF()
        white_noise_slice=fg.T_pristine.to(Tref.unit).value
        # print("white noise slice temp units:",fg.T_pristine.unit)
        # print("np.mean(white_noise_slice), np.std(white_noise_slice) =",np.mean(white_noise_slice), np.std(white_noise_slice)) # -2.0599841277224584e-18 0.20017005259588688

        # bookkeeping to prep for power law
        freqs_in_ref_unit=self.freqs_for_fg.to(nuref.unit)   
        fg_box_this_ingredient=np.zeros((self.Nvox_box_xy,self.Nvox_box_xy,self.Nvox_box_z))
        rng=np.random.default_rng(rngseed+1)
        # print("initialized slice_of_alphas RNG with seed",rngseed)
        freq_ratios=freqs_in_ref_unit/nuref
        slice_of_alphas=rng.normal(loc=alpha, scale=sigma_alpha, size=(self.Nvox_box_xy,self.Nvox_box_xy))

        # apply LoS power law renormalization to each slice  
        # print("MANUALLY OVERRIDING FOREGROUND STATISTICS")
        for i,freq_ratio_i in enumerate(freq_ratios):
            fg_slice = Tref.value*freq_ratio_i**slice_of_alphas *white_noise_slice
            # fg_slice-=np.mean(fg_slice)
            fg_box_this_ingredient[:,:,i]=fg_slice
        # print("check last slice: Tref.value, freq_ratio_i, np.mean(slice_of_alphas) =",Tref.value, freq_ratio_i, np.mean(slice_of_alphas)) # nothing super alarming here for the synchrotron test case: 335.4 3.933333333333333 -2.7989298861610825
        # print("check last slice: np.mean(fg_slice), np.std(fg_slice) =",np.mean(fg_slice), np.std(fg_slice)) # frustrating to see that 15 orders of magnitude of near-zero have gone up in smoke: 0.004150945162675765 1.4897165599263442

        fg_box_this_ingredient*=Tref.unit
        fg_box_this_ingredient=fg_box_this_ingredient.to(u.mK)
        return fg_box_this_ingredient # centre-origin (fftshifted)

    def calc_power_contamination(self, isolated:bool=False): # Monte Carlo numerical windowing of beam-aware brightness temp boxes to yield several cylindrically power spectra of interest for forecasting and diagnostics. various states of beam knowledge and fiducial spectrum as appropriate (see Memos I-II)
        if self.P_fid_for_cont_pwr is None:
            P_cosmo=np.reshape(self.Ptruesph,(self.N_theory_k),order="C")
        elif self.P_fid_for_cont_pwr=="window": # make the fiducial power spectrum a numerical top hat
            P_cosmo=np.zeros(self.N_theory_k)
            P_cosmo[self.k_idx_for_window]=1.
            P_cosmo*=u.mK**2*u.Mpc**3 # by design, not brittle
        else:
            raise ValueError("unknown P_fid_for_cont_pwr")

        foreground_temp_unit=u.K
        N_flat=10*self.Nkpar_surv
        P_flat=np.ones(N_flat) *foreground_temp_unit**2 *self.Deltabox_xy.unit**3
        self.P_flat=P_flat
        self.k_for_flat=np.linspace(self.kparmin_surv,self.kparmax_surv,10*self.Nkpar_surv)
        if self.layer_foregrounds:
            self.freqs_for_fg= np.linspace(self.nu_hi.value,self.nu_lo.value, # descending in frequency to match the iteration over increasing redshift
                                           self.Nvox_box_z,endpoint=True)*self.Deltanu.unit
            fg_box=np.zeros((self.Nvox_box_xy,self.Nvox_box_xy,self.Nvox_box_z))*u.mK
            fg_info_cases=[ [335.4*foreground_temp_unit, 150*u.MHz, -2.8,  0.1],   # synchrotron
                            [33.5 *foreground_temp_unit, 150*u.MHz, -2.15, 0.01] ] # free-free
            # fg_info_cases=[[335.4*foreground_temp_unit, 150*u.MHz, -2.8,  0.1]]
            for fg_info in fg_info_cases:
                Tref,nuref,alpha,sigma_alpha=fg_info
                fg_box_ingredient=self.get_pwr_law_FG_ingredient(Tref,nuref,alpha,sigma_alpha)
                fg_box+=fg_box_ingredient
            self.fg_box=fg_box # centre-origin
            extreme=np.max(np.abs(fg_box.value))
            comprehensive_slice_figure(fg_box.value,
                                       norm=CenteredNorm(halfrange=extreme),
                                       cmap="RdBu",
                                       name="fg_box.png")

            fg=cosmo_stats(self.Lsurv_box_xy,Lz=self.Lsurv_box_z,
                           LoS_taper=self.LoS_taper,image_taper=self.image_taper,
                           T_pristine=fg_box)
            fg.generate_P()
            fg.bin_power()
            self.P_xx_xx_xx_fg=fg.P_binned *fg_box.unit**2 *self.Lsurv_box_xy.unit**3
            print("                           fg power calc complete")

        print("beam_effects.calc_power_contamination: self.Nvox_box_xy =",self.Nvox_box_xy)
        co_fi_xx_fg=cosmo_stats(self.Lsurv_box_xy,Lz=self.Lsurv_box_z,
                                P_fid=P_cosmo,k_fid=self.ksph, 
                                Nvox=self.Nvox_box_xy,Nvoxz=self.Nvox_box_z,
                                effective_primary_beam_for_effective_volume=self.fi_eff_primary_box, eff_pri_domain=self.beam_domain,
                                synth_beam=self.fidu,
                                frac_tol=self.frac_tol_conv,seed=self.seed,    
                                beam_domain=self.pbm_for_cs,
                                LoS_taper=self.LoS_taper,image_taper=self.image_taper,
                                wedge_cut=self.wedge_cut,nu_ctr=self.nu_ctr,fg_box=fg_box)
        self.kperpbins_internal=co_fi_xx_fg.kperpbins
        self.kparbins_internal=co_fi_xx_fg.kparbins
        co_fi_sy_fg=cosmo_stats(self.Lsurv_box_xy,Lz=self.Lsurv_box_z,
                                P_fid=P_cosmo,k_fid=self.ksph,
                                Nvox=self.Nvox_box_xy,Nvoxz=self.Nvox_box_z,
                                effective_primary_beam_for_effective_volume=self.sy_eff_primary_box, eff_pri_domain=self.beam_domain,
                                synth_beam=self.thgt,
                                frac_tol=self.frac_tol_conv,seed=self.seed,
                                beam_domain=self.pbm_for_cs,
                                LoS_taper=self.LoS_taper,image_taper=self.image_taper,
                                wedge_cut=self.wedge_cut,nu_ctr=self.nu_ctr,fg_box=fg_box)
        xx_fi_sy_fg=cosmo_stats(self.Lsurv_box_xy,Lz=self.Lsurv_box_z,
                                Nvox=self.Nvox_box_xy,Nvoxz=self.Nvox_box_z,
                                effective_primary_beam_for_effective_volume=self.sy_eff_primary_box, eff_pri_domain=self.beam_domain,
                                T_pristine=fg_box,
                                synth_beam=self.thgt,
                                frac_tol=self.frac_tol_conv,seed=self.seed,
                                beam_domain=self.pbm_for_cs,
                                LoS_taper=self.LoS_taper,image_taper=self.image_taper,
                                wedge_cut=self.wedge_cut,nu_ctr=self.nu_ctr,fg_box=fg_box)
        xx_fi_xx_fg=cosmo_stats(self.Lsurv_box_xy,Lz=self.Lsurv_box_z,
                                Nvox=self.Nvox_box_xy,Nvoxz=self.Nvox_box_z,
                                effective_primary_beam_for_effective_volume=self.fi_eff_primary_box, eff_pri_domain=self.beam_domain,
                                T_pristine=fg_box,
                                synth_beam=self.fidu,
                                frac_tol=self.frac_tol_conv,seed=self.seed,
                                beam_domain=self.pbm_for_cs,
                                LoS_taper=self.LoS_taper,image_taper=self.image_taper,
                                wedge_cut=self.wedge_cut,nu_ctr=self.nu_ctr)
        co_fi_xx_xx=cosmo_stats(self.Lsurv_box_xy,Lz=self.Lsurv_box_z,
                                P_fid=P_cosmo,k_fid=self.ksph, 
                                Nvox=self.Nvox_box_xy,Nvoxz=self.Nvox_box_z,
                                effective_primary_beam_for_effective_volume=self.fi_eff_primary_box, eff_pri_domain=self.beam_domain,
                                synth_beam=self.fidu,
                                frac_tol=self.frac_tol_conv,seed=self.seed,    
                                beam_domain=self.pbm_for_cs,
                                LoS_taper=self.LoS_taper,image_taper=self.image_taper,
                                wedge_cut=self.wedge_cut,nu_ctr=self.nu_ctr,fg_box=None)
        co_fi_sy_xx=cosmo_stats(self.Lsurv_box_xy,Lz=self.Lsurv_box_z,
                                P_fid=P_cosmo,k_fid=self.ksph, 
                                Nvox=self.Nvox_box_xy,Nvoxz=self.Nvox_box_z,
                                effective_primary_beam_for_effective_volume=self.sy_eff_primary_box, eff_pri_domain=self.beam_domain,
                                synth_beam=self.thgt,
                                frac_tol=self.frac_tol_conv,seed=self.seed,    
                                beam_domain=self.pbm_for_cs,
                                LoS_taper=self.LoS_taper,image_taper=self.image_taper,
                                wedge_cut=self.wedge_cut,nu_ctr=self.nu_ctr,fg_box=None)
        co_xx_xx_fg=cosmo_stats(self.Lsurv_box_xy,Lz=self.Lsurv_box_z,
                                P_fid=P_cosmo,k_fid=self.ksph, 
                                Nvox=self.Nvox_box_xy,Nvoxz=self.Nvox_box_z,
                                frac_tol=self.frac_tol_conv,seed=self.seed,    
                                LoS_taper=self.LoS_taper,image_taper=self.image_taper,
                                wedge_cut=self.wedge_cut,nu_ctr=self.nu_ctr,fg_box=fg_box)

        recalc_co_fi_xx_fg=False
        recalc_co_fi_sy_fg=False
        recalc_xx_fi_sy_fg=False
        recalc_xx_fi_xx_fg=False
        recalc_co_fi_xx_xx=False
        recalc_co_fi_sy_xx=False
        recalc_co_xx_xx_fg=False
        if isolated==False:
            recalc_co_fi_xx_fg=True
            recalc_co_fi_sy_fg=True
            recalc_xx_fi_sy_fg=True
            recalc_xx_fi_xx_fg=True
            recalc_co_fi_xx_xx=True
            recalc_co_fi_sy_xx=True
            recalc_co_xx_xx_fg=True
        
        elif isolated=="co_fi_xx_fg":
            recalc_co_fi_xx_fg=True
        elif isolated=="co_fi_sy_fg":
            recalc_co_fi_sy_fg=True
        elif isolated=="xx_fi_sy_fg":
            recalc_xx_fi_sy_fg=True
        elif isolated=="xx_fi_xx_fg":
            recalc_xx_fi_xx_fg=True
        elif isolated=="co_fi_xx_xx":
            recalc_co_fi_xx_xx=True
        elif isolated=="co_fi_sy_xx":
            recalc_co_fi_sy_xx=True
        elif isolated=="co_xx_xx_fg":
            recalc_co_xx_xx_fg=True

        if recalc_co_fi_xx_fg:
            co_fi_xx_fg.power_Monte_Carlo(interfix="fi")
            self.N_per_realization= co_fi_xx_fg.N_per_realization
            self.P_co_fi_xx_fg=     co_fi_xx_fg.P_binned_MC_complete
            self.kperp_for_cosmo=  co_fi_xx_fg.kperpbins
            self.kpar_for_cosmo=   co_fi_xx_fg.kparbins
            print("cosmo + fidu beam +        fg MC         complete")
        if recalc_co_fi_sy_fg:
            co_fi_sy_fg.power_Monte_Carlo(interfix="co_fi_sy_fg")
            if not recalc_co_fi_xx_fg:
                self.N_per_realization= co_fi_sy_fg.N_per_realization
                self.kperp_for_cosmo=  co_fi_sy_fg.kperpbins
                self.kpar_for_cosmo=   co_fi_sy_fg.kparbins
            self.P_co_fi_sy_fg=         co_fi_sy_fg.P_binned_MC_complete
            print("cosmo + fidu beam + syst + fg MC         complete")
        if recalc_xx_fi_sy_fg:
            xx_fi_sy_fg.generate_P() # lack of interfix is not a problem because this is a quick calculation so there was never a need for hourly saves
            xx_fi_sy_fg.bin_power()
            if not recalc_co_fi_xx_fg:
                self.N_per_realization= xx_fi_sy_fg.N_per_realization
                self.kperp_for_cosmo=  xx_fi_sy_fg.kperpbins
                self.kpar_for_cosmo=   xx_fi_sy_fg.kparbins
            self.P_xx_fi_sy_fg=         xx_fi_sy_fg.P_binned *xx_fi_sy_fg.power_unit
            print("        fidu beam + syst + fg power calc complete")
        if recalc_xx_fi_xx_fg:
            xx_fi_xx_fg.generate_P()
            xx_fi_xx_fg.bin_power()
            if not recalc_co_fi_xx_fg:
                self.N_per_realization= xx_fi_xx_fg.N_per_realization
                self.kperp_for_cosmo=  xx_fi_xx_fg.kperpbins
                self.kpar_for_cosmo=   xx_fi_xx_fg.kparbins
            self.P_xx_fi_xx_fg=         xx_fi_xx_fg.P_binned *xx_fi_xx_fg.power_unit
            print("        fidu beam +      + fg power calc complete")
        if recalc_co_fi_xx_xx:
            co_fi_xx_xx.power_Monte_Carlo(interfix="co_fi_xx_xx")
            if not recalc_co_fi_xx_fg:
                self.N_per_realization= co_fi_xx_xx.N_per_realization
                self.kperp_for_cosmo=  co_fi_xx_xx.kperpbins
                self.kpar_for_cosmo=   co_fi_xx_xx.kparbins
            self.P_co_fi_xx_xx=         co_fi_xx_xx.P_binned_MC_complete
            print("cosmo + fidu beam             MC         complete")
        if recalc_co_fi_sy_xx:
            co_fi_sy_xx.power_Monte_Carlo(interfix="co_fi_sy_xx")
            if not recalc_co_fi_xx_fg:
                self.N_per_realization= co_fi_sy_xx.N_per_realization
                self.kperp_for_cosmo=  co_fi_sy_xx.kperpbins
                self.kpar_for_cosmo=   co_fi_sy_xx.kparbins
            self.P_co_fi_sy_xx=         co_fi_sy_xx.P_binned_MC_complete
            print("cosmo + fidu beam + syst      MC         complete")
        if recalc_co_xx_xx_fg:
            co_xx_xx_fg.power_Monte_Carlo(interfix="co_xx_xx_fg")
            if not recalc_co_fi_xx_fg:
                self.N_per_realization= co_xx_xx_fg.N_per_realization
                self.kperp_for_cosmo=  co_xx_xx_fg.kperpbins
                self.kpar_for_cosmo=   co_xx_xx_fg.kparbins
            self.P_co_xx_xx_fg= co_xx_xx_fg.P_binned_MC_complete
            print("cosmo +                    fg MC         complete")
        COSMOTEST=cosmo_stats(self.Lsurv_box_xy,Lz=self.Lsurv_box_z,
                              P_fid=P_cosmo,k_fid=self.ksph, 
                              Nvox=self.Nvox_box_xy,Nvoxz=self.Nvox_box_z,
                              LoS_taper=self.LoS_taper,image_taper=self.image_taper,
                              frac_tol=self.frac_tol_conv,seed=self.seed,nu_ctr=self.nu_ctr)
        COSMOTEST.power_Monte_Carlo(interfix="CO_XX_XX_XX_") # extra underscore is because numpy is fine with case-sensitive file names but MacOS is not :(
        self.P_CO_XX_XX_XX=COSMOTEST.P_binned_MC_complete
        print("COSMO                         MC         COMPLETE")

        _,_,P_co_xx_xx_xx=self.unbin_to_Pcyl(self.pars_set_cosmo, 
                                             kperp_to_use=self.kperp_for_cosmo[:-1]+0.5*(self.kperp_for_cosmo[1]-self.kperp_for_cosmo[0]), 
                                             kpar_to_use=self.kpar_for_cosmo[:-1]+0.5*(self.kpar_for_cosmo[1]-self.kpar_for_cosmo[0]))
        self.P_co_xx_xx_xx=P_co_xx_xx_xx
        # print("np.mean(COSMOTEST.T_pristine), np.std(COSMOTEST.T_pristine) =",np.mean(COSMOTEST.T_pristine),np.std(COSMOTEST.T_pristine))

        if isolated==False:
            self.Pcont_cyl=self.P_co_fi_sy_fg-self.P_co_fi_xx_fg

    def cyl_partial(self,n:int): # cylindrically binned matter power spectrum partial WRT one cosmo parameter
        dparn=self.dpar[n]
        pcopy=self.pars_set_cosmo.copy()
        pndispersed=pcopy[n]+np.linspace(-2,2,5)*dparn

        _,_,Pcyl=self.unbin_to_Pcyl(pcopy)
        P0=np.mean(np.abs(Pcyl))+self.eps
        tol=self.ftol_deriv*P0 # generalizes tol=ftol*f0 from PHYS512

        pcopy[n]=pcopy[n]+2*dparn 
        _,_,Pcyl_2plus=self.unbin_to_Pcyl(pcopy)
        pcopy=self.pars_set_cosmo.copy()
        pcopy[n]=pcopy[n]-2*dparn
        _,_,Pcyl_2minu=self.unbin_to_Pcyl(pcopy)
        deriv1=(Pcyl_2plus-Pcyl_2minu)/(4*self.dpar[n])

        pcopy=self.pars_set_cosmo.copy()
        pcopy[n]=pcopy[n]+dparn
        _,_,Pcyl_plus=self.unbin_to_Pcyl(pcopy)
        pcopy=self.pars_set_cosmo.copy()
        pcopy[n]=pcopy[n]-dparn
        _,_,Pcyl_minu=self.unbin_to_Pcyl(pcopy)
        deriv2=(Pcyl_plus-Pcyl_minu)/(2*self.dpar[n])

        Pcyl_dif=Pcyl_plus-Pcyl_minu
        if (np.mean(Pcyl_dif)<tol): # might be too strict or loose a condition
            estimate=(4*deriv2-deriv1)/3
            self.iter=0 # reset for next time
            self.del_P_del_pars[n,:,:]=estimate
        else:
            pnmean=np.mean(np.abs(pndispersed)) # the np.abs part should be redundant because, by this point, all the k-mode values and their corresponding dpns and Ps should be nonnegative, but anyway... numerical stability or something idk
            Psecond=np.abs(np.mean(2*self.Pcyl-Pcyl_minu-Pcyl_plus))/self.dpar[n]**2 # an estimate!! break out of the vicious cycle of not having enough info
            dparn=np.sqrt(self.eps*pnmean*P0/Psecond)
            self.dpar[n]=dparn # send along knowledge of the updated step size
            self.iter+=1
            self.cyl_partial(n) # recurse
            if self.iter==self.maxiter:
                print("failed to converge in {:d} iterations".format(self.maxiter))
                fallback=(4*deriv2-deriv1)/3
                print("RETURNING fallback")
                self.iter=0 # still need to reset for next time
                self.del_P_del_pars[n,:,:]=fallback

    def compute_del_P_del_pars(self): # builds a (N_pars_forecast,Nkperp,Nkpar) array of the partials of the cylindrically binned MPS WRT each cosmo param in the forecast
        for n in range(self.N_pars_set_cosmo):
            self.iter=0 # b/c starting a new partial deriv calc.
            self.cyl_partial(n)

    def compute_noise(self):
        assert self.N_per_realization is not None, "try calling the compute_noise() method again after running calc_power_contamination()"
        self.sample_variance=np.sqrt(2/self.N_per_realization)*self.P_co_fi_sy_fg # rescale according to the number of realizations 

        sen=CHORD_sense(spacing=[self.b_EW,self.b_NS], n_side=[self.N_EW,self.N_NS], orientation=def_offset, center=None, dish_diameter=D, # array layout
                        freq_cen=self.nu_ctr, integration_time=integration_s*u.s, time_per_day=hrs_per_night, n_days=100, bandwidth=self.bw, # obs config
                        Trcv=35*u.K, latitude=DRAO_lat, tsky_ref_freq=400.*u.MHz, tsky_amplitude=25*u.K, # what's going on with the sky?
                        coherent=False, horizon_buffer=0.1*littleh/u.Mpc, foreground_model="optimistic") # processing details
        sen.sense2d()
        kperp_from_21cmSense=sen.sense2d_kperp
        kpar_from_21cmSense=sen.sense2d_kpar
        thnoise_21cmSense=sen.sense2d_P
        kperp_surv_grid,kpar_surv_grid=np.meshgrid(self.kperp_surv,self.kpar_surv, indexing="ij")
        thnoise_surv=RGI((kperp_from_21cmSense.value,kpar_from_21cmSense.value),thnoise_21cmSense,
                          bounds_error=False,fill_value=None)(np.array([kperp_surv_grid.value,kpar_surv_grid.value]).T).T
        self.thermal_noise=thnoise_surv
        self.all_sigmasuncs=self.thermal_noise+self.sample_variance # ensemble stats + 21cmSense

    def compute_F(self):
        if np.all(self.del_P_del_pars==0):
            self.compute_del_P_del_pars()
        if self.uncs is None:
            self.compute_noise()

        V=0.*self.del_P_del_pars
        for i in range(self.N_pars_forecast):
            V[i,:,:]=self.del_P_del_pars[i,:,:]/self.uncs # elementwise division for an nkpar x nkperp slice
        self.V=V
        V_completely_transposed=np.transpose(V,axes=(2,1,0))
        self.V_completely_transposed=V_completely_transposed
        self.F=np.einsum("ijk,kjl->il",V,V_completely_transposed)
        print("computed F")

    def compute_B(self):
        if self.del_P_del_pars is None:
            self.compute_del_P_del_pars()
        if self.uncs is None:
            self.compute_noise()
        
        self.Pcont_div_sigma=self.Pcont_cyl/self.uncs
        self.B=np.einsum("jk,ijk->i",self.Pcont_div_sigma,self.V)
        print("computed B")  
        
    def bias(self): # collect the ingredients of the parameter bias calculation
        self.compute_del_P_del_pars()
        print("built partials")
        self.calc_power_contamination()
        print("computed Pcont")

        self.compute_noise()
        print("computed uncertainties at each k-mode")

        if self.F is None:
            self.compute_F()
        if self.B is None:
            self.compute_B()
        self.biases=(np.linalg.inv(self.F)@self.B).reshape((self.N_pars_forecast,))
        print("computed b")

    def forecast_corner_plot(self,N_Fisher_samples:int=10000):
        if self.F is None:
            self.compute_F()

        C=np.linalg.inv(self.F)
        if np.any(C==np.nan):
            C=np.linalg.pinv(self.F)
        rng=np.random.default_rng()
        samples=rng.multivariate_normal(np.zeros(self.N_pars_forecast),C,size=N_Fisher_samples)
        pygtc.plotGTC(chains=samples, 
                      paramNames=self.pars_forecast_names,
                      truths=self.pars_forecast,
                      plot_name="forecast_corner_plot.png")

    def print_results(self):
        print("\n\nbias calculation results for the survey described above.................................")
        print("........................................................................................")
        for p,par in enumerate(self.pars_forecast):
            print('{:12} = {:-10.3e} with bias {:-12.5e} (fraction = {:-10.3e})'.format(self.pars_forecast_names[p], par, self.biases[p], self.biases[p]/par))
        return None
####################################################################################################################################################################################################################################

def repoint_beam(domain,beam,rot_angles=[0.,0.,0.,]):
    rot_x,rot_y,rot_z=np.asarray(rot_angles)*np.pi/180.*u.rad
    RX=np.asarray([[np.cos(rot_x),-np.sin(rot_x), 0.],
                   [np.sin(rot_x), np.cos(rot_x), 0.],
                   [0.,            0.,            1.]])
    RY=np.asarray([[ np.cos(rot_y),  0., np.sin(rot_y)],
                   [ 0.,             1., 0.],
                   [-np.sin(rot_y),  0., np.cos(rot_y)]])
    RZ=np.asarray([[1., 0.,             0.,],
                   [0., np.cos(rot_z), -np.sin(rot_z)],
                   [0., np.sin(rot_z),  np.cos(rot_z)]])
    R=RX@RY@RZ
    xvec,yvec,zvec=domain
    nx=len(xvec)
    ny=len(yvec)
    nz=len(zvec)
    N=nx*ny*nz
    x_grid,y_grid,z_grid=np.meshgrid(xvec,yvec,zvec, indexing="ij")
    x_flat=np.reshape(x_grid,(N,),order="C")
    y_flat=np.reshape(y_grid,(N,),order="C")
    z_flat=np.reshape(z_grid,(N,),order="C")
    xyz_flat=np.asarray([x_flat,y_flat,z_flat]).T # 3xN

    # philosophy here: need 3xN for R@ compatibility, but can't just use R@xyz_flat because RGI needs something with shape ((nx,),(ny,),(nz,)), not (3,N)
    x_prime_vec,_,_=R@[x_grid[:,0,0],y_grid[:,0,0],z_grid[:,0,0]] # this is probably going to take some reslicing, re-transposing, and reassembling
    _,y_prime_vec,_=R@[x_grid[0,:,0],y_grid[0,:,0],z_grid[0,:,0]]
    _,_,z_prime_vec=R@[x_grid[0,0,:],y_grid[0,0,:],z_grid[0,0,:]]

    interpolator=RGI((x_prime_vec,y_prime_vec,z_prime_vec),beam,
                     bounds_error=False,fill_value=None)
    rotated_beam_sampled_at_original_domain=interpolator(xyz_flat)
    unflattened_output=np.reshape(rotated_beam_sampled_at_original_domain,beam.shape,order="C")
    renormalized_rotated=unflattened_output/np.max(unflattened_output)
    renormalized_rotated[unflattened_output<0.]=0. # too hacky for real science

    return renormalized_rotated

"""
this class helps connect ensemble-averaged power spectrum estimates and 
cosmological brighness temperature boxes for assorted interconnected use cases:
1. generate a power spectrum that describes the statistics of a cosmo box (temp field)
2. generate GRF realizations of a cosmo box consistent with a chosen power spectrum
3. Monte Carlo effective windowing of a power spectrum by a beam
4. interpolate a power spectrum (sph, cyl, or sph->grid)
"""

class cosmo_stats(object):
    def __init__(self,
                 Lxy:float=600.*u.Mpc,Lz:float=None,                                    # physical box length (Mpc). one scaling is nonnegotiable for box->spec and spec->box calcs; the other would be useful for rectangular prism box considerations (sky plane slice is square, but LoS extent can differ)
                 T_pristine:np.ndarray=None,T_beam:np.ndarray=None,                     # brightness temperature box realizations without ("_pristine") or with ("_beam") the beam applied (primary would be multiplied, but now the vanguard PA-CST approach uses convolution)
                 P_fid:np.ndarray=None,                                                 # power spectrum you want to window. probably comes from cosmo (like CAMB) or is flat (for a reference calculation)
                 k_fid:np.ndarray=None,                                                 # Fourier space points where the fiducial power spectrum is sampled
                 Nvox:int=None,Nvoxz:int=None,                                          # number of voxels in the x/y or z directions
                 synth_beam:np.ndarray=None,                                            # version of the beam (box of values evaluated in config space)
                 effective_primary_beam_for_effective_volume=None, eff_pri_domain=None,
                 Nkperp:int=0,Nkpar:int=0,                                              # number of k-bins in the sky plane and line of sight directions
                 binning_mode:str="lin",                                                # bin linearly or logarithmically
                 bin_each_realization:bool=False,                                       # bin each realization of the Monte Carlo? (with the current implementation there's no typical use case where this would be necessary, but the option is there)
                 frac_tol:float=0.1,                                                    # fractional tolerance in cosmic variance of the Monte Carlo ensemble -> used to calculate the number of realizations
                 kperpbins_interp:np.ndarray=None,kparbins_interp:np.ndarray=None,      # bins where you want to know about the power spectrum (if you're interested in interpolating to some binning scheme other than what you get from chopping up the box)
                 P_MC_complete:np.ndarray=None,                                         # converged Monte Carlo power spectrum
                 avoid_extrapolation:bool=False,                                        # whether or not to avoid extrapolation
                 seed=None,                                                             # Monte Carlo realization logistics: whether or not to subtract the monopole moment when you generate boxes (the option is mostly there if you're interested in off-label uses of this code to compute power spectra from fields that are not cosmological overdensity fields); RNG seed for predictable ensemble behaviour
                 beam_domain:np.ndarray=None,                                           # when using a discretely sampled beam not sampled internally using a callable, it is necessary to provide knowledge of the domain at which it was sampled
                 LoS_taper=False,image_taper=False,                                     # apodize along the sky plane or line-of-sight directions to suppress ringing originating from features that cut off sharply?
                 wedge_cut:bool=False,nu_ctr=None,                                      # throw away info from k-modes inside the foreground wedge?; when using synchrotron foregrounds AND performing a wedge cut, the calling routine should specify the central frequency of the survey in question to have a physical anchor for the foregrounds. also need central freq for FoG
                 fg_box:np.ndarray=None):                                               # foregrounds to add to the signal-of-interest map (T)
        
        # spectrum and box
        if (Lz is None): # cubic box
            self.Lz=Lxy
            self.Lxy=Lxy
        else:            # rectangular prism box
            self.Lz=Lz
            self.Lxy=Lxy
        physical_volume=self.Lxy**2*self.Lz
        self.physical_volume=physical_volume
        self.fg_box=fg_box
        self.P_fid=P_fid
        self.compute_FoG = P_fid is not None
        # print("self.compute_FoG=",self.compute_FoG)
        if self.compute_FoG:
            assert nu_ctr is not None, "centre freq is required to compute FoG"
            z_ctr=nu_HI_z0/nu_ctr-1
            self.z_ctr=z_ctr
            redshift_factor=np.sqrt( Omegar/(1+z_ctr)**4 
                                    +Omegam/(1+z_ctr)**3
                                    +Omegak/(1+z_ctr)**2
                                    +OmegaLambda)
            self.h=h*redshift_factor
        self.P_fid_box=None
        self.T_beam=T_beam
        self.T_pristine=T_pristine
        if ((T_beam is None) and (T_pristine is None) and (P_fid is None) and (synth_beam is None)): # require either a box or a fiducial power spec (il faut some way of determining #voxels/side; passing just Nvox is not good enough)
            raise ValueError("not enough info")
        else:                                                                  # there is possibly enough info to proceed, but still need to check for conflicts and gaps
            if ((T_pristine is not None) and (T_beam is not None)):
                print("WARNING: T_pristine and T_beam both passed; T_beam will be temporarily ignored and then internally overwritten to ensure consistency with beam")
                if (T_pristine.shape!=T_beam.shape):
                    raise ValueError("conflicting info")
                else:                                                          # use box shape to set cubic/ rectangular prism box attributes
                    Nvox, _, Nvoxz=T_beam.shape
            if ((Nvox is not None) and (T_pristine is not None)):              # possible conflict: if both Nvox and a box are passed, 
                T_pristine_shape0,_,T_pristine_shape2=T_pristine.shape
                if (Nvox!=T_pristine.shape[0]):                                # but Nvox and the box shape disagree,
                    raise ValueError("conflicting info")                       # estamos en problemas
                else:
                    Nvox= T_pristine_shape0                               # otherwise, initialize the Nvox attributes
                    Nvoxz=T_pristine_shape2
            elif (Nvox is not None and Nvoxz is None):                                           # if Nvox was passed but T was not, use Nvox to initialize the Nvox attributes
                Nvoxz=np.copy(Nvox)
            elif T_pristine is not None:                                                              # remaining case: T was passed but Nvox was not, so use the shape of T to initialize the Nvox attributes
                Nvox, _, Nvoxz = T_pristine.shape
            self.Nvox=Nvox
            self.Nvoxz=Nvoxz

            if (P_fid is not None): # no hi fa res si the fiducial power spectrum has a different dimensionality or bin width than the realizations you plan to generate (boxes will be generated from a grid-interpolated P_fid, anyway)
                Pfidshape=P_fid.shape
                Pfiddims=len(Pfidshape)
                if (Pfiddims==2):
                    if synth_beam is None: # trying to do a minimalistic instantiation where I merely provide a fiducial power spectrum and interpolate it
                        self.fid_Nkperp,self.fid_Nkpar=Pfidshape
                    else:
                        try: # see if the power spec is a CAMB-esque (1,npts) array
                            self.P_fid=np.reshape(P_fid,(Pfidshape[-1],),order="C") # make the CAMB MPS shape amenable to the calcs internal to this class
                        except: # barring that...
                            pass # treat the power spectrum as being truly cylindrically binned
                elif (Pfiddims==1):
                    self.fid_Nkperp=Pfidshape[0] # already checked that P_fid is 1d, so no info is lost by extracting the int in this one-element tuple, and fid_Nkperp being an integer makes things work the way they should down the line
                    self.fid_Nkpar=0
                else:
                    raise ValueError("unsupported binning mode")
                
        # units! branching is safe now that eval reaching here has been confirmed to not be missing necessary info
        self.length_unit=  Lxy.unit
        if T_pristine is not None:
            self.temp_unit=T_pristine.unit
        else:
            self.temp_unit=(self.P_fid.unit/self.length_unit**3)**0.5
        self.power_unit= self.temp_unit**2 *self.length_unit**3
        
        # config space
        self.box_shape=(self.Nvox,self.Nvox,self.Nvoxz) if self.Nvoxz>1 else (self.Nvox,self.Nvox)
        self.Deltaxy=self.Lxy/self.Nvox                           # sky plane: voxel side length
        self.xy_vec_for_box=self.Lxy*fftshift(fftfreq(self.Nvox)) # sky plane Cartesian config space coordinate axis
        self.Deltaz= self.Lz/self.Nvoxz                           # line of sight voxel side length
        self.z_vec_for_box= self.Lz*fftshift(fftfreq(self.Nvoxz)) # line of sight Cartesian config space coordinate axis
        self.d3r=self.Deltaz*self.Deltaxy**2                      # volume element = voxel volume

        self.xx_grid,self.yy_grid,self.zz_grid=np.meshgrid(self.xy_vec_for_box,
                                                           self.xy_vec_for_box,
                                                           self.z_vec_for_box, indexing="ij")      # box-shaped Cartesian coords CENTRE-ORIGIN
        self.to_eval_at=np.array([self.xx_grid.value,self.yy_grid.value,self.zz_grid.value]).T

        # Fourier space
        self.Deltakxy=twopi/self.Lxy                                        # voxel side length
        self.Deltakz= twopi/self.Lz
        d3k=self.Deltakxy**2*self.Deltakz                              # volume element / voxel volume
        self.kxy_vec_for_box_corner=twopi*fftfreq(self.Nvox,d=self.Deltaxy) # one Cartesian coordinate axis - non-fftshifted/ corner origin
        self.kz_vec_for_box_corner= twopi*fftfreq(self.Nvoxz,d=self.Deltaz)
        self.kx_grid_corner,self.ky_grid_corner,self.kz_grid_corner=np.meshgrid(self.kxy_vec_for_box_corner,
                                                                                self.kxy_vec_for_box_corner,
                                                                                self.kz_vec_for_box_corner, indexing="ij")               # box-shaped Cartesian coords
        self.kmag_grid_corner= np.sqrt(self.kx_grid_corner**2+self.ky_grid_corner**2+self.kz_grid_corner**2) # k magnitudes for each voxel (need for the box generation direction)
        self.kmag_grid_centre=fftshift(self.kmag_grid_corner)
        self.kmag_grid_centre_flat=np.reshape(self.kmag_grid_centre,(self.Nvox**2*self.Nvoxz),order="C")
        self.kmag_grid_corner_flat=np.reshape(self.kmag_grid_corner,(self.Nvox**2*self.Nvoxz,),order="C")
        self.kmag_grid_for_comparison= self.kmag_grid_corner if self.Nvoxz>1 else self.kmag_grid_corner[:,:,0]
              
        self.kpar_column_centre= np.abs(fftshift(self.kz_vec_for_box_corner))                                      # magnitudes of kpar for a representative column along the line of sight (z-like)
        self.kperp_slice_centre= np.sqrt(fftshift(self.kx_grid_corner)**2+fftshift(self.ky_grid_corner)**2)[:,:,0] # magnitudes of kperp for a representative slice transverse to the line of sight (x- and y-like)
        kperpgrid3,kpargrid3=np.meshgrid(self.kperp_slice_centre,self.kpar_column_centre, indexing="ij")
        self.kperpgrid3_flat=np.reshape(kperpgrid3,(self.Nvox**2*self.Nvoxz,),order="C")
        self.kpargrid3_flat= np.reshape(kpargrid3, (self.Nvox**2*self.Nvoxz,),order="C")

        if self.Nvoxz>1:
            self.transform_axes=(0,1,2)
            self.d3k=d3k
            self.iftnorm=twopi**3
        else:
            self.transform_axes=(0,1)
            self.d3k=self.Deltakxy**2
            self.iftnorm=twopi**2
        
        # foreground groundwork
        self.wedge_cut=wedge_cut
        if wedge_cut:
            assert(nu_ctr is not None), "an arbitrary box <-> power spectrum translation doesn't require frequency\n"+\
                                        "info. But, when you opt into the wedge cut, you must override the None\n"+\
                                        "default in the nu_ctr keyword."
            self.kperp_corner=np.sqrt(self.kx_grid_corner**2+self.ky_grid_corner**2)
            wedge_kpar_threshold_corner=wedge_kpar(nu_ctr,self.kperp_corner)
            self.voxels_in_wedge_corner=self.kz_grid_corner<=wedge_kpar_threshold_corner
        
        # rng management
        self.rng=np.random.default_rng(seed)
        # print("initialized cosmo_stats RNG with seed",seed)

        # if P_fid was passed, establish its values on the k grid (helpful when generating a box)
        self.k_fid=k_fid
        self.avoid_extrapolation=avoid_extrapolation
        if (self.P_fid is not None and self.k_fid is not None):
            if (len(self.P_fid.shape)==1): # truly 1d fiducial power spec (by this point, even CAMB-like shapes have been reshuffled)
                self.resample_P_fid_on_grid(self.compute_FoG)
            else:
                assert 1==0, "not yet implemented"
        
        # binning considerations
        self.bin_each_realization=bin_each_realization
        self.binning_mode=binning_mode

        bin_denom=2.2
        if Nkperp==0:
            Nkperp=int(Nvox/bin_denom)
        if Nkpar==0:
            Nkpar=int(Nvoxz/bin_denom)
        self.Nkperp=Nkperp # the number of bins to put in power spec realizations you construct
        self.Nkpar=Nkpar
        self.kmax_box_xy= pi/self.Deltaxy
        self.kmax_box_z=  pi/self.Deltaz
        self.kmin_box_xy= twopi/self.Lxy
        self.kmin_box_z=  twopi/self.Lz
        
        # voxel grids for cyl binning
        if (self.Nkpar is not None and self.Nkpar!=0):
            kperpbins=np.linspace(0,self.kmax_box_xy, self.Nkperp+1)
            bw=kperpbins[1]-kperpbins[0]
            self.kperpbins=kperpbins +0.5*bw
            
            kparbins=np.linspace(0,self.kmax_box_z-self.kmin_box_z, self.Nkpar+1)
            bw=kparbins[1]-kparbins[0]
            self.kparbins=kparbins +0.5*bw
            
            self.kperpbins_grid,self.kparbins_grid=np.meshgrid(self.kperpbins,self.kparbins, indexing="ij")
        
            self.bins_to_use=[self.kperpbins.value,self.kparbins.value]
            self.coords_to_use=[self.kperpgrid3_flat.value,self.kpargrid3_flat.value]

        else: # calling them perp bins for class reasons but they are just sph
            kmax_box=np.max([self.kmax_box_xy.value,self.kmax_box_z.value])/self.length_unit # ignore the voxels outside the sphere that is contained by the box's larger axis but probably exceeds its smaller axis
            
            kperpbins=np.linspace(0,kmax_box, self.Nkperp+1)
            bw=kperpbins[1]-kperpbins[0]
            self.kperpbins=kperpbins +0.5*bw

            self.kparbins=None
            self.Nkpar=0

            self.bins_to_use=[self.kperpbins.value]
            self.coords_to_use=self.kmag_grid_centre_flat.value
            
        # tapering/apodization
        taper_xy=np.ones(self.Nvox)
        taper_z=np.ones(self.Nvoxz)
        fftshift_axes=()
        if image_taper:
            taper_xy=Blackman_Harris_safe_for_FFT(Nvox)
            fftshift_axes=(0,1)
        if LoS_taper:
            taper_z= Blackman_Harris_safe_for_FFT(Nvoxz) # confirmed to be centre-, not corner-origin
            fftshift_axes+=(2,)
        if self.Nvoxz>1:
            taper_xxx,taper_yyy,taper_zzz=np.meshgrid(taper_xy,taper_xy,taper_z, indexing="ij")
            taper_xyz_product=taper_xxx*taper_yyy*taper_zzz
        else:
            taper_xx,taper_yy=np.meshgrid(taper_xy,taper_xy,indexing="ij")
            taper_xyz_product=taper_xx*taper_yy
        self.taper_xyz_centre=taper_xyz_product
        self.taper_xyz_corner=ifftshift(taper_xyz_product,axes=fftshift_axes)

        # beam
        evaled_num=None
        if effective_primary_beam_for_effective_volume is None:
            if synth_beam is not None:
                raise ValueError("not enough info")
            else:
                self.effective_volume=np.sum(self.taper_xyz_centre**2*self.d3r)
        else:
            interpolator=RGI((eff_pri_domain),effective_primary_beam_for_effective_volume,
                             bounds_error=avoid_extrapolation,fill_value=None)
            eff_pri_this_domain=interpolator(self.to_eval_at).T # the constituent self.ii_grid are centre-origin, as intended
            comprehensive_slice_figure(effective_primary_beam_for_effective_volume,
                                       norm=LogNorm(vmax=1),
                                       name="effective_primary.png")
            comprehensive_slice_figure(eff_pri_this_domain,
                                       norm=LogNorm(vmax=1),
                                       name="effective_primary_interpolated.png")
            self.effective_volume=np.sum((eff_pri_this_domain*self.taper_xyz_centre)**2*self.d3r)
        self.synth_beam=synth_beam
        self.beam_domain=beam_domain
        if (self.synth_beam is not None): # non-identity FIDUCIAL beam
            try:    # to access this branch, the numerically sampled beam needs to be close enough to a numpy array that it has a shape and not, e.g. a callable
                synth_beam.shape
            except: # beam is a callable (or something else without a shape method), which is not in line with how this part of the code is supposed to work
                raise ValueError("beam must be array-like in this pipeline version") 
            if self.beam_domain is None:
                raise ValueError("not enough info")

            x_beam,y_beam,z_beam=beam_domain
            x_beam*=u.Mpc
            y_beam*=u.Mpc
            z_beam*=u.Mpc
            x_have_lo=x_beam[0]
            x_have_hi=x_beam[-1]
            y_have_lo=y_beam[0]
            y_have_hi=y_beam[-1]
            z_have_lo=z_beam[0]
            z_have_hi=z_beam[-1]
            xy_want_lo=self.xy_vec_for_box[0]
            xy_want_hi=self.xy_vec_for_box[-1]
            z_want_lo=self.z_vec_for_box[0]
            z_want_hi=self.z_vec_for_box[-1]
            if (xy_want_lo<x_have_lo):
                extrapolation_warning("low x",   xy_want_lo,  x_have_lo)
            if (xy_want_hi>x_have_hi):
                extrapolation_warning("high x",   xy_want_hi,  x_have_hi)
            if (xy_want_lo<y_have_lo):
                extrapolation_warning("low y",   xy_want_lo,  y_have_lo)
            if (xy_want_hi>y_have_hi):
                extrapolation_warning("high y",   xy_want_hi,  y_have_hi)
            if (z_want_lo<z_have_lo):
                extrapolation_warning("low z",   z_want_lo,  z_have_lo)
            if (z_want_hi>z_have_hi):
                extrapolation_warning("high z",   z_want_hi,  z_have_hi)
            evaled_num=RGI(beam_domain,self.synth_beam,
                           bounds_error=False,fill_value=None)(self.to_eval_at).T
            self.evaled_num=evaled_num

            synth_beam_extremum=np.max(np.abs(self.synth_beam))
            synth_beam_norm=TwoSlopeNorm(0,vmin=-synth_beam_extremum,vmax=synth_beam_extremum)
            comprehensive_slice_figure(self.synth_beam, 
                                       norm=synth_beam_norm,
                                       cmap="RdBu",
                                       name="beam_box_pre__interpolation.png")
            comprehensive_slice_figure(evaled_num,
                                       norm=synth_beam_norm,
                                       cmap="RdBu",
                                       name="beam_box_post_interpolation.png")
        
        self.beam_domain=beam_domain
        self.evaled_num=evaled_num
        
        self.evaled_num_padded=None
        if evaled_num is not None:
            assert(not np.all(np.isclose(evaled_num,0,atol=1e-16))), "synthesized beam should not be identically vanishing"
            pad_lo_xy,pad_hi_xy=get_padding(self.Nvox )
            pad_lo_z, pad_hi_z =get_padding(self.Nvoxz)
            evaled_num_padded=np.pad(evaled_num,((pad_lo_xy,pad_hi_xy),(pad_lo_xy,pad_hi_xy),(pad_lo_z,pad_hi_z),),"wrap")
            taper_for_convolution=Blackman_Harris_safe_for_FFT(2*self.Nvoxz-1)
            Nxy_padded=2*self.Nvox-1
            self.taper_for_convolution=np.tile(taper_for_convolution, (Nxy_padded,Nxy_padded,1))
            self.evaled_num_padded=evaled_num_padded*self.taper_for_convolution
        
        # strictness control for realization averaging
        self.frac_tol=frac_tol
        self.N_realizations=int(np.round(self.frac_tol**-2))

        # P_MC_complete interpolation bins
        self.kperpbins_interp=kperpbins_interp
        self.kparbins_interp=kparbins_interp

        # realization, averaging, and interpolation placeholders if no prior info
        self.P_unbinned_running_sum=np.zeros(self.box_shape)*self.power_unit
        self.P_MC_complete=P_MC_complete
        self.P_interp=None
        self.MC_not_complete=None
        if self.Nkpar>0 and self.Nkpar is not None:
            self.N_cumul=np.zeros((self.Nkperp,self.Nkpar))
        else:
            self.N_cumul=np.zeros((self.Nkperp,))

    def resample_P_fid_on_grid(self,FoG=False): # resample a 1D power spec on a 3D grid. to break these symmetries, you can do a bit of reverse-engineering: do what you want to the box -> update the T_pristine attribute -> form a power spec using the same cosmo_stats object -> save that unbinned power spec as P_fid_grid -> continue with your Monte Carlo or whatever
        assert(len(self.k_fid)==len(self.P_fid) or len(self.k_fid)==len(self.P_fid.T))
        sort_array=np.argsort(self.kmag_grid_corner_flat)
        kmag_grid_corner_flat_sorted=self.kmag_grid_corner_flat[sort_array]
        P_fid_flattened_box=np.zeros(self.Nvox**2*self.Nvoxz)
        interpolator=RGI((self.k_fid.value,),self.P_fid.value,method="cubic",
                          bounds_error=False,fill_value=None)
        P_fid_flattened_box[sort_array]=interpolator(kmag_grid_corner_flat_sorted.value[:,None])
        P_fid_box=np.reshape(P_fid_flattened_box,self.box_shape,order="C")
        P_fid_box[P_fid_box<0]=0.
        P_fid_box[np.isnan(P_fid_box)]=0.

        FoG_modulation=1.
        if FoG:
            alpha_FoG=1 # what CHIME 2026 uses 
            sigma_FoG=(1.93-1.48*(self.z_ctr-1)+0.81*(self.z_ctr-1)**2)*self.h.value # cf. eq. 11 of the CHIME/cosmology 2026 interpretation paper
            kmag_safe=np.copy(self.kmag_grid_corner)
            kmag_safe[kmag_safe==0]=1./u.Mpc # try inf (bad), 1., 0...
            kmu=np.abs(self.kz_grid_corner)/kmag_safe # k-par/k
            D_FoG_HI=1/(1+ 0.5*(kmu*alpha_FoG*sigma_FoG)**2 ) # cf. eq. 10 of the CHIME/cosmology 2026 interpretation paper
            FoG_modulation=D_FoG_HI**2
            FoG_modulation=1
        self.P_fid_box=P_fid_box*FoG_modulation
            
    def generate_P(self,T_use=None): # from a box of temperature field values
        if T_use is None:
            if self.evaled_num is None:
                T_use="pristine"
            else:
                T_use="beam"
        if (T_use.lower()=="beam"):
            T_use=None
            if self.T_beam is None:
                if self.T_pristine is None:
                    raise ValueError("not enough info")
                else:
                    self.T_beam=convolve(self.evaled_num_padded,self.T_pristine.value,mode="valid")*self.temp_unit
            T_use=self.T_beam
        elif T_use.lower()=="pristine":
            T_use=self.T_pristine
        else:
            raise ValueError("invalid state of box beam knowledge. try again with pristine or beam!")
        T_use=T_use.to(u.mK)
        
        T_tilde=fftshift( fftn( 
                                ifftshift(T_use*self.taper_xyz_centre)*self.d3r,
                                s=self.box_shape, axes=self.transform_axes, norm="backward"        
                              ) 
                        ) # centre-origin
        modsq_T_tilde=np.abs(T_tilde)**2 *self.temp_unit**2*self.length_unit**6
        P_unbinned=modsq_T_tilde/self.effective_volume # box-shaped, but calculated according to the power spectrum estimator equation

        self.P_unbinned=P_unbinned # centre-origin
        
        if self.bin_each_realization:
            self.bin_power()
        
        self.P_unbinned_running_sum+=P_unbinned

    def bin_power(self,power_to_bin=None):
        if power_to_bin is None:
            power_to_bin=self.P_unbinned
        power_to_bin_flat=np.reshape(power_to_bin,(self.Nvox**2*self.Nvoxz,))

        P_binned=binned_statistic_dd(sample=self.coords_to_use, values=power_to_bin_flat,
                                     bins=self.bins_to_use, statistic="mean").statistic
        
        N_cumul= binned_statistic_dd(sample=self.coords_to_use, values=power_to_bin_flat,
                                     bins=self.bins_to_use, statistic="count").statistic

        P_binned[np.isnan(P_binned)]=0.
        self.P_binned=P_binned
        N_cumul[np.isnan(N_cumul)]=0.
        self.N_cumul=N_cumul
    
    def generate_GRF(self): # Gaussian random field realization consistent with a power spectrum of choice
        assert self.Nkperp<self.Nvox, "Nvox should be >= Nkperp"
        assert self.Nkpar<self.Nvoxz, "Nvoxz should be >= Nkpar"
        sigmas=np.sqrt(self.physical_volume*self.P_fid_box/2.) # from inverting the estimator equation and turning variances into std devs
        
        # scipy irfftn puts all the variance into the real component of the half-axis slice of the last axis it transforms in the box. I need to anticipate this by giving those voxels' real components all the variance! (Nothing will be overcounted because the imag part is thrown away)
        all_voxels_along_all_but_last_axis=tuple(slice(0,l) for l in self.box_shape[:-1])
        sigmas[all_voxels_along_all_but_last_axis + (slice(0,1),)]*=np.sqrt(2) # zero mode always needs the adjustment
        if self.box_shape[-1]%2==0: # when your last axis has an even number of voxels...
            half_axis=self.box_shape[-1]//2
            sigmas[all_voxels_along_all_but_last_axis + (slice(half_axis,half_axis+1),)]*=np.sqrt(2) # the Nyquist mode also needs the adjustment
        sigmas[self.kmag_grid_for_comparison==0.]=0. # enforce zero-mean. This point is self-conjugate anyway!!
        T_tilde_Re,T_tilde_Im=self.rng.normal(loc=0.*sigmas,scale=sigmas,size=np.insert(sigmas.shape,0,2)) # corner-origin
        T_tilde=T_tilde_Re+1j*T_tilde_Im # have not yet applied the symmetry that ensures T is real-valued 
        if self.wedge_cut:
            T_tilde[self.voxels_in_wedge_corner]=0.

        T=fftshift(irfftn(T_tilde*self.d3k.value,
                          s=self.box_shape,
                          axes=self.transform_axes,
                          norm="forward"))/self.iftnorm

        T*=self.temp_unit # centre_origin
        if self.fg_box is not None:
            T+=self.fg_box
        
        self.T_pristine=T
        if self.synth_beam is not None:
            self.T_beam=convolve(self.evaled_num_padded,T.value,mode="valid")*self.temp_unit

    def power_Monte_Carlo(self,interfix:str=""): # since box generation is not deterministic
        self.MC_not_complete=True
        if self.synth_beam is None:
            T_use="pristine"
        else: 
            T_use="beam"
        i=0

        t0=time.time()
        for i in range(self.N_realizations):
            self.generate_GRF()
            self.generate_P(T_use=T_use)
            ti=time.time()
            if ((ti-t0)>3600): # actually save the realizations every hour
                np.save("P_"+interfix+"_MC_incomplete.npy",self.P_unbinned_running_sum.value/i)
                t0=time.time()

        P_unbinned_MC_complete=self.P_unbinned_running_sum/self.N_realizations
        self.P_unbinned_MC_complete=P_unbinned_MC_complete
        self.bin_power(power_to_bin=P_unbinned_MC_complete)
        P_binned_MC_complete=self.P_binned
        self.P_binned_MC_complete=P_binned_MC_complete*self.power_unit

        self.N_per_realization=self.N_cumul/self.N_realizations
        # if self.evaled_num_padded is not None: 
            # print("self.evaled_num_padded.shape, self.T_pristine.shape, self.T_beam.shape =",self.evaled_num_padded.shape, self.T_pristine.shape, self.T_beam.shape)

    def interpolate_P(self,use_P_fid:bool=False):
        if use_P_fid:
            self.P_MC_complete=self.P_fid
        else:
            if (self.P_MC_complete is None):
                print("WARNING: P_MC_complete DNE yet. \nAttempting to calculate it now...")
                self.power_Monte_Carlo()
            if (self.kperpbins_interp is None):
                raise ValueError("not enough info")

        if (self.kparbins_interp is not None):
            kpar_have_lo=  self.kperpbins[0]
            kpar_have_hi=  self.kperpbins[-1]
            kperp_have_lo= self.kparbins[0]
            kperp_have_hi= self.kparbins[-1]

            kpar_want_lo=  self.kperpbins_interp[0]
            kpar_want_hi=  self.kperpbins_interp[-1]
            kperp_want_lo= self.kparbins_interp[0]
            kperp_want_hi= self.kparbins_interp[-1]

            if (kpar_want_lo<kpar_have_lo):
                extrapolation_warning("low kpar",   kpar_want_lo,  kpar_have_lo)
            if (kpar_want_hi>kpar_have_hi):
                extrapolation_warning("high kpar",  kpar_want_hi,  kpar_have_hi)
            if (kperp_want_lo<kperp_have_lo):
                extrapolation_warning("low kperp",  kperp_want_lo, kperp_have_lo)
            if (kperp_want_hi>kperp_have_hi):
                extrapolation_warning("high kperp", kperp_want_hi, kperp_have_hi)
            modes_defined_at=(self.kperpbins_grid,self.kparbins_grid)
            modes_to_eval_at=(self.kperp_interp_grid,self.kpar_interp_grid).T
        else:
            k_have_lo=self.kperpbins[0]
            k_have_hi=self.kperpbins[-1]
            k_want_lo=self.kperpbins_interp[0]
            k_want_hi=self.kperpbins_interp[-1]
            if (k_want_lo<k_have_lo):
                extrapolation_warning("low k",k_want_lo,k_have_lo)
            if (k_want_hi>k_have_hi):
                extrapolation_warning("high k",k_want_hi,k_have_hi)
            modes_defined_at=(self.kperpbins.value,)
            modes_to_eval_at=(self.kperpbins_interp.value,)
        P_interpolator=RGI(modes_defined_at,self.P_MC_complete,
                           bounds_error=self.avoid_extrapolation,fill_value=None)
        P_interp=P_interpolator(modes_to_eval_at)
        if self.kparbins_interp is not None:
            P_interp=P_interp.T # anticipate the RGI behaviour
        self.P_interp=P_interp
####################################################################################################################################################################################################################################

def beam_type_distribution(N_NS,N_EW,N_types,distribution="random",frame_width=2):
    N_ant=N_NS*N_EW
    if N_types>0:
        rng=np.random.default_rng()
        if distribution=="random":
            synthesized_beam_types=rng.integers(0,N_types,size=(N_ant,))
        elif distribution=="corner":
            if N_types!=4:
                raise ValueError("conflicting info") # in order to use corner systematics layout, you need four beam types
            synthesized_beam_types=np.zeros((N_NS,N_EW),dtype=np.int32)
            half_NS=N_NS//2
            half_EW=N_EW//2
            synthesized_beam_types[:half_NS,half_EW:]=1
            synthesized_beam_types[half_NS:,:half_EW]=2
            synthesized_beam_types[half_NS:,half_EW:]=3 # the quarter of the array with no explicit overwriting keeps its idx=0 (as necessary)
        elif distribution=="diagonal":
            raise ValueError("not yet implemented")
        elif distribution=="column":
            synthesized_beam_types=np.zeros((N_NS,N_EW),dtype=np.int32)
            if N_types>1:
                for i in range(1,N_types):
                    synthesized_beam_types[:,i::N_types]=i
        elif distribution=="frame":
            synthesized_beam_types=np.zeros((N_NS,N_EW),dtype=np.int8)
            if N_types>1: # stricter threshold for this case based on where the random numbers come from
                sh=synthesized_beam_types.shape
                sz=synthesized_beam_types.size
                sz_ind=np.arange(sz)
                sz_ind_rectangular=sz_ind.reshape(sh)
                indices_for_systs=~np.isin(sz_ind_rectangular, 
                                           sz_ind_rectangular[frame_width:-frame_width, 
                                                              frame_width:-frame_width])
                synthesized_beam_types[indices_for_systs]=1
                synthesized_beam_types[indices_for_systs]=rng.integers(1,high=N_types,
                                                                  size=np.sum(synthesized_beam_types[indices_for_systs]))
                                                                #   size=2*(N_NS+N_EW)-4)
        else:
            raise ValueError("beam distribution pattern not yet implemented")
        
        synthesized_beam_types=np.reshape(synthesized_beam_types,(N_ant,),order="C")
    else:
        synthesized_beam_types=np.zeros(N_ant,dtype=np.float64)

    weights=np.bincount(synthesized_beam_types)/N_ant
    return synthesized_beam_types,weights

"""
this class helps compute numerical windowing boxes for brightness temp boxes resulting
from beams that have the flexibility to differ on a per-antenna basis.
"""

class synthesize_beam(beam_effects): # developed with rectangular arrays in mind
    def __init__(self,
                 array_version:str="full",                                         # run a simulation for full or pathfinder CHORD?
                 b_NS:float=b_NS,b_EW:float=b_EW,                                  # N-S and E-W baseline lengths (m)
                 offset_rad:float=def_offset,                                      # (astropy-unitless because this class expects rad) CHORD is aligned with magnetic, not geographical north, so, when mathematically constructing the uv coverage, rotate the rectangular array grid
                 observing_dec:float=def_observing_dec,                            # declination to observe at (º)
                 N_pbws_pert:int=0,                                                # number of antennas with perturbed primary beams
                 N_timesteps:float=def_N_timesteps,                                # number of timesteps in rotation synthesis
                 nu_ctr:float=nu_HI_z0,                                            # central frequency of the survey of interest
                 N_grid_pix:int=def_PA_N_grid_pix,                                 # number of pixels per side of the gridded uv plane
                 Delta_nu:float=CHORD_channel_width_MHz,                           # channel width in frequency (MHz)
                 distribution:str="random",                                        # distribution of per-antenna systematics. the options I've encoded for now are random, column, and corner, based on where the fiducial beam types are placed within the array
                 evol_restriction_threshold:float=def_evol_restriction_threshold,  # max \delta z/z you will tolerate for the survey of interest and still consider the box close enough to coeval
                 weighting="uniform", Npix=1024,

                 sub_ensemble_of_CST_beams=None,                                   # array-like with shape (N_CST_types, N_pointing_errors+1, N_CST_xy, N_CST_xy, N_CST_freqs)
                 CST_xy=None,CST_freqs=None,                                       # domain of each CST box in the ensemble. this domain is currently assumed to be the same for each box (not very rigorous/robust, but in practice, if you're running a simulation for a given survey frequency, it would be fairly pathological/ unintuitive/ anti–Occam's razor to get these boxes from CST slices at different frequencies. I guess the practical guidance/takeaway here is that my initial implementation will not support getting different boxes from different CST box resolutions)
                 supplementary_name=None # literally just for the July 16th 2026 histogram validation
                 ): 
        # array and observation geometry
        self.N_pbws_pert=N_pbws_pert
        self.N_timesteps=N_timesteps
        self.N_grid_pix=N_grid_pix
        self.distribution=distribution
        self.evol_restriction_threshold=evol_restriction_threshold
        self.Delta_nu=Delta_nu
        N_NS=N_NS_full
        N_EW=N_EW_full
        self.DRAO_lat=DRAO_lat
        if (array_version=="pathfinder"):
            N_NS=N_NS//2
            N_EW=N_EW//2
        N_ant=N_NS*N_EW
        N_bl=N_ant*(N_ant-1)//2
        self.nu_ctr_MHz=nu_ctr.to(u.MHz)
        self.nu_ctr_Hz=nu_ctr.to(u.Hz)
        self.Dc_ctr=comoving_distance(nu_HI_z0/nu_ctr-1)
        self.N_hrs=hrs_per_night
        
        # antenna positions xyz
        antennas_EN=np.zeros((N_ant,2))
        for i in range(N_NS):
            for j in range(N_EW):
                antennas_EN[i*N_EW+j,:]=[j*b_EW.value,i*b_NS.value]
        antennas_EN-=np.mean(antennas_EN,axis=0) # centre the Easting-Northing axes in the middle of the array
        try:
            offset_rad.to(u.rad)
        except:
            offset_rad=offset_rad*u.rad
        offset_from_latlon_rotmat=np.array([[np.cos(offset_rad),-np.sin(offset_rad)],
                                            [np.sin(offset_rad), np.cos(offset_rad)]]) # use this rotation matrix to adjust the NS/EW-only coords
        for i in range(N_ant):
            antennas_EN[i,:]=np.dot(antennas_EN[i,:].T,offset_from_latlon_rotmat)
        dif=antennas_EN[0,0]-antennas_EN[0,-1]+antennas_EN[0,-1]-antennas_EN[-1,-1]
        up=np.reshape(2+(-antennas_EN[:,0]+antennas_EN[:,1])/dif, (N_ant,1), order="C") # eyeballed ~2 m vertical range that ramps ~linearly from a high near the NW corner to a low near the SE corner
        antennas_ENU=np.hstack((antennas_EN,up))
        print("extrema of antennas_ENU:",np.min(antennas_ENU),np.min(np.abs(antennas_ENU)),np.max(antennas_ENU))
        
        zenith=np.array([np.cos(DRAO_lat),0,np.sin(DRAO_lat)]) # Jon math
        east=np.array([0,1,0])
        north=np.cross(zenith,east)
        lat_mat=np.vstack([north,east,zenith])
        antennas_xyz=antennas_ENU@lat_mat.T
        print("extrema of antennas_xyz:",np.min(antennas_xyz),np.min(np.abs(antennas_xyz)),np.max(antennas_xyz))
        
        # line-of-sight quantities
        bw_MHz=self.nu_ctr_MHz*evol_restriction_threshold
        N_chan=int(bw_MHz/self.Delta_nu)
        self.N_chan=N_chan
        nu_lo=self.nu_ctr_MHz-bw_MHz/2.
        nu_hi=self.nu_ctr_MHz+bw_MHz/2.
        surv_channels_MHz=np.linspace(nu_hi,nu_lo,N_chan) # decr.
        surv_channels_Hz=surv_channels_MHz.to(u.Hz)
        surv_wavelengths=c/surv_channels_Hz # incr.
        self.surv_wavelengths=surv_wavelengths.decompose()
        z_channels=nu_HI_z0/surv_channels_MHz-1.
        comoving_distances_channels=np.asarray([comoving_distance(chan).value for chan in z_channels]) # incr.
        self.comoving_distances_channels=comoving_distances_channels*u.Mpc
        self.ctr_chan_comov_dist=self.comoving_distances_channels[N_chan//2]
        self.surv_channels_MHz=surv_channels_MHz
        self.lambda_obs=surv_wavelengths[0]

        # helper args
        self.CST_xy=CST_xy
        CST_Delta_xy=CST_xy[1]-CST_xy[0]
        N_CST_xy=len(CST_xy)
        self.uvbins_CST=fftshift(fftfreq(N_CST_xy,d=CST_Delta_xy))
        self.CST_freqs=CST_freqs
        self.CST_deltanu=CST_freqs[1]-CST_freqs[0]
        self.N_CST_xy=N_CST_xy
        self.N_CST_freqs=len(CST_freqs)
        self.CST_freqs_obs_units=self.CST_freqs.to(u.Hz)
        self.CST_deltanu_obs_units=self.CST_deltanu.to(u.Hz)

        # beam synthesis numerics
        self.weighting=weighting
        self.Npix=Npix

        if type(sub_ensemble_of_CST_beams) is not list: # can't use .ndim because it doesn't behave well for the inhomog arrays of the else
            print("synthesize_beam received only a !fiducial! beam box")
            fidu_box=sub_ensemble_of_CST_beams
            self.all_boxes=np.expand_dims(sub_ensemble_of_CST_beams,axis=0)
            N_total_beam_types=1
            self.N_total_beam_types=1
        else:
            print("synthesize_beam received both !fiducial and systematic-laden! beam boxes")
            fidu_box,syst_boxes=sub_ensemble_of_CST_beams # should be unpackable into two arrays:
            assert fidu_box.ndim==3 and syst_boxes.ndim==5 # one box and one "2D array of 3D boxes"
            self.N_CST_types,self.N_max_pointing_errors,_,_,_=syst_boxes.shape

            # figure out the actual number of beam types and store the beam types as a list of boxes, not 2D array of boxes + standalone box
            N_pointing_errors_per_CST_case=np.zeros(self.N_CST_types,dtype=int)
            nnn=0
            all_boxes=[fidu_box]
            for i in range(self.N_CST_types):
                for j in range(self.N_max_pointing_errors):
                    box_to_add=syst_boxes[i,j,:,:,:]
                    if not np.all(np.isclose(box_to_add,0.)): # NEW
                        N_pointing_errors_per_CST_case[i]=j # only the (j-1)st case is meaningful, but that is in zero-based indexing, not one-based counting.
                        all_boxes.append(box_to_add)
                        nnn+=1
            self.N_pointing_errors_per_CST_case=N_pointing_errors_per_CST_case
            N_total_beam_types=nnn
            self.N_total_beam_types=N_total_beam_types

            all_boxes=np.asarray(all_boxes)
            self.all_boxes=all_boxes
        self.pb_types,self.weights=beam_type_distribution(N_NS,N_EW,N_total_beam_types, distribution=self.distribution)
        np.save("antennas_xyz_"+supplementary_name+".npy",antennas_xyz) # for the baseline type vs length histogram
        np.save("pb_types_"+supplementary_name+".npy",self.pb_types) # for the baseline type vs length histogram
        if self.N_total_beam_types>1:
            self.N_baseline_classes=self.N_total_beam_types*(self.N_total_beam_types-1)
        else:
            self.N_baseline_classes=1

        comprehensive_slice_figure(fidu_box,
                                   norm=LogNorm(vmax=1),
                                   name="fidu_box_unprocessed.png")

        # ungridded instantaneous uv-coverage (baselines in xyz)
        # second use of the loop: iterate over baselines to make arrays of beam type indices     
        uvw_inst=np.zeros((N_bl,3))
        indices_of_constituent_ant_pb_types=np.zeros((N_bl,2))
        k=0
        for i in range(N_ant):
            for j in range(i+1,N_ant):
                uvw_inst[k,:]=antennas_xyz[i,:]-antennas_xyz[j,:]
                indices_of_constituent_ant_pb_types[k]=[self.pb_types[i],self.pb_types[j]]
                k+=1
        uvw_inst=np.vstack((uvw_inst,-uvw_inst))
        self.uvw_inst=uvw_inst
        indices_of_constituent_ant_pb_types=np.vstack((indices_of_constituent_ant_pb_types,indices_of_constituent_ant_pb_types)) # get the opposite-permutation baselines for free
        self.indices_of_constituent_ant_pb_types=indices_of_constituent_ant_pb_types
        print("computed ungridded instantaneous uv-coverage")
        print("extrema of uvw_inst:",np.min(uvw_inst),np.min(np.abs(uvw_inst)),np.max(uvw_inst))

        # rotation-synthesized uv-coverage *******(N_bl,3,N_timesteps), accumulating xyz->uvw transformations at each timestep
        hour_angle_ceiling=np.pi*self.N_hrs/12
        hour_angles=np.linspace(0,hour_angle_ceiling,self.N_timesteps)
        thetas=hour_angles.value*15*np.pi/180*u.rad # don't use built-in astropy conversions for this because it won't realize my hr<->rad conversion is about the rotation rate of the earth
        
        try:
            observing_dec.to(u.rad)
        except:
            observing_dec=observing_dec*u.rad
        zenith=np.array([np.cos(observing_dec),0,np.sin(observing_dec)]) # Jon math redux
        east=np.array([0,1,0])
        north=np.cross(zenith,east)
        project_to_dec=np.vstack([east,north])

        uv_synth=np.zeros((2*N_bl,2,self.N_timesteps)) # N_baselines, u and v, N_timesteps
        for i,theta in enumerate(thetas): # thetas are the rotation synthesis angles (converted from hr. angles using 15 deg/hr rotation rate)
            accumulate_rotation=np.array([[ np.cos(theta),np.sin(theta),0],
                                          [-np.sin(theta),np.cos(theta),0],
                                          [ 0,            0,            1]])
            uvw_rotated=uvw_inst@accumulate_rotation
            uvw_projected=uvw_rotated@project_to_dec.T
            uv_synth[:,:,i]=uvw_projected/self.lambda_obs # ok for this to be the first LoS slice!! this *is* just for one LoS slice
        self.uv_synth=uv_synth # units are wavelengths
        print("extrema of uv_synth are",np.min(uv_synth),np.min(np.abs(uv_synth)),np.max(uv_synth))
        print("synthesized rotation")

        uvmagmax=np.max(np.abs(self.uv_synth)) # have not yet implemented the horizon constraint (inability to see more than horizon-to-horizon)
        uvmagmin=uvmagmax/Npix
        thetamax=1/uvmagmin # these are 1/-convention Fourier duals, not 2pi/-convention Fourier duals
        self.thetamax=thetamax
        self.xy_image_broadest=self.ctr_chan_comov_dist*np.arange(-thetamax,thetamax,Npix)
        uvbins=np.linspace(-uvmagmax,uvmagmax,Npix)
        self.d2u=uvbins[1]-uvbins[0]
        self.uvbins_use=np.append(uvbins,uvbins[-1]+uvbins[1]-uvbins[0])

    def calc_uv_slice(self):
        implane=np.zeros((self.Npix,self.Npix))
        for i in range(self.N_total_beam_types):
            type_i=self.pb_types[i]
            for j in range(i+1):
                type_j=self.pb_types[j]

                here=(self.indices_of_constituent_ant_pb_types[:,0]==i
                        )&(self.indices_of_constituent_ant_pb_types[:,1]==j)
                u_here=self.uv_synth[here,0,:] # [N_bl,2,N_hr_angles]
                v_here=self.uv_synth[here,1,:]
                N_bl_here,N_hr_angles_here=u_here.shape # (N_bl,N_hr_angles)
                N_here=N_bl_here*N_hr_angles_here
                reshaped_u=np.reshape(u_here,N_here,order="C")
                reshaped_v=np.reshape(v_here,N_here,order="C")
                gridded_uv,_,_=np.histogram2d(reshaped_u,reshaped_v,bins=self.uvbins_use) # natural weighting
                comb=np.nonzero(gridded_uv)
                if self.weighting=="custom":
                    gridded_uv[comb]=1/gridded_uv[comb]
                elif self.weighting=="uniform":
                    gridded_uv[comb]=1
                elif self.weighting!="natural":
                    raise ValueError("unknown uv plane weighting scheme")
                gridded_im=fftshift(irfftn(ifftshift(gridded_uv*self.d2u), # irfftn silently discarding imag part of symmetry slices of the last transformed axis is not a problem here because the uv slices in question are entirely real-valued
                                           norm="forward",s=(self.Npix,self.Npix)))
                LoS_1st,LoS_2nd=np.argsort(np.abs(self.nu_obs-self.CST_freqs_obs_units))[:2]
                weight_1st=np.abs(self.nu_obs-self.CST_freqs_obs_units[LoS_1st])/self.CST_deltanu_obs_units
                weight_2nd=np.abs(self.nu_obs-self.CST_freqs_obs_units[LoS_2nd])/self.CST_deltanu_obs_units
                beam_i=self.all_boxes[type_i,:,:,LoS_1st]*weight_1st + self.all_boxes[type_i,:,:,LoS_2nd]*weight_2nd
                beam_j=self.all_boxes[type_j,:,:,LoS_1st]*weight_1st + self.all_boxes[type_j,:,:,LoS_2nd]*weight_2nd
                product=beam_i*beam_j
                beam_ij=np.sqrt(product) # geo mean of the beams of this baseline's two constituent antennas. still on initial CST grid
                beam_ij/=np.max(beam_ij) # beam should already be peak-normalized, pero mejor asegurarse que no haya nada raro... for example, a 2-3-pixel offset in the peak location
                
                interpolator=RBS(self.CST_xy,self.CST_xy, beam_ij) # originally on CST-derived grid
                beam_ij_interpolated=interpolator(self.xy_image_broadest,self.xy_image_broadest) # want to evaluate on image grid that is Fourier-dual to the uv grid
                implane+=gridded_im*beam_ij_interpolated

        plt.figure()
        plt.imshow(gridded_uv.T,origin="lower")
        plt.colorbar()
        plt.savefig("single_slice_gridded_uv.png")
        plt.close()
        # mid=int(self.N_CST_xy//2)
        # norm=implane[mid,mid]
        norm=np.max(implane) # this is what I meant to override on the AM of July 15th 2026 but actually left uncommented somewhere else. maybe it doesn't lead to such crazy artifacts... and if it didn't, that would be convenient... because it would be robust against division-by-zero errors
        implane/=norm
        return implane

    def stack_to_box(self):
        if (self.nu_ctr_MHz.value<(350/(1-self.evol_restriction_threshold/2)) or 
            self.nu_ctr_MHz>(nu_HI_z0/(1+self.evol_restriction_threshold/2))):
            raise ValueError("{:6.2f} is out of bounds".format(self.nu_ctr_MHz))
        N_grid_pix=self.N_grid_pix

        box_xyz=np.zeros((N_grid_pix,N_grid_pix,self.N_chan))
        for i in range(self.N_chan): # rescale the uv-coverage to this channel's frequency
            self.uv_synth=self.uv_synth*self.lambda_obs/self.surv_wavelengths[i] # rescale according to observing frequency: multiply up by the prev lambda to cancel, then divide by the current/new lambda
            self.lambda_obs=self.surv_wavelengths[i] # update the observing frequency for next time
            nu_obs=c/self.lambda_obs
            self.nu_obs=nu_obs.decompose()

            chan_gridded_implane=self.calc_uv_slice() # compute this LoS slice's synthesized beam            
            # box_xyz[:,:,i]=chan_gridded_implane/np.max(chan_gridded_implane) # peak-normalize in configuration space
            box_xyz[:,:,i]=chan_gridded_implane
            if ((i%(self.N_chan//3))==0):
                print("{:7.1f} pct complete".format(i/self.N_chan*100))
        self.box=box_xyz

        # generate a box of r-values (necessary for interpolation to survey domain in cosmo_stats as called by beam_effects)
        xy_vec=self.CST_xy # making the coeval approximation
        z_vec=self.comoving_distances_channels-self.ctr_chan_comov_dist 
        self.xy_vec=xy_vec
        self.z_vec=z_vec
####################################################################################################################################################################################################################################

class reconfigure_CST_beam(object):
    def __init__(self,
                 freq_lo:float=0.580*u.GHz,freq_hi:float=0.620*u.GHz, # low and high frequencies (MHz) for which to translate CST beams
                 delta_nu_CST:float=2e-5*u.GHz,                       # frequency spacing of the CST simulations to use to build up a picture of the beam
                 beam_sim_directory=None,                             # where to import CST beam files from
                 f_head:str="farfield_(f=",                           # beginning of CST beam file names
                 f_mid1:str=")_[1]",f_mid2:str=")_[2]",               # middle of CST beam file names. should include something to distinguish the two polarizations (expected but not strictly enforced... although there's no other part of the file name reading that currently anticipates differences in polarization)
                 f_tail:str="_efield.txt",                            # end of CST beam file names
                 box_outname:str="placeholder",                       # what to call the config space box of CST-informed beam values that results from a complete use of this class
                 Nxy:int=128,                                         # number of pixels per side of frequency slides (get one sky plane square per CST file)
                 multi_CST=True):                                     # set to False to go back to the file name construction order ok for the Jan-Apr 2026 CST
        self.beam_sim_directory=beam_sim_directory
        self.f_head=f_head
        self.f_mid1=f_mid1
        self.f_mid2=f_mid2
        self.f_tail=f_tail
        self.box_outname=box_outname
        self.multi_CST=multi_CST

        freq_hi=freq_hi.to(u.GHz)
        freq_lo=freq_lo.to(u.GHz)
        delta_nu_CST=delta_nu_CST.to(u.GHz)
        freqs_GHz=np.arange(freq_hi.value,freq_lo.value,-delta_nu_CST.value)*delta_nu_CST.unit # descending; usually still in GHz
        freqs=freqs_GHz.to(u.MHz) # descending; MHz
        self.freqs=freqs
        Nfreqs=len(freqs)
        self.Nfreqs=Nfreqs
        zs_for_xis=[nu_HI_z0/freq-1 for freq in freqs] # ascending
        xis=[comoving_distance(z) for z in zs_for_xis] # ascending
        xis=Quantity(xis) # for the typical coeval approximation
        self.xis=xis
        comoving_middle=xis[int(Nfreqs//2)]
        self.CST_z_vec=xis-comoving_middle

        if beam_sim_directory is None:
            print("Do you really mean to attempt CST imports from the working directory?")

        L_xy=comoving_middle
        xy_for_box=L_xy*fftshift(fftfreq(Nxy))
        print("reconfigure_CST_beam.__init__: len(xy_for_box) =",len(xy_for_box))
        self.xy_for_box=xy_for_box
        np.save("xy_vec_for_box"+box_outname,xy_for_box.value)
        self.Nxy=Nxy
        self.xx_grid,self.yy_grid=np.meshgrid(xy_for_box,xy_for_box, indexing="ij") # config space points of interest for the slice (guided by the transverse extent of the eventual config-space box)
        freq_names=np.zeros(Nfreqs,dtype="U6") # store the GHz CST frequencies as strings of the format that Aditya's sims use
        for i,freq in enumerate(freqs_GHz):
            freq_name=f"{freq:.4f}" # round to four decimal places and convert to string
            if multi_CST: # do not strip trailing zeros because of how those file names are formatted
                freq_names[i]=freq_name
            else:
                freq_names[i]=freq_name.rstrip("0") # strip trailing zeros because of how those file names are formatted
            assert freq_names[i]!="0"
        self.freq_names=freq_names

    def translate_sim_beam_slice(self,CST_filename:str,i:int=0):
        df = pd.read_table(CST_filename, skiprows=[0, 1], sep="\s+", engine="python", 
                           
                           # lots of fields in each CST sim, but only the first three are helpful for forming Stokes I E-field beams (precursor for the Stokes I power beams I form from two pols of a given simulation setup and frequency)
                           names=["theta", "phi",  # spherical coordinates, only valid locally, that specify which direction each beam value describes
                                  "AbsE",          # amplitude of E-field beam for the pol you are currently reading
                                                   # two terms in breaking AbsE into Ludwig-III, which is more nonlocally generalizable than the native CST sph coords
                                  "AbsCr", "PhCr", # 1. if you excite a current in x and your feed is x-polarized, what is the response in y?       
                                  "AbsCo", "PhCo", # 2. if you excite a current in x and your feed is x-polarized, what is the response in x?
                                  "AxRat"])        # polarization ellipticity
        theta_deg=df.theta.values*u.deg
        idx_with_theta_to_keep=np.nonzero(np.abs(theta_deg)<=90.*u.deg)
        linear_units_one_pol=10**(df.AbsE.values/10)[idx_with_theta_to_keep] # non-log values
        theta=theta_deg[idx_with_theta_to_keep].to(u.rad)
        phi_deg=df.phi.values*u.deg
        phi=phi_deg[idx_with_theta_to_keep].to(u.rad)
        x=self.xis[i]*np.sin(theta)*np.cos(phi)
        y=self.xis[i]*np.sin(theta)*np.sin(phi)
        sky_xy_points=np.array([x,y]).T
        return sky_xy_points,linear_units_one_pol
    
    def construct_CST_box(self):
        slice_grid_points=np.array([self.xx_grid,self.yy_grid]).T
        box=np.zeros((self.Nxy,self.Nxy,self.Nfreqs)) # hold interpolated beam slices
        for i in range (self.Nfreqs):
            if self.multi_CST:
                name1=self.beam_sim_directory+self.f_head+self.f_mid1+self.freq_names[i]+self.f_tail
                name2=self.beam_sim_directory+self.f_head+self.f_mid2+self.freq_names[i]+self.f_tail
            else:
                name1=self.beam_sim_directory+self.f_head+self.freq_names[i]+self.f_mid1+self.f_tail
                name2=self.beam_sim_directory+self.f_head+self.freq_names[i]+self.f_mid2+self.f_tail
            sky_xy_points,uninterp_slice_pol1=self.translate_sim_beam_slice(name1, i=i) # both polarizations will be sampled at the same (theta,phi) because they come from the same simulation = same discretization
            _,            uninterp_slice_pol2=self.translate_sim_beam_slice(name2, i=i)            

            product=uninterp_slice_pol1*uninterp_slice_pol2
            product_interpolated=gd(sky_xy_points,product,slice_grid_points,  # assumes pol1, pol2 discretized the same way... they will be, for sensibly-configured simulations
                                    method="nearest") # linear applies nans when extrap would be necessary
            power=product_interpolated/np.max(product_interpolated)
            box[:,:,i]=power

            if ((i%(self.Nfreqs//3))==0):
                print("{:7.1f} pct complete".format(i/self.Nfreqs*100))
        np.save("CST_box_"+self.box_outname,box)
        self.box=box # centre-origin
        print("reconfigure_CST_beam.construct_CST_box: box.shape=",box.shape)

class CHORD_sense(object): # modified from a notebook helpfully shared by Debanjan Sarkar in April 2025
    def __init__(
        self,
        spacing:np.ndarray=[b_EW,b_NS], # N-S and E-W baselines (m)
        n_side:np.ndarray=[22,24],    # number of dishes per side of the array (N-S, E-W) directions
        orientation=None,             # same comment about CHORD alignment as in the synthesize_beam documentation (expects rad!)
        center:np.ndarray=[0,0],      # where to put the axis origin of the antenna location x-y coordinates (if you leave the default in place, it'll make the zero point the physical centre of the array)
        
        freq_cen:float = 900.*u.MHz,                  # central frequency of the observation/survey
        dish_diameter:float = 6.*u.m,                 # dish diameter
        Trcv:float = 30.*u.K,                         # receiver temperature. default = optimistic CHORD prognosis
        latitude:float = DRAO_lat,                    # latitude of the observatory (default = DRAO)
        integration_time:float= 10.*u.s,              # duration of a single integration
        time_per_day:float = 6.*u.hour,               # time spent observing per day
        n_days:int = 100 ,                            # number of days in the observation
        bandwidth:float=80.*u.MHz,                    # bandwidth of the survey/observation
        coherent:bool = False,                        # add baselines coherently if they are not instantaneously redundant?
        tsky_ref_freq:float = 400.*u.MHz,             # frequency to which the sky temp is referenced
        tsky_amplitude:float = 25.*u.K,               # sky temp
        
        horizon_buffer:float = 0.1*littleh/u.Mpc, # how many near-the-horizon modes to exclude
        foreground_model:str = "optimistic",      # foreground model for sensitivity calculations

        sv:bool=False, # extract sample variance from 21cmSense? (defaults to false because 21cmSense is ill-suited to performing these calculations for post-EoR experiments with wide fields of view [like CHORD] and I get this info for free for my CHORD forecasts from my Monte Carlo ensembles)
        tn:bool=True   # thermal noise (this is the other big contributor to the noise calculation, and my main motivation for using 21cmSense at all for CHORD forecasts)
    ):
        bl_max=bl_max.to(u.m)
        bl_min=bl_min.to(u.m)
        dish_diameter=dish_diameter.to(u.m)
        freq_cen=freq_cen.to(u.MHz)
        bandwidth=bandwidth.to(u.MHz)
        integration_time=integration_time.to(u.s)
        tsky_ref_freq=tsky_ref_freq.to(u.MHz)
        tsky_amplitude=tsky_amplitude.to(u.K)
        horizon_buffer=horizon_buffer.to(littleh/u.Mpc)
        bl_max=np.sqrt((spacing[0]*n_side[0])**2+(spacing[1]*n_side[1])**2)
        bl_min=np.min(spacing)
        self.spacing = spacing
        self.n_side = n_side
        self.orientation = orientation
        self.center = center
        self.freq_cen = freq_cen
        self.dish_diameter = dish_diameter
        self.Trcv =  Trcv
        self.latitude = latitude
        self.integration_time = integration_time
        self.time_per_day = time_per_day
        self.n_days = n_days
        n_channels = bandwidth.value/CHORD_channel_width_MHz
        self.n_channels = n_channels
        self.bandwidth = bandwidth
        self.coherent = coherent
        self.bl_max = bl_max
        self.bl_min = bl_min
        self.tsky_ref_freq = tsky_ref_freq
        self.tsky_amplitude = tsky_amplitude
        self. horizon_buffer =  horizon_buffer
        self.foreground_model = foreground_model 
        self.sv=sv
        self.tn=tn

        ant_pos = self.rectangle_generator()
        
        observatory = Observatory(antpos=ant_pos,
                          beam = GaussianBeam(frequency=self.freq_cen,
                                              dish_diameter=self.dish_diameter),
                          Trcv = self.Trcv,   # The receiver temp will dominate over sky temp at this freq. (unlike EoR)
                          latitude = self.latitude)
        
        observation = Observation(observatory = observatory,
                          integration_time = self.integration_time, # The time in sec, telescope integrates to give one sanpshot
                          time_per_day = self.time_per_day,  # The time in hours, to observe per day (a typical choice of 8 hrs)
                          #hours_per_day = self.time_per_day,  # The time in hours, to observe per day (a typical choice of 8 hrs)
                          n_days = self.n_days,    # Total number of days of observation
                          n_channels = self.n_channels, # The number of channels
                          bandwidth = self.bandwidth,  # Bandwidth of obs
                          coherent = self.coherent, # Whether to add different baselines coherently if they are not instantaneously redundant.
                          tsky_ref_freq = self.tsky_ref_freq,
                          tsky_amplitude = self.tsky_amplitude
                          )

        sensitivity = PowerSpectrum(
            observation = observation,
            horizon_buffer = self. horizon_buffer,
            foreground_model = self.foreground_model)
        self.sensitivity=sensitivity
        
    def rectangle_generator(self): # Generate a grid of baseline locations filling a rectangular array for CHORD/HIRAX. 
        if self.spacing is not None:
            if not isinstance(self.spacing, (int, float, list, np.ndarray)):
                raise TypeError('spacing must be a scalar or list/numpy array')
            self.spacing = np.asarray(self.spacing)
            if self.spacing.size < 2:
                self.spacing = np.resize(self.spacing,(1,2))
            if np.all(np.less_equal(self.spacing,np.zeros((1,2)))):
                raise ValueError('spacing must be positive')

        if self.orientation is not None:
            if not isinstance(self.orientation, (int,float)):
                raise TypeError('orientation must be a scalar')

        if self.center is not None:
            if not isinstance(self.center, (list, np.ndarray)):
                raise TypeError('center must be a list or numpy array')
            self.center = np.asarray(self.center)
            if self.center.size != 2:
                raise ValueError('center should be a 2-element vector')
            self.center = self.center.reshape(1,-1)

        if self.n_side is None:
            raise NameError('Atleast one value of n_side must be provided')
        else:
            if not isinstance(self.n_side,  (int, float, list, np.ndarray)):
                raise TypeError('n_side must be a scalar or list/numpy array')
            self.n_side = np.asarray(self.n_side)
            if self.n_side.size < 2:
                self.n_side = np.resize(self.n_side,(1,2))
            if np.all(np.less_equal(self.n_side,np.zeros((1,2)))):
                raise ValueError('n_side must be positive')

            n_total = np.prod(self.n_side, dtype=np.uint8)
            xn,yn = self.n_side
            xs,ys=self.spacing
            n_total = xn*yn

            x = np.arange(0, xn)
            x = x - np.mean(x)
            x = x*xs

            y = np.arange(0, yn)
            y = y - np.mean(y)
            y = y*ys 
        
            z = np.zeros(n_total)
            xv, yv = np.meshgrid(x,y, indexing="ij")
            xy = np.hstack((xv.reshape(-1,1),yv.reshape(-1,1)))

        if len(xy) != n_total:
            raise ValueError('Sizes of x- and y-locations do not agree with n_total')

        try:
            self.orientation.to(u.rad)
        except:
            self.orientation=self.orientation*u.rad
        if self.orientation is not None:   # Perform any rotation
            rot_matrix = np.asarray([[np.cos(self.orientation),-np.sin(self.orientation)], 
                                     [np.sin(self.orientation), np.cos(self.orientation)]])
            xy = np.dot(xy, rot_matrix.T)

        if self.center is not None:   # Shift the center
            xy += self.center
     
        z = np.zeros(shape=(n_total,1))
        XY = np.hstack((xy,z))

        return (np.asarray(XY)*u.m)
    
    def sense_1d(self):
        sense1d = self.sensitivity.calculate_sensitivity_1d(thermal=self.tn, sample=self.sv) #default: only thermal
        self.sense1d_k=self.sensitivity.k1d
        self.sense1d_P=sense1d

    def sense_2d(self):
        sense2d = self.sensitivity.calculate_sensitivity_2d(thermal=self.tn, sample=self.sv) # power_thermal = sensitivity.calculate_sensitivity_1d(thermal=tn, sample=sv)#only thermal
        self.sensitivity.plot_sense_2d(sense2d,plttitle="2d sense case: CHORD-like layout, default cyl k-bins",savename="CHORD_sens_default_k.png")
        kperp_keys=sorted(sense2d.keys())
        self.sense2d_kperp=np.array([k.value for k in kperp_keys]) # keys = sorted(sense2d.keys()); x = np.array([v.value for v in keys])
        self.sense2d_kpar= self.sensitivity.observation.kparallel
        self.sense2d_P=sense2d

def memo_ii_plotter(ensemble_of_spectra:np.ndarray,                       # indexed as (N_complexity_cases, N_k_perp, N_k_par)
                    ensemble_ids:np.ndarray,                              # names for each power spectrum quantity ("spectrum" for short, even though this is a misnomer in the case of ratios and residuals) in the ensemble according to the number of fiducial and perturbed beam types (N_complexity_cases,)
                    colourmap,                                            # for imshowing each power spectrum quantity
                    plot_log:bool,                                        # plot absolute or log of the power spectrum quantity?
                    k_perp:np.ndarray, k_par:np.ndarray,                  # k-perp and k-par bins that anchor each plotted spectrum
                    case_title:str, case_units:str,                       # title describing this power spectrum quantity and the corresponding units
                    save_name:str,                                        # name for the summary figure
                    norm_ext,                                             # if there is a physically motivated natural middle of the colour bar (e.g. 1 for a ratio or 0 for a residual), pass it to the plotter along with the extent of the range about this midpoint (possibly informed by the extent of the systematics you plugged into the simulation)
                    nu_ctr:float,                                         # only necessary if I insist on plotting the wedge 
                    k1_inset:float=0.06/u.Mpc, 
                    k2_inset:float=0.1/u.Mpc,
                    k3_inset:float=0.4/u.Mpc): #2.5/u.Mpc): # k-scales of interest to sample each spectrum in the ensemble
    N_spectra=len(ensemble_of_spectra)
    assert(N_spectra==len(ensemble_ids)), "mismatched number of spectra and spectrum names"
    Na=int(np.ceil(np.sqrt(N_spectra)))
    Nb=int(np.ceil(N_spectra/Na))
    if Na>Nb:
        N_LHS_rows=Nb
        N_LHS_cols=Na
    else:
        N_LHS_rows=Na
        N_LHS_cols=Nb
    k_perp=k_perp.to(1/u.Mpc)
    k_par=k_par.to(1/u.Mpc)
    cyl_extent=[k_perp[0].value,k_perp[-1].value,k_par[0].value,k_par[-1].value]
    k_perp_grid,k_par_grid=np.meshgrid(k_perp,k_par, indexing="ij")*k_par.unit
    k_mag_grid=np.sqrt(k_perp_grid**2+k_par_grid**2)
    values_of_k=np.zeros((N_spectra,3))

    fig = plt.figure(figsize=(N_LHS_cols*4, N_LHS_cols*3),layout="constrained")
    gs = gridspec.GridSpec(N_LHS_rows, N_LHS_cols+2, figure=fig)
    axs = [[fig.add_subplot(gs[row, col]) for col in range(N_LHS_cols)] for row in range(N_LHS_rows)] # grid for the left
    ax_right = fig.add_subplot(gs[:, N_LHS_cols:]) # summary holder on the right

    print("\n")
    for k in range(N_spectra):
        i=k//N_LHS_cols
        j=k%N_LHS_cols
        spec=ensemble_of_spectra[k,:,:] # remaining indices: N complexity cases, N k-perp, N k-par
        specshape=spec.shape
        spec_to_plot=np.copy(spec)
        if isinstance(spec_to_plot, Quantity):
            spec_to_plot_de_dimensionalized=spec_to_plot.value
            ensemble_of_spectra_de_dimensionalized=ensemble_of_spectra.value
        else:
            spec_to_plot_de_dimensionalized=spec_to_plot
            ensemble_of_spectra_de_dimensionalized=ensemble_of_spectra
        if plot_log:
            spec_to_plot=np.log10(spec_to_plot_de_dimensionalized)

            vminlog=np.log10(np.min(spec_to_plot_de_dimensionalized))
            if (type(norm_ext)==list):
                vminlog,vmaxlog=norm_ext
            if vminlog>0:
                vminlog=-0.01
            vmaxlog=np.log10(np.max(spec_to_plot_de_dimensionalized))
            if vmaxlog<0:
                vmaxlog=0.01
            norm=TwoSlopeNorm(0.,vmin=vminlog,
                                 vmax=vmaxlog)
        else:
            large=np.max(np.abs(ensemble_of_spectra_de_dimensionalized))
            half_middle=0.5*large # fallback: put all power spectra in the ensemble on the same colour scales, informed by the extreme range
            if norm_ext is None:
                norm_ext=half_middle # branch for absolute quantities: 
            if (type(norm_ext)==list):
                ne,vmax=norm_ext
                if ne<0:
                    ne=0.01*vmax
                elif ne==0:
                    ne=1e-9
                norm=SymLogNorm(ne,vmin=-vmax,vmax=vmax)
            else:
                ne=norm_ext
                if np.min(ensemble_of_spectra_de_dimensionalized)>=0:
                    norm=LogNorm(vmin=0.01*norm_ext,vmax=2*norm_ext)
                else:
                    if isinstance(ne, u.Quantity):
                        ne=ne.value
                    norm=SymLogNorm(0.01*ne,vmin=-ne,vmax=ne)
        
        im=axs[i][j].imshow(spec_to_plot.T, cmap=colourmap, origin="lower", extent=cyl_extent, norm=norm)
        xlims_to_use=axs[i][j].get_xlim()
        ylims_to_use=axs[i][j].get_ylim()
        axs[i][j].plot(k_perp,wedge_kpar(nu_ctr,k_perp),c="tab:green")
        axs[i][j].set_xlim(xlims_to_use)
        axs[i][j].set_ylim(ylims_to_use)
        axs[i][j].set_xlabel("k$_\perp$")
        axs[i][j].set_ylabel("k$_{||}$")
        axs[i][j].tick_params(axis='x', labelrotation=30)
        axs[i][j].set_title(ensemble_ids[k])
        axs[i][j].set_aspect("equal")
        if plot_log:
            neg_ticks = np.linspace(vminlog, 0., num=4, endpoint=False)
            pos_ticks = np.linspace(0., vmaxlog, num=4, endpoint=True)
            ticks = [*neg_ticks, 0, *pos_ticks]
            plt.colorbar(im,ax=axs[i][j],shrink=0.6,extend="both", ticks=ticks)
        else: 
            plt.colorbar(im,ax=axs[i][j],shrink=0.6,extend="both")

        idx_for_k1=np.argmin(np.abs(k_mag_grid-k1_inset))
        idx_for_k1=np.unravel_index(idx_for_k1,specshape)
        par_idx_for_k2=np.argmin(np.abs(k_par-k2_inset))
        idx_for_k3=np.argmin(np.abs(k_mag_grid-k3_inset))
        idx_for_k3=np.unravel_index(idx_for_k3,specshape)
        values_of_k[k]=[ spec[idx_for_k1], spec[0,par_idx_for_k2], spec[idx_for_k3] ]
        print("LIM review k=0.1:",spec[0,par_idx_for_k2])

    complexity_indices=np.arange(N_spectra)
    ax_right.scatter(complexity_indices,values_of_k[:,0],label=str(np.round(k1_inset,4))+" (~1st BAO wiggle scale)")
    ax_right.scatter(complexity_indices,values_of_k[:,1],label=str(np.round(k2_inset,4))+" (LIM review comparison scale)")
    ax_right.scatter(complexity_indices,values_of_k[:,2],label=str(np.round(k3_inset,4))+" (~CHIME scale)")
    ax_right.set_xticks(complexity_indices, labels=ensemble_ids, rotation=40)
    ax_right.set_xlabel("N CST types, N pointing errors")
    ax_right.set_ylabel("power spectrum quantity "+case_units)
    ax_right.set_title("insets for k closest to...")
    ax_right.legend()

    plt.suptitle("ingredients of this power spectrum quantity: "+case_title)
    plt.savefig(save_name+".png",dpi=250)
    plt.close()

def save_args_to_file(frame:str, filepath:str="settings.json"):
    args, _, _, values = inspect.getargvalues(frame)
    settings = {arg: values[arg] for arg in args}
    with open(filepath, "w") as f:
        json.dump(settings, f, indent=2, default=str)

def get_f_types_prefacs(cases):
    f_types_prefacs=[] # ends up as a ragged array in the general case, so list of lists is generally better. this is so small and quick of a calculation that I don't care about it being slow or stylistically questionable
    for case in cases:
        Nft,_=case
        if Nft==1:
            term= [1.]
        else:
            term= np.linspace(0.95,1.05,Nft)
        f_types_prefacs.append(term)
    return f_types_prefacs

def pointing_family(original_pointing,N,seed=270426):
    rng=np.random.default_rng(seed)
    orig_norm=np.linalg.norm(original_pointing)
    unscaled_pointings=rng.normal(size=(N,3))
    unscaled_norms=    np.linalg.norm(unscaled_pointings,axis=1,keepdims=True)
    rescaled_pointings=unscaled_pointings/unscaled_norms*orig_norm
    return rescaled_pointings

def power_comparison_plots(redo_window_calc:bool=False, redo_box_calc:bool=False,
              array_version:str="pathfinder", nu_ctr:float=800, epsxy:float=0.1,
              frac_tol_conv=0.1, N_th_k=1024,
              N_pbws_pert=0, antenna_dist="random", 
              which_power="P",
                  
              wedge_cut=False, layer_foregrounds=True, pointing_errors=[0.,0.,0.],
                  
              freq_bin_width=0.1953125*u.MHz,

              CST_lo=None,CST_hi=None,CST_deltanu=None,
              beam_sim_directory=None,f_mid1="pol1/f_",f_mid2="pol2/f_",f_tail="_GHz.txt",
              CST_f_head_fidu="farfield_(f=",CST_f_head_syst="farfield_(f=",
              
              from_incomplete_MC=False,
              contaminant_or_window=None, k_idx_for_window=0,
              isolated=False,seed=None):
    save_args_to_file(inspect.currentframe())

    ############################## other survey management factors ########################################################################################################################
    nu_ctr=nu_ctr.to(u.MHz)
    nu_ctr_Hz=nu_ctr.to(u.Hz)
    wl_ctr_m=c/nu_ctr_Hz
    wl_ctr_m=wl_ctr_m.decompose()

    ############################## baselines and beams ########################################################################################################################
    b_NS_CHORD=8.5*u.m
    N_NS_CHORD=24
    b_EW_CHORD=6.3*u.m
    N_EW_CHORD=22
    bminCHORD=np.min([b_NS_CHORD.value,b_EW_CHORD.value])*u.m.decompose() # force astropy to simplify 1/Hz * 1/s

    if (array_version=="pathfinder"): # 10x7=70 antennas (64 w/ gaps for receiver huts and site geometry constraints), 123 baselines
        bmaxCHORD=np.sqrt((b_NS_CHORD*10)**2+(b_EW_CHORD*7)**2) # pathfinder (as per the CHORD-all telecon on May 26th, but without holes)
    elif array_version=="full": # 24x22=528 antennas (512 w/ receiver hut gaps), 1010 baselines
        bmaxCHORD=np.sqrt((b_NS_CHORD*N_NS_CHORD)**2+(b_EW_CHORD*N_EW_CHORD)**2)
    else:
        raise ValueError("unknown array layout (not pathfinder or full)")

    ############################## pipeline administration ########################################################################################################################
    if contaminant_or_window is not None:
        c_or_w="wind"
    else:
        c_or_w="cont"
    per_chan_syst_string="none"
    per_chan_syst_name=""
    antenna_dist_string="rand"
    if antenna_dist=="corner":
        antenna_dist_string="corn"
    elif antenna_dist=="column":
        antenna_dist_string="rwcl"
    elif antenna_dist=="frame":
        antenna_dist_string="frme"
    elif antenna_dist!="random":
        raise ValueError("unknown antenna_dist")

    if type(CST_f_head_syst)==str: # make even the single-CST-type case iterable
        CST_f_head_syst=[CST_f_head_syst]
    assert (type(CST_f_head_syst)==np.ndarray or type(CST_f_head_syst)==list) and type(CST_f_head_syst[0])==str
    N_CST_types=len(CST_f_head_syst)
    if (len(pointing_errors)==3 and type(pointing_errors[0])==int):
        pointing_errors=[pointing_errors] # associate one pointing error with one CST case
    if N_CST_types>1: # length-M list of length-N_m lists of length-3 lists
        N_max_pointing_errors_each_CST=[len(pter_per_CST) for pter_per_CST in pointing_errors]
    else: # pointing_errors is a list of three-element lists
        N_max_pointing_errors_each_CST=[len(pointing_errors)]
    N_pointing_errors_each_CST=[np.arange(0,N_max_pt_er+1) for N_max_pt_er in N_max_pointing_errors_each_CST]
    complexity_cases=[]
    for a in range(1,N_CST_types+1):
        for b in range(N_max_pointing_errors_each_CST[a-1]+1):
            point=N_pointing_errors_each_CST[a-1][b]
            complexity_cases.append([a,point])

    complexity_ids=[str(case) for case in complexity_cases]

    power_quantities_all=[]
    Delta2_quantities_all=[]
    for i,complexity_type in enumerate(complexity_cases):
        print("\n\n\nabout to handle complexity case",complexity_type)
        t00=time.time()
        if N_CST_types==0:
            N_fidu_types_i,N_pert_types_i=complexity_type
            if N_pert_types_i==0: # loop over complexity cases–friendly number of antennas with perturbed beams
                N_pbws_pert_i=0
            else:
                N_pbws_pert_i=N_pbws_pert
            complexity_part="Nreal_"+str(N_fidu_types_i)+"__"\
                            "Npert_"+str(N_pert_types_i)+"_"+str(N_pbws_pert)+"__"\
                            "epsxy_"+str(epsxy)+"__"
            related_to_N_of_types={"N_fidu_types":N_fidu_types_i,"N_pert_types":N_pert_types_i}
        else:
            NCST_i,Npoint_i=complexity_type
            related_to_N_of_types={} # this info comes from unpacking in this mode now
            complexity_id_i=str(complexity_type)
            complexity_part="N_CST_types_"+str(NCST_i)+"__"+"N_ptg_err_"+str(Npoint_i)
            if Npoint_i>0:
                pointing_errors_i=pointing_errors[NCST_i-1][:Npoint_i] # the non-+1 version is actually fine because the indices are zero-based
            else:
                pointing_errors_i=[[0.,0.,0.,]]

            CST_f_head_syst_i=CST_f_head_syst[:NCST_i]
            N_pbws_pert_i=N_pbws_pert
        
        ioname=array_version+"_"+c_or_w+"_"+"_"\
           ""+per_chan_syst_string+"_"+per_chan_syst_name+"_"\
           ""+str(int(nu_ctr.value))+"MHz__"+complexity_part+"_"\
           "dist_"+antenna_dist_string+"__"\
           "layer_"+str(layer_foregrounds)+"__"\
           "wedge_"+str(wedge_cut)+"__"\
           "seed_"+str(seed)
        
        if (complexity_type!=4 and antenna_dist=="corner"):
            continue

        # PIPELINE ADMIN FOR THIS PA SYSTEMATIC PERMUTATION
        antdist=antenna_dist
        pointing_errors_to_use_i=pointing_errors_i
        if Npoint_i==0:
            if NCST_i==1:
                CST_f_head_syst_i=[CST_f_head_fidu] # literally just use fiducial for both num and denom everywhere
            else:
                pointing_errors_to_use_i=[[0.,0.,0.,]]                # use the syst beams, but don't apply any pointing errors
        windowed_survey=beam_effects(# SCIENCE
                                    # the observation
                                    bminCHORD,bmaxCHORD,                                                       
                                    nu_ctr,freq_bin_width,                                                 
                                    evol_restriction_threshold=def_evol_restriction_threshold,           
                                        
                                    # beam generalities
                                    beam_domain=None,                              

                                    # numerical beam perturbation parameters
                                    N_pbws_pert=N_pbws_pert_i,
                                    antenna_distribution=antdist,array_version=array_version,
                                    **related_to_N_of_types,
                                    CST_lo=CST_lo,CST_hi=CST_hi,CST_deltanu=CST_deltanu,ioname=ioname,
                                    beam_sim_directory=beam_sim_directory,f_mid1=f_mid1,f_mid2=f_mid2,f_tail=f_tail,
                                    CST_f_head_fidu=CST_f_head_fidu,CST_f_head_syst=CST_f_head_syst_i,
                                    pointing_errors=pointing_errors_to_use_i,

                                    # FORECASTING
                                    P_fid_for_cont_pwr=contaminant_or_window, k_idx_for_window=k_idx_for_window,
                                    wedge_cut=wedge_cut, layer_foregrounds=layer_foregrounds,

                                    # NUMERICAL 
                                    N_theory_k=N_th_k,                                        
                                    init_and_box_tol=0.05,CAMB_tol=0.05,                                 
                                    frac_tol_conv=frac_tol_conv,seed=seed,                                         
                                    ftol_deriv=1e-16,maxiter=5,   
                                    LoS_taper=True,image_taper=False,
                                    # LoS_taper=False,image_taper=False,        

                                    # CONVENIENCE
                                    heavy_beam_recalc=redo_box_calc                                                 
                                    
                                    )
        
        recalc_co_fi_xx_fg=False
        recalc_co_fi_sy_fg=False
        recalc_xx_fi_sy_fg=False
        recalc_xx_fi_xx_fg=False
        recalc_co_fi_xx_xx=False
        recalc_co_fi_sy_xx=False
        recalc_co_xx_xx_fg=False
        if isolated==False:
            recalc_co_fi_xx_fg=True
            recalc_co_fi_sy_fg=True
            recalc_xx_fi_sy_fg=True
            recalc_xx_fi_xx_fg=True
            recalc_co_fi_xx_xx=True
            recalc_co_fi_sy_xx=True
            recalc_co_xx_xx_fg=True
        
        elif isolated=="co_fi_xx_fg":
            recalc_co_fi_xx_fg=True
        elif isolated=="co_fi_sy_fg":
            recalc_co_fi_sy_fg=True
        elif isolated=="xx_fi_sy_fg":
            recalc_xx_fi_sy_fg=True
        elif isolated=="xx_fi_xx_fg":
            recalc_xx_fi_xx_fg=True
        elif isolated=="co_fi_xx_xx":
            recalc_co_fi_xx_xx=True
        elif isolated=="co_fi_sy_xx":
            recalc_co_fi_sy_xx=True
        elif isolated=="co_xx_xx_fg":
            recalc_co_xx_xx_fg=True

        print("about to perform or load Monte Carlos")
        P_unit=u.mK**2 *windowed_survey.Deltabox_xy.unit**3
        if not from_incomplete_MC:
            if redo_window_calc:
                t0=time.time()
                windowed_survey.calc_power_contamination(isolated=isolated) # loops over complexity
                P_co_xx_xx_xx=windowed_survey.P_co_xx_xx_xx
                np.save("P_co_xx_xx_xx_"+ioname+".npy",P_co_xx_xx_xx.value)
                P_xx_xx_xx_fg=windowed_survey.P_xx_xx_xx_fg
                np.save("P_xx_xx_xx_fg_"+ioname+".npy",P_xx_xx_xx_fg.value)
                t1=time.time()
                print("Pcont calculation time was",t1-t0)

                if recalc_co_fi_xx_fg:
                    P_co_fi_xx_fg=windowed_survey.P_co_fi_xx_fg
                    np.save("P_co_fi_xx_fg_"+ioname+".npy",P_co_fi_xx_fg.value)
                if recalc_co_fi_sy_fg:
                    P_co_fi_sy_fg=windowed_survey.P_co_fi_sy_fg
                    np.save("P_co_fi_sy_fg_"+ioname+".npy",P_co_fi_sy_fg.value)
                if recalc_xx_fi_sy_fg:
                    P_xx_fi_sy_fg=windowed_survey.P_xx_fi_sy_fg
                    np.save("P_xx_fi_sy_fg_"+ioname+".npy",P_xx_fi_sy_fg.value)
                if recalc_xx_fi_xx_fg:
                    P_xx_fi_xx_fg=windowed_survey.P_xx_fi_xx_fg
                    np.save("P_xx_fi_xx_fg_"+ioname+".npy",P_xx_fi_xx_fg.value)
                if recalc_co_fi_xx_xx:
                    P_co_fi_xx_xx=windowed_survey.P_co_fi_xx_xx
                    np.save("P_co_fi_xx_xx_"+ioname+".npy",P_co_fi_xx_xx.value)
                if recalc_co_fi_sy_xx:
                    P_co_fi_sy_xx=windowed_survey.P_co_fi_sy_xx
                    np.save("P_co_fi_sy_xx_"+ioname+".npy",P_co_fi_sy_xx.value)
                if recalc_co_xx_xx_fg:
                    P_co_xx_xx_fg=windowed_survey.P_co_xx_xx_fg
                    np.save("P_co_xx_xx_fg_"+ioname+".npy",P_co_xx_xx_fg.value)

                P_CO_XX_XX_XX=windowed_survey.P_CO_XX_XX_XX
                np.save("P_CO_XX_XX_XX__"+ioname+".npy",P_CO_XX_XX_XX.value)

                N_per_realization=windowed_survey.N_per_realization
                np.save("N_per_realization_"+ioname+".npy",N_per_realization)
                kperp_internal=windowed_survey.kperpbins_internal[:-1]
                kpar_internal=windowed_survey.kparbins_internal[:-1]
                np.save("kpar_internal_"+ioname+".npy",kpar_internal.value)
                np.save("kperp_internal_"+ioname+".npy",kperp_internal.value)
                if isolated is not False: # break early if you just calculate one windowed power spectrum at a time
                    return None

            else:
                P_co_fi_xx_fg=np.load("P_co_fi_xx_fg_"+ioname+".npy")*P_unit
                P_co_fi_sy_fg=np.load("P_co_fi_sy_fg_"+ioname+".npy")*P_unit
                P_xx_fi_sy_fg=np.load("P_xx_fi_sy_fg_"+ioname+".npy")*P_unit
                P_xx_fi_xx_fg=np.load("P_xx_fi_xx_fg_"+ioname+".npy")*P_unit
                P_co_fi_xx_xx=np.load("P_co_fi_xx_xx_"+ioname+".npy")*P_unit
                P_co_fi_sy_xx=np.load("P_co_fi_sy_xx_"+ioname+".npy")*P_unit
                P_co_xx_xx_fg=np.load("P_co_xx_xx_fg_"+ioname+".npy")*P_unit

                P_co_xx_xx_xx=np.load("P_co_xx_xx_xx_"+ioname+".npy")*P_unit
                P_xx_xx_xx_fg=np.load("P_xx_xx_xx_fg_"+ioname+".npy")*P_unit
                P_CO_XX_XX_XX=np.load("P_CO_XX_XX_XX__"+ioname+".npy")*P_unit
                
                N_per_realization=np.load("N_per_realization_"+ioname+".npy")
                kpar_internal=np.load("kpar_internal_"+ioname+".npy")/u.Mpc # units also by construction
                kperp_internal=np.load("kperp_internal_"+ioname+".npy")/u.Mpc
        else:
            P_co_fi_xx_fg=np.load("P_co_fi_xx_fg_MC_incomplete.npy")*P_unit
            P_co_fi_sy_fg=np.load("P_co_fi_sy_fg_MC_incomplete.npy")*P_unit
            P_xx_fi_sy_fg=np.load("P_xx_fi_sy_fg_MC_incomplete.npy")*P_unit
            P_xx_fi_xx_fg=np.load("P_xx_fi_xx_fg_MC_incomplete.npy")*P_unit
            P_co_fi_xx_xx=np.load("P_co_fi_xx_xx_MC_incomplete.npy")*P_unit
            P_co_fi_sy_xx=np.load("P_co_fi_sy_xx_MC_incomplete.npy")*P_unit
            P_co_xx_xx_fg=np.load("P_co_xx_xx_fg_MC_incomplete.npy")*P_unit

            P_co_xx_xx_xx=np.load("P_co_xx_xx_xx_MC_incomplete.npy")*P_unit
            P_xx_xx_xx_fg=np.load("P_xx_xx_xx_fg_MC_incomplete.npy")*P_unit
            P_CO_XX_XX_XX=np.load("P_CO_XX_XX_XX__MC_incomplete.npy")*P_unit
            
            N_per_realization=np.load("N_per_realization_MC_incomplete.npy")
            kpar_internal=np.load("kpar_internal_"+ioname+".npy")/u.Mpc # units by construction if importing from same pipeline generation
            kperp_internal=np.load("kperp_internal_"+ioname+".npy")/u.Mpc

        Presidual= P_co_fi_sy_fg-P_co_fi_xx_fg
        Pratio=    P_xx_fi_sy_fg/P_co_xx_xx_xx
        Pisoratio= P_xx_fi_xx_fg/P_co_xx_xx_xx
        assert(Pratio.unit.physical_type=="dimensionless" and Pisoratio.unit.physical_type=="dimensionless")
        co_xx_xx_fg_lin=( P_co_xx_xx_fg - P_co_xx_xx_xx - P_xx_xx_xx_fg ).value /P_co_xx_xx_fg.value
        co_fi_xx_fg_lin=( P_co_fi_xx_fg - P_co_fi_xx_xx - P_xx_fi_xx_fg ).value /P_co_fi_xx_fg.value
        co_fi_sy_fg_lin=( P_co_fi_sy_fg - P_co_fi_sy_xx - P_xx_fi_sy_fg ).value /P_co_fi_sy_fg.value
        co__divby__fg=P_co_xx_xx_xx/P_xx_xx_xx_fg

        k_perp_grid,k_par_grid=np.meshgrid(kperp_internal,kpar_internal, indexing="ij")
        k_mag_grid=np.sqrt(k_perp_grid**2+k_par_grid**2)

        power_quantities_this_complexity=np.array([P_xx_fi_sy_fg.value,  P_co_fi_xx_fg.value, P_co_fi_sy_fg.value, Presidual.value,     Pratio,        
                                                   P_xx_xx_xx_fg.value,  Pisoratio,           P_co_xx_xx_xx.value, P_co_fi_xx_xx.value, P_co_fi_sy_xx.value, 
                                                   P_co_xx_xx_fg.value,                                              P_xx_fi_xx_fg.value, P_CO_XX_XX_XX.value,
                                                   co_xx_xx_fg_lin,      co_fi_xx_fg_lin,     co_fi_sy_fg_lin,     co__divby__fg  ,
                                                   P_co_fi_sy_fg/P_co_fi_xx_fg, P_co_fi_sy_xx-P_co_fi_xx_xx]) # N_pspec_types x Nkperp x Nkpar
        power_quantities_all.append(power_quantities_this_complexity) # N_complexity_cases x N_pspec_types x Nkperp x Nkpar
        
        Delta2_quantities_this_complexity=[P_qty*k_mag_grid**3/(2*pi**2) for P_qty in power_quantities_this_complexity]
        Delta2_quantities_all.append(Delta2_quantities_this_complexity)
        t01=time.time()
        print("handled complexity case",complexity_id_i,"in",t01-t00,"s")

    power_quantities_all=np.asarray(power_quantities_all)
    Delta2_quantities_all=np.asarray(Delta2_quantities_all)
    N_plots=power_quantities_this_complexity.shape[0]
    abs_map=cmasher.voltage # also consider rainforest, fall, ...others
    rel_map=cmasher.prinsenvlag

    if which_power=="P":
        absolute_units="(mK$^2$ Mpc$^3$)"
    elif which_power=="Delta2":
        absolute_units="(mK$^2$)"
    relative_units="(unitless)"

    ###    ###   ###    ###   ###    ###   ###    ###   ###    ###   ###    ###   ###    ###   ###    ###   ###    ###   ###    ###   ###    ###   ###    ###   
    abs_co_no_fg_indices=np.r_[7,8,9,12]
    abs_co_fg_indices=np.r_[1,2,10]
    abs_co_beam_indices=np.r_[8,9]
    abs_co_indices=np.r_[7,12]

    abs_residual=[np.percentile(Presidual.value,90),
                    np.max(np.abs(Presidual.value))]
    coxxxxfg_lin=[np.percentile(co_xx_xx_fg_lin,90),
                    np.max(np.abs(co_xx_xx_fg_lin))]
    cofixxfg_lin=[np.percentile(co_fi_xx_fg_lin,90),
                    np.max(np.abs(co_fi_xx_fg_lin))]
    cofisyfg_lin=[np.percentile(co_fi_sy_fg_lin,90),
                    np.max(np.abs(co_fi_sy_fg_lin))]
    co_d_fg=[np.min(np.log10(co__divby__fg)),
             np.percentile(np.log10(co__divby__fg),98)]
    fgext=None
    if which_power=="P":
        abs_co_no_fg=np.percentile(power_quantities_all[:,abs_co_no_fg_indices,:,:],98) 
        abs_co_beam=np.percentile(power_quantities_all[:,abs_co_beam_indices,:,:],98)
        abs_co_fg=np.percentile(power_quantities_all[:,abs_co_fg_indices,:,:],90)
        abs_co=np.percentile(power_quantities_all[:,abs_co_indices,:,:],90)
        # fgext=np.percentile(P_xx_xx_xx_fg.value,97)
    elif which_power=="Delta2":
        abs_co_no_fg=None
        abs_co_fg=np.percentile(Delta2_quantities_all[:,abs_co_fg_indices,:,:],90) # good for whole dynamic range
        # fgext=np.percentile(Delta2_quantities_all[:,5,:,:],97)
        abs_residual=[np.percentile(Delta2_quantities_all[:,3,:,:],90),
                      np.max(np.abs(Delta2_quantities_all[:,3,:,:]))]
        coxxxxfg_lin=[np.percentile(Delta2_quantities_all[:,-6,:,:],90),
                      np.max(np.abs(Delta2_quantities_all[:,-6,:,:]))]
        cofixxfg_lin=[np.percentile(Delta2_quantities_all[:,-5,:,:],90),
                      np.max(np.abs(Delta2_quantities_all[:,-5,:,:]))]
        cofisyfg_lin=[np.percentile(Delta2_quantities_all[:,-4,:,:],90),
                      np.max(np.abs(Delta2_quantities_all[:,-4,:,:]))]
        co_d_fg=[np.min(np.log10(Delta2_quantities_all[:,-3,:,:])),
                 np.percentile(np.abs(np.log10(Delta2_quantities_all[:,-3,:,:])),98)]

    co_fi_sy_fg_str="cosmo + fidu beam + syst + fg"
    co_fi_xx_fg_str="cosmo + fidu beam + fg"

    # vers_name, units, save_name, norm_ext, cmap, plotlog
    xx_fi_sy_fg_params=                       ["log10[ fidu beam + syst + fg ]",
                                                absolute_units, 
                                               "fidu_syst_fg",
                                                fgext,
                                                abs_map,
                                                True]
    
    co_fi_xx_fg_params=                       [ co_fi_xx_fg_str,
                                                absolute_units,
                                               "cosmo_fidu_fg",
                                                abs_co_fg,
                                                abs_map,
                                                False]
    
    co_fi_sy_fg_params=                       [ co_fi_sy_fg_str,
                                                absolute_units,
                                               "cosmo_fidu_syst_fg",
                                                abs_co_fg,
                                                abs_map,
                                                False]
    
    Presidual_params=                         ["("+co_fi_sy_fg_str+") - ("+co_fi_xx_fg_str+")",
                                                absolute_units,
                                               "cosmo_fidu_syst_fg__minus__cosmo_fidu_fg",
                                                abs_residual,
                                                rel_map,
                                                False]
    
    Pratio_params=                            ["log10[ (fidu beam + syst + fg) / cosmo ]", 
                                                relative_units,
                                               "fidu_syst_fg__divby__cosmo",
                                                None,
                                                rel_map,
                                                True]
    
    xx_xx_xx_fg_params=                       ["log10[ fg ]",                                
                                                absolute_units,
                                               "fg",
                                                fgext,
                                                abs_map,
                                                True]
    
    isoratio_params=                          ["log10[ (fidu beam + fg) / cosmo ]",
                                                relative_units,
                                               "fidu_fg__divby__cosmo",
                                                None,
                                                rel_map,
                                                True]
    
    co_xx_xx_xx_params=                       ["cosmo",
                                                absolute_units,
                                               "cosmo",
                                                # abs_co_no_fg,
                                                abs_co,
                                                abs_map,
                                                False]
    
    co_fi_xx_xx_params=                       ["cosmo + fidu beam",
                                                absolute_units,
                                               "cosmo_fidu",
                                                # abs_co_no_fg,
                                                abs_co_beam,
                                                abs_map,
                                                False]
    
    co_fi_sy_xx_params=                       ["cosmo + fidu beam + syst",
                                                absolute_units,
                                               "cosmo_fidu_syst",
                                                # abs_co_no_fg,
                                                abs_co_beam,
                                                abs_map,
                                                False]
    
    co_xx_xx_fg_params=                       ["cosmo + fg",
                                                absolute_units,
                                               "cosmo_fg",
                                                abs_co_fg,
                                                abs_map,
                                                False]
    
    xx_fi_xx_fg_params=                       ["log10[ fidu beam + fg ]",
                                                absolute_units,
                                               "fidu_fg",
                                                fgext,
                                                abs_map,
                                                True]
    
    P_CO_XX_XX_XX_params=                     ["COSMO",
                                                absolute_units,
                                               "COSMOCOSMO",
                                                # abs_co_no_fg,
                                                abs_co,
                                                abs_map,
                                                False]
    
    co_xx_xx_fg_lin_params=                   ["cosmo–fg linearity frac dif",
                                                relative_units,
                                               "cosmo_fg_linearity",
                                                coxxxxfg_lin,
                                                rel_map,
                                                False]
    
    co_fi_xx_fg_lin_params=                   ["cosmo–fidu beam–fg linearity frac dif",
                                                relative_units,
                                               "cosmo_fidu_fg_linearity",
                                                cofixxfg_lin,
                                                rel_map,
                                                False]
    
    co_fi_sy_fg_lin_params=                   ["all linearity frac dif",
                                                relative_units, 
                                               "all_linearity",
                                                cofisyfg_lin,
                                                rel_map,
                                                False]
    
    co__divby__fg_params=                     ["log10[ cosmo / fg ]",
                                                relative_units,
                                               "cosmo__divby__fg",
                                                co_d_fg,
                                                rel_map,
                                                True]
    
    co_fi_sy_fg__divby__P_co_fi_xx_fg_params= ["( "+co_fi_sy_fg_str+") / ("+co_fi_xx_fg_str+" )",
                                                relative_units,
                                               "cosmo_fidu_syst_fg__divby__cosmo_fidu_fg",
                                                None,
                                                rel_map,
                                                True]
    
    co_fi_sy_xx__minus__co_fi_xx_xx_params=   ["( cosmo + fidu beam + syst ) - ( cosmo + fidu beam )",
                                                relative_units,
                                               "cosmo_fidu_syst__minus__cosmo_fidu",
                                                None,
                                                rel_map,
                                                False]

    ensemble_of_plot_params=[xx_fi_sy_fg_params,                    co_fi_xx_fg_params,     co_fi_sy_fg_params,                       
                             Presidual_params,                      Pratio_params,          xx_xx_xx_fg_params,     
                             isoratio_params,                       co_xx_xx_xx_params,     co_fi_xx_xx_params,   co_fi_sy_xx_params, 
                             co_xx_xx_fg_params,                    xx_fi_xx_fg_params,     P_CO_XX_XX_XX_params, co_xx_xx_fg_lin_params, 
                             co_fi_xx_fg_lin_params,                co_fi_sy_fg_lin_params, co__divby__fg_params, co_fi_sy_fg__divby__P_co_fi_xx_fg_params, 
                             co_fi_sy_xx__minus__co_fi_xx_xx_params]
    ###    ###   ###    ###   ###    ###   ###    ###   ###    ###   ###    ###   ###    ###   ###    ###   ###    ###   ###    ###   ###    ###   ###    ###   

    print("\n\n")
    if which_power=="P":
        power_quantities_all_correct_type=power_quantities_all
    elif which_power=="Delta2":
        power_quantities_all_correct_type=Delta2_quantities_all
    for i in range(N_plots): # iterate over plot cases
        power_quantity_this_plot_case=power_quantities_all_correct_type[:,i,:,:] # [:,i,:,:] = all complexity cases, ith power spectrum quantity, all kperps, all kpars
        plot_params_i=ensemble_of_plot_params[i]
        vers_name, units, save_name, norm_ext, cmap, plotlog = plot_params_i
        memo_ii_plotter(power_quantity_this_plot_case, complexity_ids, cmap, plotlog,
                        kperp_internal, kpar_internal, 
                        vers_name, units, save_name, norm_ext, nu_ctr)
        print("plotted ",vers_name)