import numpy as np
from astropy import units as u
from matplotlib import pyplot as plt
from matplotlib.colors import CenteredNorm
from forecasting_pipeline import simulate_array

omega_e=(np.pi*u.rad)/(12*u.hr)

def fringe_rate(H=12*u.hourangle, delta=0*u.hourangle, # source hr angle and dec (defaults to zenith for drift-scan)
                h=0*u.hourangle, d=49*u.deg,           # baseline-at-current-pointing hr angle and dec
                D=100*u.m,                             # baseline length
                obsfreq=600*u.MHz,                     # obs freq (default = 600 MHz)
                f_r_instr=0                            # instrumental clock contrib to fringe rate
                ):
    lambdaobs=3e8*u.m/u.s/obsfreq
    D_lambda=D/lambdaobs
    D_lambda=D_lambda.decompose()
    f_r=-omega_e*D_lambda\
        *np.cos(d.to(u.rad))\
        *np.cos(delta.to(u.rad))\
        *np.sin((H-h).to(u.rad))\
        +f_r_instr
    return f_r.decompose()

Ndec=50
Nbl=75
Nfreq=5

pointing_decs=np.linspace(19,79,Ndec)*u.deg
baseline_lengths=np.linspace(6.3,246.6,Nbl)*u.m
obs_freqs=np.linspace(300,1500,Nfreq,endpoint=True)*u.MHz

fringe_rates=np.zeros((Ndec,Nbl,Nfreq))
for i,obsfreq in enumerate(obs_freqs):
    for j,pointing_dec in enumerate(pointing_decs):
        fringe_rates[j,:,i]=fringe_rate(d=pointing_dec,
                                        D=baseline_lengths,
                                        obsfreq=obsfreq)

fig,axs=plt.subplots(1,Nfreq,figsize=(15,5),sharey=True)
for i,obsfreq in enumerate(obs_freqs):
    im=axs[i].imshow(fringe_rates[:,:,i].T,
                  origin="lower",
                  extent=[pointing_decs[0].value,    pointing_decs[-1].value,
                          baseline_lengths[0].value, baseline_lengths[-1].value],
                  cmap="RdBu",
                  norm=CenteredNorm(vcenter=np.mean(fringe_rates)))
    axs[i].set_xlabel("pointing dec (º)")
    axs[i].set_title("ν={} MHz".format(obsfreq.value))
axs[0].set_ylabel("baseline length (m)")
fig.colorbar(im, ax=axs.ravel(), orientation='vertical', label="rad s$^{-1}$")
plt.suptitle("fringe rate envelope")
plt.savefig("fringe_rate_envelope.png")


obs_decs=np.array([19,49,79])*u.deg
nu_ctrs=np.array([300,600,900,1200,1500])*u.MHz
summary=np.zeros((15,4))
k=0
for obs_dec in obs_decs:
    for nu_ctr in nu_ctrs:
        simulated_array=simulate_array(observing_dec=obs_dec,
                                       nu_ctr=nu_ctr)
        uv=simulated_array.uv_synth
        max_abs_u=np.max(np.abs(uv))
        min_theta_mag=1/max_abs_u
        summary[k,:]=[obs_dec.value,nu_ctr.value,max_abs_u,min_theta_mag]
        k+=1
np.savetxt("max_abs_u_summary.txt",summary)