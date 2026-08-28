from forecasting_pipeline import *
from matplotlib.colors import LogNorm
fourpi=4*np.pi

kperpmin,kperpmax=np.array([0.005, 0.1])/u.Mpc # N even
# kperpmin,kperpmax=np.array([0.006, 0.1])/u.Mpc # N odd
kparmin, kparmax= np.array([0.004,0.2])/u.Mpc # N even
# kparmin, kparmax= np.array([0.003,0.2])/u.Mpc # N odd

rt=True # True or None
it=True

kmin=np.sqrt(kperpmin**2+kparmin**2)
kmax=np.sqrt(kperpmax**2+kparmax**2)

Lxy=twopi/kperpmin
Nxy=int(Lxy*kperpmax/pi)
Lz=twopi/kparmin
Nz=int(Lz*kparmax/pi)
print("Nxy,Nz=",Nxy,Nz)

N_in=1024
k_in=np.linspace(0.25*kmin,np.sqrt(3)*kmax,N_in)/u.Mpc
dec_pwr=k_in.value**-4.1
dec_pwr_norm=np.mean(dec_pwr)
inc_pwr=k_in.value**1.8 
inc_pwr_norm=np.mean(inc_pwr)
power_unit=u.mK**2*u.Mpc**3

P_in_options=[[np.ones(N_in)        *power_unit, "flat"              ],
              [dec_pwr/dec_pwr_norm *power_unit, "decaying_power_law"],
              [inc_pwr/inc_pwr_norm *power_unit, "growing_power_law" ] ]
# P_in_options=[[np.ones(N_in)        *power_unit, "flat"              ]]
ft=0.05

for P_case in P_in_options:
    P_in,P_name=P_case
    stats_container=cosmo_stats(Lxy,Lz=Lz,
                                P_fid=P_in,k_fid=k_in,
                                Nxy=Nxy,Nz=Nz,
                                Nkpar=None,
                                LoS_apo=rt,transverse_apo=it,
                                frac_tol=ft, nu_ctr=600.*u.MHz) # appease the FoG assertion w/ ctr freq but FoG is turned off for now
    stats_container.power_Monte_Carlo()
    print("completed Monte Carlo")

    k_out_b=stats_container.kperpbins[:-1]
    half_width=0.5*(k_out_b[1]-k_out_b[0])
    k_out_b+=half_width
    P_out_b=stats_container.P_binned_MC_complete
    # print("k_out_b.shape, P_out_b.shape=",k_out_b.shape, P_out_b.shape)
    flat_shape=(Nxy**2*Nz,)
    k_out_u=np.reshape(stats_container.kmag_grid_centre_flat,flat_shape)
    P_unbinned_MC=stats_container.P_unbinned_MC_complete
    P_out_u=np.reshape(stats_container.P_unbinned_MC_complete, flat_shape)

    T_slice=stats_container.T_pristine[0,:,:]
    plt.figure()
    plt.imshow(T_slice.value.T,origin="lower")
    plt.savefig("T_"+P_name+".png")
    plt.close()

    plt.figure(layout="constrained")
    plt.plot(   k_in,    P_in,    label="input power",         c="C0")
    plt.plot(   k_out_b, P_out_b, label="end-to-end binned",   c="C1", marker=".")
    plt.scatter(k_out_u, P_out_u, label="end-to-end unbinned", c="C2", s=0.5)
    if P_name=="flat":
        plt.ylim(2e-1,1.5)
    plt.yscale("log")
    plt.xlabel("k (1/Mpc)")
    plt.ylabel("P (mK^2 Mpc^3)")
    plt.title(P_name+" power end-to-end comparison")
    plt.legend(loc="lower left")
    plt.savefig("P_"+P_name+"_end_to_end.png")
    plt.close()

    print("P_out_b[0]=",P_out_b[0])
    print("np.mean(P_out_b[1:])=",np.mean(P_out_b[1:]))

    P_in,P_name=P_case
    print("Nxy,Nz=",Nxy,Nz)
    stats_container=cosmo_stats(Lxy,Lz=Lz,
                                P_fid=P_in,k_fid=k_in,
                                Nxy=Nxy,Nz=Nz,
                                LoS_apo=rt,transverse_apo=it,
                                frac_tol=ft, nu_ctr=600*u.MHz)
    stats_container.power_Monte_Carlo()
    print("completed Monte Carlo")

    k_perp_out=stats_container.kperpbins[:-1]
    k_par_out= stats_container.kparbins[:-1]
    cyl_extent=[k_perp_out[0].value,k_perp_out[-1].value,k_par_out[0].value,k_par_out[-1].value]
    P_out=stats_container.P_binned_MC_complete.value.T
    P_out_min=np.min(P_out)
    print("P_out_min=",P_out_min)
    P_out_to_plot= np.clip(P_out, a_min=1e-8, a_max=P_out.max())

    _,axs=plt.subplots(1,2,layout="constrained")
    im=axs[0].imshow(P_out_to_plot,extent=cyl_extent,origin="lower",norm=LogNorm(vmin=1e-8,vmax=P_out.max()))
    plt.colorbar(im,ax=axs[0])
    axs[0].set_title("Log-scaled")
    axs[0].axis("equal")
    im=axs[1].imshow(P_out,extent=cyl_extent,origin="lower")
    plt.colorbar(im,ax=axs[1])
    axs[1].set_title("Lin-scaled")
    axs[1].axis("equal")

    plt.suptitle(P_name+" power end-to-end comparison")
    plt.savefig("P_cyl_"+P_name+"_end_to_end.png",dpi=400)
    plt.close()

    print("np.mean(P_out[0,:])=",np.mean(P_out[0,:]))
    print("np.mean(P_out[:,0])=",np.mean(P_out[:,0]))
    print("np.mean(P_out[1:,:])=",np.mean(P_out[1:,:]))
    print("np.mean(P_out[:,1:])=",np.mean(P_out[:,1:]))