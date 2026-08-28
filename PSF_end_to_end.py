from forecasting_pipeline import *
from matplotlib.colors import LogNorm
fourpi=4*np.pi

Nxy=512
Nz=191
Lxy=13265.317*u.Mpc
Lz=319.469187*u.Mpc

N_in=1024
k_in=np.linspace(1e-4,0.5,N_in)/u.Mpc
dec_pwr=k_in.value**-4.1
dec_pwr_norm=np.mean(dec_pwr)
inc_pwr=k_in.value**1.8 
inc_pwr_norm=np.mean(inc_pwr)
power_unit=u.mK**2*u.Mpc**3

P_flat=np.ones(N_in)*power_unit

P_in_options=[
            #   [dec_pwr/dec_pwr_norm *power_unit, "decaying_power_law", None],
            #   [inc_pwr/inc_pwr_norm *power_unit, "growing_power_law", None ],
              [P_flat, "flat"              , CenteredNorm(vcenter=1,halfrange=0.1)] 

              ]
# P_in_options=[[np.ones(N_in)        *power_unit, "flat"              ]]
ft=0.5
PSF=np.load("fidu_box_PSF_full_cont__none__600MHz__N_CST_types_1__N_ptg_err_0_dist_hybr__layer_True__wedge_False__seed_None.npy")

forpower1=cosmo_stats(Lxy,Lz=Lz,
                      P_fid=P_flat,k_fid=k_in,
                      Nxy=Nxy,Nz=Nz)
forpower1.generate_GRF()
power1map=forpower1.T_pristine


for P_case in P_in_options:
    P_in,P_name,e2enorm=P_case
    print("Nxy,Nz=",Nxy,Nz)
    stats_container0=cosmo_stats(Lxy,Lz=Lz,
                                P_fid=P_in,k_fid=k_in,
                                Nxy=Nxy,Nz=Nz,
                                LoS_apo=True,transverse_apo=False,
                                frac_tol=ft, nu_ctr=600*u.MHz)
    stats_container1=cosmo_stats(Lxy,Lz=Lz,
                                 P_fid=P_in,k_fid=k_in,
                                 Nxy=Nxy,Nz=Nz,
                                 PSF=PSF,power1map=power1map,
                                 LoS_apo=True,transverse_apo=False,
                                 frac_tol=ft, nu_ctr=600*u.MHz)
    stats_container0.power_Monte_Carlo()
    stats_container1.power_Monte_Carlo()
    print("completed Monte Carlo")

    k_perp_out=stats_container0.kperpbins[:-1]
    k_par_out= stats_container0.kparbins[:-1]
    cyl_extent=[k_perp_out[0].value,k_perp_out[-1].value,k_par_out[0].value,k_par_out[-1].value]
    P0=stats_container0.P_binned_MC_complete.value.T
    P1=stats_container1.P_binned_MC_complete.value.T

    _,axs=plt.subplots(1,3,layout="constrained")
    im=axs[0].imshow(P0,extent=cyl_extent,origin="lower",norm=e2enorm)
    plt.colorbar(im,ax=axs[0])
    axs[0].set_title("E2E w/o PSF\nmean,median={%7.4f}, {%7.4f}".format(np.mean(P0),np.median(P0)))
    axs[0].axis("equal")
    im=axs[1].imshow(P1,extent=cyl_extent,origin="lower",norm=e2enorm)
    plt.colorbar(im,ax=axs[1])
    axs[1].set_title("E2E w/ PSF\non target scale\nmean={%7.4f}, {%7.4f}".format(np.mean(P1),np.median(P1)))
    axs[1].axis("equal")
    im=axs[2].imshow(P1,extent=cyl_extent,origin="lower",norm=CenteredNorm(vcenter=np.median(P1),halfrange=0.05*np.std(P1)))
    plt.colorbar(im,ax=axs[2])
    axs[2].set_title("E2E w/ PSF\non data scale\nmean={%7.4f}, {%7.4f}".format(np.mean(P1),np.median(P1)))
    axs[2].axis("equal")

    plt.suptitle(P_name+" power end-to-end comparison")
    plt.savefig("PSF_A_B_"+P_name+"_end_to_end.png",dpi=400)
    plt.close()