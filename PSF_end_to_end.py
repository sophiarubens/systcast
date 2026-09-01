from forecasting_pipeline import *
from matplotlib.colors import LogNorm
import cmasher
fourpi=4*np.pi

Nxy=512
Nz=191
Lxy=13265.317*u.Mpc
Lz=319.469187*u.Mpc

Deltaxy=Lxy/Nxy

N_in=1024
k_in=np.linspace(1e-4,0.5,N_in)/u.Mpc
dec_pwr=k_in.value**-4.1
dec_pwr_norm=np.mean(dec_pwr)
inc_pwr=k_in.value**1.8 
inc_pwr_norm=np.mean(inc_pwr)
power_unit=u.mK**2*u.Mpc**3

P_flat=np.ones(N_in)*power_unit
Norm1=CenteredNorm(vcenter=1,halfrange=0.1)

P_in_options=[
            #   [dec_pwr/dec_pwr_norm *power_unit, "decaying_power_law", None],
            #   [inc_pwr/inc_pwr_norm *power_unit, "growing_power_law", None ],
              [P_flat, "flat"              , Norm1] 

              ]
# P_in_options=[[np.ones(N_in)        *power_unit, "flat"              ]]
ft=1/np.sqrt(2)
PSF=np.load("fidu_box_PSF_full_cont__none__600MHz__N_CST_types_1__N_ptg_err_0_dist_hybr__layer_True__wedge_False__seed_None.npy")
FFTPSF=fftshift(fftn(ifftshift(PSF)*Deltaxy**2,axes=(0,1),norm="backward"))
FFTPSF0=FFTPSF[Nxy//2,Nxy//2,Nz//2]
print(FFTPSF0)

forpower1=cosmo_stats(Lxy,Lz=Lz,
                      P_fid=P_flat,k_fid=k_in,
                      frac_tol=ft,
                      Nxy=Nxy,Nz=Nz)
forpower1.generate_GRF()
T1=forpower1.T_pristine
np.save("T1.npy",T1.value)
forpower1.power_Monte_Carlo()
power1E2E=forpower1.P_binned_MC_complete.value.T
forpower1_2=cosmo_stats(Lxy,Lz=Lz,
                        P_fid=P_flat,k_fid=k_in,
                        frac_tol=ft,
                        Nxy=Nxy,Nz=Nz)
forpower1_2.power_Monte_Carlo()

bwP=cosmo_stats(Lxy,Lz,
                T_pristine=T1,
                PSF=PSF,T1=T1,
                frac_tol=ft,
                Nxy=Nxy,Nz=Nz)
bwP.generate_P()
bwP.bin_power()
double_T1=bwP.P_binned

for P_case in P_in_options:
    P_in,P_name,e2enorm=P_case
    print("Nxy,Nz=",Nxy,Nz)
    stats_container0=cosmo_stats(Lxy,Lz=Lz,
                                P_fid=P_in,k_fid=k_in,
                                Nxy=Nxy,Nz=Nz,
                                LoS_apo=True,transverse_apo=False,
                                frac_tol=ft, nu_ctr=600*u.MHz)
    stats_container0.power_Monte_Carlo()
    stats_container1=cosmo_stats(Lxy,Lz=Lz,
                                 P_fid=P_in,k_fid=k_in,
                                 Nxy=Nxy,Nz=Nz,
                                 PSF=PSF,T1=T1,
                                 LoS_apo=True,transverse_apo=False,
                                 frac_tol=ft, nu_ctr=600*u.MHz)
    stats_container1.power_Monte_Carlo()
    stats_container2=cosmo_stats(Lxy,Lz=Lz,
                                 P_fid=P_in,k_fid=k_in,
                                 Nxy=Nxy,Nz=Nz,
                                 PSF=PSF,T1=stats_container0.T_pristine,
                                 LoS_apo=True,transverse_apo=False,
                                 frac_tol=ft, nu_ctr=600*u.MHz)
    stats_container2.power_Monte_Carlo()
    stats_container3=cosmo_stats(Lxy,Lz=Lz,
                                 P_fid=P_in,k_fid=k_in,
                                 Nxy=Nxy,Nz=Nz,
                                 PSF=PSF,T1=stats_container0.T_pristine,
                                 LoS_apo=False,transverse_apo=False,
                                 frac_tol=ft, nu_ctr=600*u.MHz)
    stats_container3.power_Monte_Carlo()
    print("completed Monte Carlo")

    k_perp_out=stats_container0.kperpbins[:-1]
    k_par_out= stats_container0.kparbins[:-1]
    cyl_extent=[k_perp_out[0].value,k_perp_out[-1].value,k_par_out[0].value,k_par_out[-1].value]
    P0=stats_container0.P_binned_MC_complete.value.T
    P1=stats_container1.P_binned_MC_complete.value.T
    double_Tdata=stats_container2.P_binned_MC_complete.value.T
    noXyesB=stats_container3.P_binned_MC_complete.value.T
    noXyesB_numerator=stats_container3.P_numerator.value
    # custom_denom=np.abs(FFTPSF0)**2 # math is super wrong
    custom_denom_arg=fftshift(fftn(
                                    ifftshift(FFTPSF*Lxy**2*Lz),
                                    axes=(2),norm="backward"
                                  ))
    custom_denom=np.abs(custom_denom_arg)**2
    noXyesB_customdenom_unbinned=noXyesB_numerator/custom_denom
    # need to bin this so make a new cosmo_stats object just to bin it
    stats_container3.P_unbinned=noXyesB_customdenom_unbinned
    stats_container3.bin_power()
    noXyesB_customdenom=stats_container3.P_binned.T
    SimpNumRealizRat_unbinned=forpower1.P_numerator/forpower1_2.P_numerator
    stats_container3.P_unbinned=SimpNumRealizRat_unbinned
    stats_container3.bin_power()
    SimpNumRealizRat=stats_container3.P_binned.T

    _,axs=plt.subplots(2,4,layout="constrained",figsize=(11,9))
    im=axs[0,0].imshow(power1E2E,extent=cyl_extent,cmap=cmasher.horizon,origin="lower",
                       norm=Norm1)
    plt.colorbar(im,ax=axs[0,0])
    axs[0,0].set_title("E2E w/o PSF or apodization\nmean,med={:.4f}, {:.4f}".format(np.mean(power1E2E),np.median(power1E2E)))

    im=axs[0,1].imshow(P1,extent=cyl_extent,cmap=cmasher.horizon,origin="lower",
                       norm=e2enorm)
    plt.colorbar(im,ax=axs[0,1])
    axs[0,1].set_title("E2E w/ PSF\non target scale\nmean,med={:.4f}, {:.4f}".format(np.mean(P1),np.median(P1)))

    im=axs[0,2].imshow(SimpNumRealizRat,extent=cyl_extent,cmap=cmasher.horizon,origin="lower",
                       norm=CenteredNorm(vcenter=np.median(SimpNumRealizRat),halfrange=0.5*np.median(SimpNumRealizRat)))
    plt.colorbar(im,ax=axs[0,2])
    axs[0,2].set_title("simple ratio of \nnumerator realizations\nmean,med={:.4f}, {:.4f}".format(np.mean(P0),np.median(P0)))

    im=axs[0,3].imshow(P1,extent=cyl_extent,cmap=cmasher.horizon,origin="lower",
                       norm=CenteredNorm(vcenter=np.median(P1),halfrange=0.5*np.median(P1)))
    plt.colorbar(im,ax=axs[0,3])
    axs[0,3].set_title("E2E w/ PSF\non data scale\nmean,med={:.4f}, {:.4f}".format(np.mean(P1),np.median(P1)))



    im=axs[1,0].imshow(double_T1,extent=cyl_extent,cmap=cmasher.horizon,origin="lower",
                       norm=CenteredNorm(vcenter=np.median(double_T1),halfrange=0.5*np.median(double_T1)))
    plt.colorbar(im,ax=axs[1,0])
    axs[1,0].set_title("double T1; PSF\nmean,med={:.4f}, {:.4f}".format(np.mean(double_T1),np.median(double_T1)))

    im=axs[1,1].imshow(double_Tdata,extent=cyl_extent,cmap=cmasher.horizon,origin="lower",
                       norm=CenteredNorm(vcenter=np.median(double_Tdata),halfrange=0.5*np.median(double_Tdata)))
    plt.colorbar(im,ax=axs[1,1])
    axs[1,1].set_title("double Tdata; PSF\nmean,med={:.4f}, {:.4f}".format(np.mean(double_Tdata),np.median(double_Tdata)))

    im=axs[1,2].imshow(noXyesB,extent=cyl_extent,cmap=cmasher.horizon,origin="lower",
                       norm=CenteredNorm(vcenter=np.median(noXyesB),halfrange=0.5*np.median(noXyesB)))
    plt.colorbar(im,ax=axs[1,2])
    axs[1,2].set_title("PSF but no apo\nmean,med={:.4f}, {:.4f}".format(np.mean(noXyesB),np.median(noXyesB)))

    # im=axs[1,3].imshow(noXyesB_customdenom,extent=cyl_extent,cmap=cmasher.horizon,origin="lower",
                      #  norm=CenteredNorm(vcenter=np.median(noXyesB_customdenom),halfrange=0.5*np.median(noXyesB_customdenom)))
    # plt.colorbar(im,ax=axs[1,3])
    # axs[1,3].set_title("PSF but no apo\ncustom denom\nmean,med={:.4e}, {:.4e}".format(np.mean(noXyesB_customdenom),np.median(noXyesB_customdenom)))

    plt.suptitle(P_name+" power end-to-end comparison")
    plt.savefig("PSF_A_B_"+P_name+"_end_to_end.png",dpi=400)
    plt.close()