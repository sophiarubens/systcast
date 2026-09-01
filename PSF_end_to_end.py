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
ioname="full_cont__none__600MHz__N_CST_types_1__N_ptg_err_0_dist_hybr__layer_True__wedge_False__seed_None"
PSF=np.load("fidu_box_PSF_"+ioname+".npy")
Aeff=np.load("fidu_Aeff_"+ioname+".npy")

P_in_options=[
            #   [dec_pwr/dec_pwr_norm *power_unit, "decaying_power_law", None],
            #   [inc_pwr/inc_pwr_norm *power_unit, "growing_power_law", None ],
              [P_flat, "flat"              , Norm1] 

              ]
ft=1 # 1/np.sqrt(2)

for P_case in P_in_options:
    P_in,P_name,e2enorm=P_case
    print("Nxy,Nz=",Nxy,Nz)
    fromTb=cosmo_stats(Lxy,Lz=Lz,
                       P_fid=P_in,k_fid=k_in,
                       Nxy=Nxy,Nz=Nz,
                       frac_tol=ft, nu_ctr=600*u.MHz)
    fromTb.power_Monte_Carlo()
    print("P Tb calc complete")
    P_Tb=fromTb.P_binned_MC_complete.value
    fromTbA=cosmo_stats(Lxy,Lz=Lz,
                        P_fid=P_in,k_fid=k_in,
                        Nxy=Nxy,Nz=Nz,
                        Aeff=Aeff,
                        frac_tol=ft, nu_ctr=600*u.MHz)
    fromTbA.power_Monte_Carlo()
    print("P Tb A calc complete")
    P_TbA=fromTbA.P_binned_MC_complete.value.T
    fromTbB=cosmo_stats(Lxy,Lz=Lz,
                        P_fid=P_in,k_fid=k_in,
                        Nxy=Nxy,Nz=Nz,
                        PSF=PSF,
                        frac_tol=ft, nu_ctr=600*u.MHz)
    fromTbB.power_Monte_Carlo()
    print("P Tb B calc complete")
    P_TbB=fromTbB.P_binned_MC_complete.value.T
    fromTbX=cosmo_stats(Lxy,Lz=Lz,
                        P_fid=P_in,k_fid=k_in,
                        Nxy=Nxy,Nz=Nz,
                        LoS_apo=True,
                        frac_tol=ft, nu_ctr=600*u.MHz)
    fromTbX.power_Monte_Carlo()
    print("P Tb X calc complete")
    P_TbX=fromTbX.P_binned_MC_complete.value.T
    fromTbAB=cosmo_stats(Lxy,Lz=Lz,
                         P_fid=P_in,k_fid=k_in,
                         Nxy=Nxy,Nz=Nz,
                         Aeff=Aeff,
                         PSF=PSF,
                         frac_tol=ft, nu_ctr=600*u.MHz)
    fromTbAB.power_Monte_Carlo()
    print("P Tb A B calc complete")
    P_TbAB=fromTbAB.P_binned_MC_complete.value.T
    fromTbAX=cosmo_stats(Lxy,Lz=Lz,
                         P_fid=P_in,k_fid=k_in,
                         Nxy=Nxy,Nz=Nz,
                         Aeff=Aeff,
                         LoS_apo=True,
                         frac_tol=ft, nu_ctr=600*u.MHz)
    fromTbAX.power_Monte_Carlo()
    print("P Tb A X calc complete")
    P_TbAX=fromTbAX.P_binned_MC_complete.value.T
    fromTbBX=cosmo_stats(Lxy,Lz=Lz,
                         P_fid=P_in,k_fid=k_in,
                         Nxy=Nxy,Nz=Nz,
                         PSF=PSF,
                         LoS_apo=True,
                         frac_tol=ft, nu_ctr=600*u.MHz)
    fromTbBX.power_Monte_Carlo()
    print("P Tb B X calc complete")
    P_TbBX=fromTbBX.P_binned_MC_complete.value.T
    fromTbABX=cosmo_stats(Lxy,Lz=Lz,
                          P_fid=P_in,k_fid=k_in,
                          Nxy=Nxy,Nz=Nz,
                          Aeff=Aeff,
                          PSF=PSF,
                          LoS_apo=True,
                          frac_tol=ft, nu_ctr=600*u.MHz)
    fromTbABX.power_Monte_Carlo()
    print("P Tb A B X calc complete")
    P_TbABX=fromTbABX.P_binned_MC_complete.value.T
    print("completed Monte Carlos")

    k_perp_out=fromTb.kperpbins[:-1]
    k_par_out= fromTb.kparbins[:-1]
    cyl_extent=[k_perp_out[0].value,k_perp_out[-1].value,k_par_out[0].value,k_par_out[-1].value]

    _,axs=plt.subplots(2,4,layout="constrained",figsize=(11,9))
    im=axs[0,0].imshow(P_Tb,extent=cyl_extent,cmap=cmasher.horizon,origin="lower",
                       norm=Norm1)
    plt.colorbar(im,ax=axs[0,0])
    axs[0,0].set_title("P from T$_b$\nmean,med={:.4f}, {:.4f}".format(np.mean(P_Tb),np.median(P_Tb)))

    im=axs[0,1].imshow(P_TbA,extent=cyl_extent,cmap=cmasher.horizon,origin="lower",
                       norm=Norm1)
    plt.colorbar(im,ax=axs[0,1])
    axs[0,1].set_title("P from T$_b$, A\nmean,med={:.4f}, {:.4f}".format(np.mean(P_TbA),np.median(P_TbA)))

    im=axs[0,2].imshow(P_TbB,extent=cyl_extent,cmap=cmasher.horizon,origin="lower",
                       norm=Norm1)
    plt.colorbar(im,ax=axs[0,2])
    axs[0,1].set_title("P from T$_b$, B\nmean,med={:.4f}, {:.4f}".format(np.mean(P_TbB),np.median(P_TbB)))

    im=axs[0,3].imshow(P_TbX,extent=cyl_extent,cmap=cmasher.horizon,origin="lower",
                       norm=Norm1)
    plt.colorbar(im,ax=axs[0,3])
    axs[0,3].set_title("P from T$_b$, X\nmean,med={:.4f}, {:.4f}".format(np.mean(P_TbX),np.median(P_TbX)))



    im=axs[1,0].imshow(P_TbAB,extent=cyl_extent,cmap=cmasher.horizon,origin="lower",
                       norm=Norm1)
    plt.colorbar(im,ax=axs[1,0])
    axs[1,0].set_title("P from T$_b$, A, B\nmean,med={:.4f}, {:.4f}".format(np.mean(P_TbAB),np.median(P_TbAB)))

    im=axs[1,1].imshow(P_TbAX,extent=cyl_extent,cmap=cmasher.horizon,origin="lower",
                       norm=Norm1)
    plt.colorbar(im,ax=axs[1,1])
    axs[1,1].set_title("P from T$_b$, A, X\nmean,med={:.4f}, {:.4f}".format(np.mean(P_TbAX),np.median(P_TbAX)))

    im=axs[1,2].imshow(P_TbBX,extent=cyl_extent,cmap=cmasher.horizon,origin="lower",
                       norm=Norm1)
    plt.colorbar(im,ax=axs[1,2])
    axs[1,2].set_title("P from T$_b$, B, X\nmean,med={:.4f}, {:.4f}".format(np.mean(P_TbBX),np.median(P_TbBX)))

    im=axs[1,3].imshow(P_TbABX,extent=cyl_extent,cmap=cmasher.horizon,origin="lower",
                       norm=Norm1)
    plt.colorbar(im,ax=axs[1,3])
    axs[1,3].set_title("P from T$_b$, A, B, X\nmean,med={:.4f}, {:.4f}".format(np.mean(P_TbABX),np.median(P_TbABX)))

    
    plt.suptitle(P_name+" power end-to-end comparison")
    plt.savefig("PSF_A_B_"+P_name+"_end_to_end.png",dpi=400)
    plt.close()