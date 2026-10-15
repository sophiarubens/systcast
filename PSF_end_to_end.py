from forecasting_pipeline import *
from matplotlib.colors import LogNorm
import cmasher
fourpi=4*np.pi

plt.rcParams.update({'font.size': 16})

Nxy=512
Nz=191
Lxy=13265.317*u.Mpc
Lz=319.469187*u.Mpc

Deltaxy=Lxy/Nxy
dec_idx=-1.1
inc_idx=1.8

def dec_law(k,dec_idx=dec_idx):
    vals= k**dec_idx
    return vals/np.mean(vals)
def flat_law(k):
    return np.ones_like(k)
def incr_law(k,inc_idx=1.8):
    vals=k**inc_idx
    return vals/np.mean(vals)

N_in=1024
Nsigma=1
aspect=0.25
k_in=np.linspace(1e-4,2,N_in)/u.Mpc
dec_pwr=k_in.value**dec_idx
dec_pwr_norm=np.mean(dec_pwr)
inc_pwr=k_in.value**inc_idx
inc_pwr_norm=np.mean(inc_pwr)
power_unit=u.mK**2*u.Mpc**3

normed_dec_pwr=dec_pwr/dec_pwr_norm*power_unit
normed_inc_pwr=inc_pwr/inc_pwr_norm*power_unit

P_flat=np.ones(N_in)*power_unit
Norm_flat=CenteredNorm(vcenter=1,halfrange=5e-2) # 1e-2 ok
Norm_dec=LogNorm()
Norm_inc=CenteredNorm(vcenter=inc_pwr_norm,halfrange=inc_pwr_norm)
ioname="full_cont__none__600MHz__N_CST_types_1__N_ptg_err_0_dist_hybr__layer_True__wedge_False__seed_None"
M=np.load("fidu_box_MM_"+ioname+".npy")

P_in_options=[
              [ normed_dec_pwr, "decaying_power_law", Norm_dec,  dec_law  ],
              [ P_flat,         "flat",               Norm_flat, flat_law ],
              [ normed_inc_pwr, "growing_power_law",  Norm_inc,  incr_law  ]

              ]
ft=0.25 # 1/np.sqrt(2)
recalc=False

for P_case in P_in_options:
    P_in,P_name,e2enorm,law=P_case
    if recalc:
        print("Nxy,Nz=",Nxy,Nz)
        fromTb=cosmo_stats(Lxy,Lz=Lz,
                            P_fid=P_in,k_fid=k_in,
                            Nxy=Nxy,Nz=Nz,
                            frac_tol=ft, nu_ctr=600*u.MHz)
        fromTb.power_Monte_Carlo()
        print("P Tb calc complete")
        P_Tb=fromTb.P_binned_MC_complete.value.T
        np.save("P_"+P_name+"_Tb.npy",P_Tb)

        Delta_kperp=fromTb.Deltakxy
        Delta_kpar=fromTb.Deltakz
        kperp_bin_grid=fromTb.kperpbins_grid[:-1,:-1]+0.5*Delta_kperp
        kpar_bin_grid=fromTb.kparbins_grid[:-1,:-1]+0.5*Delta_kpar
        print("kperp_bin_grid.shape=",kperp_bin_grid.shape)
        kcyl_bin_grid=np.sqrt(kperp_bin_grid**2+kpar_bin_grid**2)
        P_true=law(kcyl_bin_grid.value).T
        np.save("P_true_"+P_name+"_Tb.npy",P_Tb)

        fromTbM=cosmo_stats(Lxy,Lz=Lz,
                            P_fid=P_in,k_fid=k_in,
                            Nxy=Nxy,Nz=Nz,
                            MM=M,
                            frac_tol=ft, nu_ctr=600*u.MHz)
        fromTbM.power_Monte_Carlo()
        print("P Tb M calc complete")
        P_TbM=fromTbM.P_binned_MC_complete.value.T
        P_denominator_TbM=fromTbM.estimator_denom.value
        np.save("P_"+P_name+"_TbM.npy",P_TbM)
        np.save("P_denominator_"+P_name+"_TbM.npy",P_denominator_TbM)

        fromTbX=cosmo_stats(Lxy,Lz=Lz,
                            P_fid=P_in,k_fid=k_in,
                            Nxy=Nxy,Nz=Nz,
                            LoS_apo=True,
                            frac_tol=ft, nu_ctr=600*u.MHz)
        fromTbX.power_Monte_Carlo()
        print("P Tb X calc complete")
        P_TbX=fromTbX.P_binned_MC_complete.value.T
        np.save("P_"+P_name+"_TbX.npy",P_TbX)

        fromTbMX=cosmo_stats(Lxy,Lz=Lz,
                                P_fid=P_in,k_fid=k_in,
                                Nxy=Nxy,Nz=Nz,
                                MM=M,
                                LoS_apo=True,
                                frac_tol=ft, nu_ctr=600*u.MHz)
        fromTbMX.power_Monte_Carlo()
        print("P Tb M X calc complete")
        P_TbMX=fromTbMX.P_binned_MC_complete.value.T
        P_denominator_TbMX=fromTbMX.estimator_denom.value
        np.save("P_"+P_name+"_TbMX.npy",P_TbMX)
        np.save("P_denominator_"+P_name+"_TbMX.npy",P_denominator_TbMX)

        print("completed Monte Carlos")

        k_perp_out=fromTb.kperpbins[:-1]
        k_par_out= fromTb.kparbins[:-1]
        cyl_extent=[k_perp_out[0].value,k_perp_out[-1].value,k_par_out[0].value,k_par_out[-1].value]
        np.savetxt("cyl_extent.txt",cyl_extent)
    else:
        P_true=np.load("P_true_"+P_name+"_Tb.npy")
        P_Tb=np.load("P_"+P_name+"_Tb.npy")
        P_TbM=np.load("P_"+P_name+"_TbM.npy")
        P_TbX=np.load("P_"+P_name+"_TbX.npy")
        P_TbMX=np.load("P_"+P_name+"_TbMX.npy")

        P_denominator_TbM=np.load("P_denominator_"+P_name+"_TbM.npy")
        P_denominator_TbMX=np.load("P_denominator_"+P_name+"_TbMX.npy")

        cyl_extent=np.genfromtxt("cyl_extent.txt")

    _,axs=plt.subplots(1,5,figsize=(12,6),
                       sharey=True,layout="constrained")
    im=axs[0].imshow(P_true,extent=cyl_extent,cmap=cmasher.horizon,origin="lower",
                    norm=e2enorm,
                    aspect=aspect)
    plt.colorbar(im,ax=axs[0])
    axs[0].set_title("True power\n ")

    res=P_Tb-P_true
    im=axs[1].imshow(res/P_true,extent=cyl_extent,cmap="managua",origin="lower",
                    norm=CenteredNorm(halfrange=Nsigma*np.std(res/P_true)),
                    aspect=aspect)
    plt.colorbar(im,ax=axs[1])
    axs[1].set_title("From T$_b$")

    res=P_TbM-P_true
    im=axs[2].imshow(res/P_true,extent=cyl_extent,cmap="managua",origin="lower",
                     norm=CenteredNorm(halfrange=Nsigma*np.std(res/P_true)),
                     aspect=aspect)
    plt.colorbar(im,ax=axs[2])
    axs[2].set_title("                      Fractional residuals\nFrom T$_b$, M")

    res=P_TbX-P_true
    im=axs[3].imshow(res/P_true,extent=cyl_extent,cmap="managua",origin="lower",
                    norm=CenteredNorm(halfrange=Nsigma*np.std(res/P_true)),
                    aspect=aspect)
    plt.colorbar(im,ax=axs[3])
    axs[3].set_title("From T$_b$, X")

    res=P_TbMX-P_true
    im=axs[4].imshow(res/P_true,extent=cyl_extent,cmap="managua",origin="lower",
                     norm=CenteredNorm(halfrange=Nsigma*np.std(res/P_true)),
                     aspect=aspect)
    plt.colorbar(im,ax=axs[4])
    axs[4].set_title("From T$_b$, M, X")
    
    # plt.suptitle(P_name+" power end-to-end comparison")

    for i in range(5):
        axs[i].set_xlabel("k$_{\perp}$ (1/Mpc)")
        axs[i].tick_params("x", rotation=45)
    axs[0].set_ylabel("k$_{||}$ (1/Mpc)")
    plt.savefig("E2E_M_X_"+P_name+"_end_to_end.png",dpi=400)
    plt.close()