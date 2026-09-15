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
k_in=np.linspace(1e-4,2,N_in)/u.Mpc
dec_pwr=k_in.value**-4.1
dec_pwr_norm=np.mean(dec_pwr)
inc_pwr=k_in.value**1.8 
inc_pwr_norm=np.mean(inc_pwr)
power_unit=u.mK**2*u.Mpc**3

normed_dec_pwr=dec_pwr/dec_pwr_norm*power_unit
normed_inc_pwr=inc_pwr/inc_pwr_norm*power_unit

P_flat=np.ones(N_in)*power_unit
Norm1=CenteredNorm(vcenter=1,halfrange=0.1)
Norm2=LogNorm(vmin=np.min(normed_dec_pwr.value),vmax=np.max(normed_dec_pwr.value))
ioname="full_cont__none__600MHz__N_CST_types_1__N_ptg_err_0_dist_hybr__layer_True__wedge_False__seed_None"
UA=np.load("fidu_box_PSF_"+ioname+".npy")

P_in_options=[
              [ P_flat,         "flat",               Norm1 ],
              [ normed_dec_pwr, "decaying_power_law", Norm2 ],
              [ normed_inc_pwr, "growing_power_law",  None  ]

              ]
ft=1 # 1/np.sqrt(2)
recalc=True

for P_case in P_in_options:
    P_in,P_name,e2enorm=P_case
    if recalc:
        print("Nxy,Nz=",Nxy,Nz)
        fromTb=cosmo_stats(Lxy,Lz=Lz,
                            P_fid=P_in,k_fid=k_in,
                            Nxy=Nxy,Nz=Nz,
                            frac_tol=ft, nu_ctr=600*u.MHz)
        fromTb.power_Monte_Carlo()
        print("P Tb calc complete")
        P_Tb=fromTb.P_binned_MC_complete.value.T
        P_numerator_Tb=fromTb.P_numerator.value
        np.save("P_"+P_name+"_Tb.npy",P_Tb)
        np.save("P_numerator_"+P_name+"_Tb.npy",P_numerator_Tb)

        fromTbUA=cosmo_stats(Lxy,Lz=Lz,
                            P_fid=P_in,k_fid=k_in,
                            Nxy=Nxy,Nz=Nz,
                            UA=UA,
                            frac_tol=ft, nu_ctr=600*u.MHz)
        fromTbUA.power_Monte_Carlo()
        print("P Tb UA calc complete")
        P_TbUA=fromTbUA.P_binned_MC_complete.value.T
        P_numerator_TbUA=fromTbUA.P_numerator.value
        P_denominator_TbUA=fromTbUA.estimator_denom.value
        np.save("P_"+P_name+"_TbUA.npy",P_TbUA)
        np.save("P_numerator_"+P_name+"_TbUA.npy",P_numerator_TbUA)
        np.save("P_denominator_"+P_name+"_TbUA.npy",P_denominator_TbUA)

        fromTbX=cosmo_stats(Lxy,Lz=Lz,
                            P_fid=P_in,k_fid=k_in,
                            Nxy=Nxy,Nz=Nz,
                            LoS_apo=True,
                            frac_tol=ft, nu_ctr=600*u.MHz)
        fromTbX.power_Monte_Carlo()
        print("P Tb X calc complete")
        P_TbX=fromTbX.P_binned_MC_complete.value.T
        P_numerator_TbX=fromTbX.P_numerator.value
        np.save("P_"+P_name+"_TbX.npy",P_TbX)
        np.save("P_numerator_"+P_name+"_TbX.npy",P_numerator_TbX)

        fromTbUAX=cosmo_stats(Lxy,Lz=Lz,
                                P_fid=P_in,k_fid=k_in,
                                Nxy=Nxy,Nz=Nz,
                                UA=UA,
                                LoS_apo=True,
                                frac_tol=ft, nu_ctr=600*u.MHz)
        fromTbUAX.power_Monte_Carlo()
        print("P Tb UA X calc complete")
        P_TbUAX=fromTbUAX.P_binned_MC_complete.value.T
        P_numerator_TbUAX=fromTbUAX.P_numerator.value
        P_denominator_TbUAX=fromTbUAX.estimator_denom.value
        np.save("P_"+P_name+"_TbUAX.npy",P_TbUAX)
        np.save("P_numerator_"+P_name+"_TbUAX.npy",P_numerator_TbUAX)
        np.save("P_denominator_"+P_name+"_TbUAX.npy",P_denominator_TbUAX)

        print("completed Monte Carlos")

        k_perp_out=fromTb.kperpbins[:-1]
        k_par_out= fromTb.kparbins[:-1]
        cyl_extent=[k_perp_out[0].value,k_perp_out[-1].value,k_par_out[0].value,k_par_out[-1].value]
        np.savetxt("cyl_extent.txt",cyl_extent)
    else:
        P_Tb=np.load("P_"+P_name+"_Tb.npy")
        P_TbUA=np.load("P_"+P_name+"_TbUA.npy")
        P_TbX=np.load("P_"+P_name+"_TbX.npy")
        P_TbUAX=np.load("P_"+P_name+"_TbUAX.npy")

        P_numerator_Tb=np.load("P_numerator_"+P_name+"_Tb.npy")
        P_numerator_TbUA=np.load("P_numerator_"+P_name+"_TbUA.npy")
        P_numerator_TbX=np.load("P_numerator_"+P_name+"_TbX.npy")
        P_numerator_TbUAX=np.load("P_numerator_"+P_name+"_TbUAX.npy")

        P_denominator_TbUA=np.load("P_denominator_"+P_name+"_TbUA.npy")
        P_denominator_TbUAX=np.load("P_denominator_"+P_name+"_TbUAX.npy")

        cyl_extent=np.genfromtxt("cyl_extent.txt")

    _,axs=plt.subplots(1,4,layout="constrained",figsize=(11,9))
    im=axs[0].imshow(P_Tb,extent=cyl_extent,cmap=cmasher.sapphire,origin="lower",
                    norm=e2enorm)
    plt.colorbar(im,ax=axs[0])
    axs[0].set_title("P from T$_b$\nmean,med=\n{:.4f}, {:.4f}".format(np.mean(P_Tb),np.median(P_Tb)))

    localnorm=CenteredNorm(vcenter=np.median(P_TbUA),halfrange=np.median(P_TbUA))
    im=axs[1].imshow(P_TbUA,extent=cyl_extent,cmap=cmasher.sapphire,origin="lower",
                    norm=localnorm)
    plt.colorbar(im,ax=axs[1])
    axs[1].set_title("P from T$_b$, UA\nmean,med=\n{:.4f}, {:.4f}".format(np.mean(P_TbUA),np.median(P_TbUA)))

    im=axs[2].imshow(P_TbX,extent=cyl_extent,cmap=cmasher.sapphire,origin="lower",
                    norm=e2enorm)
    plt.colorbar(im,ax=axs[2])
    axs[2].set_title("P from T$_b$, X\nmean,med=\n{:.4f}, {:.4f}".format(np.mean(P_TbX),np.median(P_TbX)))

    localnorm=CenteredNorm(vcenter=np.median(P_TbUAX),halfrange=np.median(P_TbUAX))
    im=axs[3].imshow(P_TbUAX,extent=cyl_extent,cmap=cmasher.sapphire,origin="lower",
                    norm=localnorm)
    plt.colorbar(im,ax=axs[3])
    axs[3].set_title("P from T$_b$, UA, X\nmean,med=\n{:.4f}, {:.4f}".format(np.mean(P_TbUAX),np.median(P_TbUAX)))
    
    plt.suptitle(P_name+" power end-to-end comparison")
    plt.savefig("E2E_UA_X_"+P_name+"_end_to_end.png",dpi=400)
    plt.close()

    percentile=75
    medhigh_num_Tb=np.nanpercentile(np.abs(P_numerator_Tb),percentile)
    medhigh_num_TbUA=np.nanpercentile(np.abs(P_numerator_TbUA),percentile)
    medhigh_num_TbX=np.nanpercentile(np.abs(P_numerator_TbX),percentile)
    medhigh_num_TbUAX=np.nanpercentile(np.abs(P_numerator_TbUAX),percentile)
    comprehensive_slice_figure(P_numerator_Tb,
                               cmap=cmasher.sapphire,
                               norm=CenteredNorm(vcenter=medhigh_num_Tb,halfrange=medhigh_num_Tb),
                               title="P "+P_name+" numerator Tb",
                               name="P_numerator_"+P_name+"_Tb.png")
    comprehensive_slice_figure(P_numerator_TbUA,
                               cmap=cmasher.sapphire,
                               norm=CenteredNorm(vcenter=medhigh_num_TbUA,halfrange=medhigh_num_TbUA),
                               title="P "+P_name+" numerator Tb UA",
                               name="P_numerator_"+P_name+"_TbUA.png")
    comprehensive_slice_figure(P_numerator_TbX,
                               cmap=cmasher.sapphire,
                               norm=CenteredNorm(vcenter=medhigh_num_TbX,halfrange=medhigh_num_TbX),
                               title="P "+P_name+" numerator Tb X",
                               name="P_numerator_"+P_name+"_TbX.png")
    comprehensive_slice_figure(P_numerator_TbUAX,
                               cmap=cmasher.sapphire,
                               norm=CenteredNorm(vcenter=medhigh_num_TbUAX,halfrange=medhigh_num_TbUAX),
                               title="P "+P_name+" numerator Tb UA X",
                               name="P_numerator_"+P_name+"_TbUAX.png")


    medhigh_den_TbUA=np.nanpercentile(np.abs(P_denominator_TbUA),percentile)
    medhigh_den_TbUAX=np.nanpercentile(np.abs(P_denominator_TbUAX),percentile)
    comprehensive_slice_figure(P_denominator_TbUA,
                               cmap=cmasher.sapphire,
                               norm=CenteredNorm(vcenter=medhigh_den_TbUA,halfrange=medhigh_den_TbUA),
                               title="P "+P_name+" denominator Tb UA",
                               name="P_denominator_"+P_name+"_TbUA.png")    
    comprehensive_slice_figure(P_denominator_TbUAX,
                               cmap=cmasher.sapphire,
                               norm=CenteredNorm(vcenter=medhigh_den_TbUAX,halfrange=medhigh_den_TbUAX),
                               title="P "+P_name+" denominator Tb UA X",
                               name="P_denominator_"+P_name+"_TbUAX.png")