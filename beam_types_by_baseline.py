import numpy as np 
from matplotlib import pyplot as plt
import cmasher as cmr
from forecasting_pipeline import comoving_distance
pipeline_dir="/Users/sophiarubens/Downloads/research/code/pipeline/"

N_all= [[1,0], [1,1], [2,0], [2,1], [2,2]]
cases = [["full_column/", "rwcl"],
         ["full_frame/",  "frme"],
         ["full_random/", "rand"]]

bmax=np.sqrt((6.5*22)**2 +(8.3*24)**2)
Nbin=15 # 10 is ok but kind of coarse; 20 is pretty overresolved
bin_edges=np.linspace(0,bmax,Nbin+1)
bin_delta=bin_edges[1]-bin_edges[0]
re_centred_bin_floors=bin_edges[:-1]+0.5*bin_delta

for case in cases:
    directory, ant_distrib = case
    for N in N_all:
        N_CST,N_ptg_err=N

        ioname="full_cont__none__600MHz__N_CST_types_"+str(N_CST)+"__N_ptg_err_"+str(N_ptg_err)+"_dist_"+ant_distrib+"__layer_True__wedge_False__seed_None"

        baseline_coords=np.load(pipeline_dir+directory+"uvw_inst_"+ioname+".npy") # should have shape (N_bl,3). 3 is for uvw
        beam_type_indices=np.load(pipeline_dir+directory+"pb_types_"+ioname+".npy") # should have shape (N_bl,2). 2 is for antenna i and j types
        N_types=len(np.unique(beam_type_indices)) # combinatorially aware version, not just N_CST+N_ptg_err
        baseline_lengths=np.sqrt(np.einsum("ij,ij->i", baseline_coords,baseline_coords)) # (N_bl,)

        N_baselines_per_bin, _=np.histogram(baseline_lengths, bins=bin_edges) # shape should be (N_bins,)
        N_baselines_per_bin[N_baselines_per_bin==0]=1
        if N==[1,0]:
            N_baselines=baseline_coords.shape[0]
        Dc=comoving_distance(z=1420.5/600-1)
        prefac=2*np.pi*1420.5*1e6/(2.998e8*Dc.value)

        list_of_baseline_lengths_ij=[]
        type_names=[]
        for i in range(N_types):
            type_i=beam_type_indices[i]
            for j in range(i+1):
                type_j=beam_type_indices[j]

                here=((beam_type_indices[:,0]==i
                        )&(beam_type_indices[:,1]==j)
                    )|((beam_type_indices[:,0]==j
                        )&(beam_type_indices[:,1]==i))
                
                if len(here)>0:
                    baseline_lengths_ij=baseline_lengths[here]
                    list_of_baseline_lengths_ij.append(baseline_lengths_ij)
                    type_names.append("{}{}<->{}{}".format(i,j,j,i))

        N_baseline_types=len(list_of_baseline_lengths_ij)
        Ncol= 2 if N_baseline_types>4 else 1
        colours=cmr.take_cmap_colors('cmr.horizon', N_baseline_types, cmap_range=(0., 0.925), return_fmt='hex')
        title_ij="{} [{},{}]".format(ant_distrib, N_CST,N_ptg_err)

        plt.figure()
        values, _, _ = plt.hist(list_of_baseline_lengths_ij, bins=bin_edges,
                                histtype="barstacked", 
                                color=colours, label=type_names)
        plt.xlabel("baseline length (m)")
        ax=plt.gca()
        ax2=ax.secondary_xaxis(location="top",
                               functions=(lambda x: prefac*x,  # plug baseline length, get kperp
                                          lambda x: x/prefac)) # plug kperp, get baseline length
        ax2.set_xticks(np.linspace(0,2.2,11)) # values in arg are of k-perp, not baseline length
        ax2.set_xlabel("k$_\perp$ (1/Mpc)")
        plt.ylabel("number of baselines")
        plt.legend()
        plt.title(title_ij)
        plt.savefig("stacked_histogram_{}_{}_{}.png".format(ant_distrib,N_CST,N_ptg_err))
        plt.close()

        values=np.atleast_2d(values) # shape is (N_baseline_types, N_bins)
        baseline_counts_all=[]
        plt.figure()
        for i,colour in enumerate(colours):
            baseline_counts, _=np.histogram(list_of_baseline_lengths_ij[i], bins=bin_edges)
            baseline_counts_all.append(baseline_counts)
            baseline_fractions=baseline_counts/N_baselines
            plt.plot(re_centred_bin_floors,baseline_fractions,
                     c=colour,label=type_names[i])
        plt.xlabel("baseline length for bin centre (m)")
        ax=plt.gca()
        ax2=ax.secondary_xaxis(location="top",
                               functions=(lambda x: prefac*x,  # plug baseline length, get kperp
                                          lambda x: x/prefac)) # plug kperp, get baseline length
        ax2.set_xticks(np.linspace(0,2.2,10)) # values in arg are of k-perp, not baseline length
        ax2.set_xlabel("k$_\perp$ (1/Mpc)")
        plt.ylabel("fraction of total number of baselines")
        plt.title(title_ij)
        plt.legend(ncol=Ncol)
        plt.savefig("proportions_of_total_{}_{}_{}.png".format(ant_distrib,N_CST,N_ptg_err))
        plt.close()
        baseline_counts_all=np.array(baseline_counts_all)

        values_per_bin=np.sum(values, axis=0) # shape should be (N_bins,)
        plt.figure(layout="constrained")
        for i,colour in enumerate(colours):
            weighted_baseline_fraction=baseline_counts_all[i,:]/N_baselines_per_bin # bruh this is so cryptic... if it works I need to update the nomenclature to something that I will still understand when I am not in the tunnel vision induced by coffee oatmeal, a cup of iced mint black tea, a cup of hot oolong tea, and a cup of hot jasmine tea on ~6.5 hrs of sleep
            plt.plot(re_centred_bin_floors,weighted_baseline_fraction,
                     c=colour,label=type_names[i])
        plt.xlabel("baseline length for bin centre (m)")
        ax=plt.gca()
        ax2=ax.secondary_xaxis(location="top",
                               functions=(lambda x: prefac*x,  # plug baseline length, get kperp
                                          lambda x: x/prefac)) # plug kperp, get baseline length
        ax2.set_xticks(np.linspace(0,2.2,10)) # values in arg are of k-perp, not baseline length
        ax2.set_xlabel("k$_\perp$ (1/Mpc)")
        plt.ylabel("fraction of the number of baselines in that bin")
        plt.ylim(0,1.1)
        plt.title(title_ij)
        plt.legend(ncol=Ncol)
        plt.savefig("proportions_of_bin_{}_{}_{}.png".format(ant_distrib,N_CST,N_ptg_err))
        plt.close()