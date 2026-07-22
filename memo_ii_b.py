from forecasting_pipeline import *

# >>>>> where to look for CST <<<<<
CST_dir="/Users/sophiarubens/Downloads/research/code/pipeline/CST_beams/CHORD_feed_tilts_integ_dom_600/farfield_" # local
# CST_dir="/home/sophiaru/scratch/CHORD_CST/farfield_" # Fir

# >>>>> details of fidu and syst CST cases <<<<< 
fiduname="fiducial/"
systnames= [ "+0.0+0.0+0.5", "+0.0+0.0+1.0", "+0.0+0.0+1.5",     # local. commented-out part would be accessible if I were to download more CST from Fir!
             "+0.0+0.5+0.0", "+0.0+1.0+0.0" ] # "+0.0+1.5+0.0",
            #  "+0.0+0.0-0.5", "+0.0+0.0-1.0", "+0.0+0.0-1.5", 
            #  "+0.0-0.5+0.0", "+0.0-1.0+0.0", "+0.0-1.5+0.0"  ]
# systnames= [ "+0.0+0.0+0.5", "+0.0+0.0+1.0", "+0.0+0.0+1.5",     # Fir
#              "+0.0+0.5+0.0", "+0.0+1.0+0.0", "+0.0+1.5+0.0",
#              "+0.0+0.0-0.5", "+0.0+0.0-1.0", "+0.0+0.0-1.5", 
#              "+0.0-0.5+0.0", "+0.0-1.0+0.0", "+0.0-1.5+0.0"  ]
all_syst_dirs=[sn+"_deg/" for sn in systnames]

N_systs_use=len(systnames) # exhaustive case
N_systs_use=1 # pared-down case for debugging. as of July 20th: I believe my synthesized beams (well, I believe the ways in which 
              # different systematics are showing up, even ... more updates soon), but the changes I'm interested in are visible even 
              # in the fiducial beam–aware power spec, so I'm running some extremely toy tests
N_CST_total=N_systs_use+1

# configure pointing errors
base_pointing_error=[1.2,-0.7,0.4]
base_seed=5920185708
meta_rng=np.random.default_rng(base_seed-1)
N_ptg_errs_per_CST=meta_rng.integers(low=0,high=2,
                                     size=N_CST_total,endpoint=False) # +1 is to account for the fiducial beam
print("N_ptg_errs_per_CST=",N_ptg_errs_per_CST)
pointingerrs=[pointing_family(base_pointing_error,Ni,seed=base_seed+i) for i,Ni in enumerate(N_ptg_errs_per_CST)]
print("pointingerrs=",pointingerrs)
with open("ptg_err.json", "w") as f:
   json.dump(pointingerrs, f, indent=2, default=str)

# re-simulate / re-plot
power_comparison_plots(redo_window_calc=True, # redo the Monte Carlos
                       redo_box_calc=True,    # re-synthesize the beams; reimports CST only if files DNE
                       array_version="full", nu_ctr=600.*u.MHz, 
                       frac_tol_conv=0.25, which_power="P",
                       antenna_dist="frame", # comment out to revert to the two-pixel frame systematic
                       pointing_errors=pointingerrs[:N_CST_total],
                       CST_lo=0.59*u.GHz,CST_hi=0.61*u.GHz,CST_deltanu=2e-4*u.GHz,
                       N_timesteps=1,
                     #   overresolve=[True,True],
                       beam_sim_directory=CST_dir, CST_f_head_fidu=fiduname, CST_f_head_syst=all_syst_dirs[:N_systs_use])