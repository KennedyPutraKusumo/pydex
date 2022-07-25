#####################################
Esterification
#####################################

To reproduce the results, Pydex requires the following dependencies:
 (i) Pyomo - you would need to install pyomo onto your Python environment's path
 (ii) moddex - supplied within this "esterification" folder

All python scripts with "restricted_" leading its name requires the file "output.pkl"
in the folder "restricted_sample_DEUS" to run.

[IMPORTANT]: results were obtained using cvxpy v1.1.12. There have been changes to the
cvxpy-MOSEK interface that automatically dualizes the problem that may lead to different
solutions. Consider this alongside the other computational details in the paper when
reproducing the results.

To simply re-plot the figures of the paper (not resolving the problem), please
run the script "illustrate_6D_ds.py". You should only need to modify the variables
"local_design", "unrestricted", and "offset_axes_limits" in appropriate combinations to
reproduce the published results.

To reproduce the main result presented in the paper:
 (i) Re-run the sensitivity analyses: "restricted_pb_mp.py"
    - runs the sensitivity analyses required to compute the atomic matrices for the
      restricted average design
    - saves the atomic matrices into the pickle file:
      "restricted_pb_atomic_fims_1080_n_c_100_n_scr_fix_switch_rxn_order.pkl"
    - [WARNING]: the memory requirement for the atomic matrices can be large (~150 MB)
                 do not repeatedly run the script as the atomics will be saved each time
 (ii) Re-optimize the Average D-optimal Design: "restricted_pb_oed.py"
    - requires the atomic matrices to run:
      "restricted_average_results/restricted_pb_atomic_fims_1080_n_c_100_n_scr_fix_switch_rxn_order.pkl"
    - computes the D-optimal average design using MOSEK
[WARNING]: you should NOT need to modify any of the variables in any of the scripts

The main illustration scripts are:
 (i) Restricted Experimenatl Space: "illustrate_6D_DS.py"
 (ii)

Results presented in the ESI can also be reproduced using the files:
 (i) Restricted Local Design: "restricted_local_oed.py"
 (ii) Unrestricted Local Design: "local_oed.py"
 (iii) Plot of Tcf vs. time for the Restricted Average Design: "vary_parameters_order.py"
 (iv) average criterion values of:
    - restricted local design: "restricted_pb_oed.py" with the variable
      "calculate_local_average" fixed to "True".
    - unrestricted local design: "pb_oed.py"
