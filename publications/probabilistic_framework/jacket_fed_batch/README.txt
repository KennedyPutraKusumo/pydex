#####################################
Stirred Tank Reactor
#####################################

To reproduce the results in the ESI of the paper, run the following:
 (i) Unrestricted Local Design: "jacket_fed_batch_oed.py"
 (ii) Resample the restricted experimental space: "jacket_fed_batch_ds.py"
 (iii) Restricted Local Design: "jacket_fed_batch_oed_cc.py"
 (iv) Restricted Average Design: "jfb_oed_cc_pb_100_scr_30_spt.py"

The atomic matrices required to reproduce the restricted average design is provided as
re-computing the sensitivities may take a long time. The atomics take (~250 MB). To force
the scripts to re-compute the sensitivities, simply set "load_atomics = False" in script
"jfb_oed_cc_pb_100_scr_30_spt.py". Atomics for local designs are not provided as these
take significantly less time.

The final figures, illustration scripts, and logs for the results presented in the ESI
are also included within this folder, with relatively clearer namings. Notice that the
logs for the restricted average (or pseudo-bayesian, abbreviated as pb) design is split
into the sensitivity analysis, and optimization run log.

The main illustration scripts are as follows:
 (i) the restricted space samples only: "illustrate_restricted_space"
 (ii) presents the restricted local design: "illustrate_restricted_local"
 (iii) presents the restricted average design: "illustrate_restricted_pb"

Note that when the illustrations scripts are run, they save the figures as .png files.
