# -*- coding: utf-8 -*-
import sys
from mozaik.meta_workflow.parameter_search import CombinationParameterSearch, SlurmSequentialBackend
import numpy
import time


if True:
    #CombinationParameterSearch(SlurmSequentialBackend(num_threads=1, num_mpi=16,slurm_options=['--hint=nomultithread'],path_to_mozaik_env='/home/cagnol/virt_env/mozaik_paper_numpy/bin/activate'), {
    #CombinationParameterSearch(SlurmSequentialBackend(num_threads=1, num_mpi=16,slurm_options=['--exclusive','--hint=nomultithread','-x w[1-8,10,12]'],path_to_mozaik_env='/home/cagnol/virt_env/mozaik_LSV1M/bin/activate'), {
    CombinationParameterSearch(SlurmSequentialBackend(num_threads=1, num_mpi=16,slurm_options=['--hint=nomultithread','-w w8'],path_to_mozaik_env='/home/cagnol/virt_env/mozaik_paper_test2/bin/activate'), {
    #CombinationParameterSearch(SlurmSequentialBackend(num_threads=1, num_mpi=16,slurm_options=['--hint=nomultithread','-x w[1-2,9,11-12]'],path_to_mozaik_env='/home/cagnol/virt_env/mozaik_paper_test/bin/activate'), {
      'trial' : [1],
      #'sheets.l4_cortex_inh.params.cell.params.cm': [0.0275],
      #'sheets.l23_cortex_inh.params.cell.params.cm': [0.03],
      #'only_afferent' : [True],
      #'sheets.l4_cortex_exc.params.artificial_stimulators.background_act.params.exc_firing_rate' : [720],
      #'sheets.l4_cortex_exc.params.artificial_stimulators.background_act.params.exc_weight' : [0.000555,0.00056],
      #'sheets.l4_cortex_exc.L4ExcL4ExcConnection.short_term_plasticity.tau_psc' : [2],
      #'sheets.l23_cortex_exc.L4ExcL23ExcConnection.base_weight' : [0.0007],
      #'sheets.l4_cortex_inh.AfferentConnection.gauss_coefficient' : [0.5],
      #'sheets.l4_cortex_inh.L4InhL4InhConnection.base_weight' : [0.0005,0.00075,0.0009,0.001],
      #'sheets.l4_cortex_exc.L4ExcL4InhConnection.base_weight' : [0.00025,0.00028,0.00031,0.00035],
      #'sheets.l4_cortex_exc.density' : [1000,1500],
      #'sheets.l4_cortex_inh.AfferentConnection.base_weight' : [0.0012,0.0013,0.0014,0.0015,0.0016,0.0017],
      #'sheets.l4_cortex_inh.AfferentConnection.gauss_coefficient' : [0.085,1000000],
      #'sheets.l4_cortex_exc.L4ExcL4InhConnection.weight_functions.f1.params.sigma' : [10],
      #'sheets.l23_cortex_inh.L4ExcL23InhConnection.weight_functions.f2.params.sigma' : [1.3],
      #'sheets.l23_cortex_exc.L23ExcL23InhConnection.weight_functions.f1.params.sigma' : [1.3],
      #'only_afferent': [True],
      #'steps_get_data': [10000],
      #'sheets.l4_cortex_inh.params.cell.params.cm': [0.0275],
      #'pynn_seed': [5,995],
      #'sheets.l4_cortex_exc.L4ExcL4InhConnection.weight_functions.f1.params.sigma' : [2,3,4,5],
      #'pynn_seed' : [995,263,1503,1701,1947,619,811],
    }).run_parameter_search()

