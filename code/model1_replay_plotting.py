#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 29 2021

@author: wbraun
"""

#check python version
# import sys
# print(sys.path)
# print(sys.version)

###
#from replay_disinhibition_synfire_structured4.ipynb, replay without dendritic spikes, just with model 1
#from replay_model_1.ipynb
###


from brian2 import *
from brian2 import seed
import numpy as np
#clear_cache('cython')

prefs.codegen.target = 'cython'

import matplotlib.pyplot as pl
import matplotlib as mpl

from matplotlib.ticker import ScalarFormatter
from matplotlib.ticker import MaxNLocator
from scipy import signal

##############
#plotting stuff
from matplotlib import rc
#matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
#rc('font',**{'family':'sans-serif','sans-serif':['Arial']})

#matplotlib.rcParams['mathtext.fontset'] = 'custom'
#matplotlib.rcParams['mathtext.rm'] = 'Arial'
#matplotlib.rcParams['mathtext.it'] = 'Arial'

pl.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial"]})

#https://stackoverflow.com/questions/43248336/matplotlib-tick-labels-are-inconsist-with-font-setting-latex-text-example
#https://stackoverflow.com/questions/29188757/matplotlib-specify-format-of-floats-for-tick-labels
fmt = matplotlib.ticker.StrMethodFormatter("{x:,.0f}")

axis_label_size = 30
axis_number_size = 25

#sharp wave profiles
def gaussian_form(t, sigma, mu):
    return (1./(np.sqrt(2.*np.pi*sigma**(2))))*np.exp(-(t-mu)**(2)/(2.0*sigma**(2)))

def rectangular_form(t, sigma, mu):
    if np.abs(t-mu) <= sigma:
        x = 1.
    else:
        x = 0

    return x

def spike_generation(plot_indicator, amplitude_initial, CV_initial, n_excited_initial, amplitude_bulk, CV_bulk, n_excited_bulk):
    plotting = plot_indicator
    spike_computation = True
    printing = True
    print('===============================')
    print('parameters:', amplitude_initial, CV_initial, n_excited_initial, amplitude_bulk, CV_bulk, n_excited_bulk)
    print('===============================')

    start_scope()
    #np.random.seed()
    seed(1)
    N_e = 12000
    N_i = 200 
    print('neuron numbers E/I', N_e, N_i)
    E_rest_i = -65.*mV
    E_I_i = -75.*mV
    C_i = 100.0*pF
    gl_i = 10.0*nS

    #E population
    E_e = 0.*mV
    E_i = -68.*mV
    E_rest = -67.*mV
    C = 275.0*pF
    gl = 25.*nS
    print("E membrane time constant:", C/gl)
    
    #synaptic parameters
    #E-synapses
    #"AMPA on pyramidal cells" (E to E)
    tau_r_E_e = 0.5*ms
    tau_d_E_e = 1.8*ms
    g_peak_E_e = 0.9*nS

    #"GABA on CA1 pyramidal cells" (I to E)
    tau_r_E_i = 0.4*ms
    tau_d_E_i = 2.0*ms
    g_peak_E_i = 9.0*nS 

    #I-synapses
    #"AMPA on interneurons" (E to I)
    tau_r_I_e = 0.5*ms
    tau_d_I_e = 1.2*ms 
    g_peak_I_e = 1.0*nS 

    #"GABA on interneurons" (I to I)
    tau_r_I_i = 0.45*ms
    tau_d_I_i = 1.2*ms
    g_peak_I_i = 5.*nS

    invpeak_I_e = (tau_r_I_e/tau_d_I_e) ** (tau_d_I_e/(tau_r_I_e-tau_d_I_e))
    invpeak_I_i = (tau_r_I_i/tau_d_I_i) ** (tau_d_I_i/(tau_r_I_i-tau_d_I_i))

    invpeak_E_e = (tau_r_E_e/tau_d_E_e) ** (tau_d_E_e/(tau_r_E_e-tau_d_E_e))
    invpeak_E_i = (tau_r_E_i/tau_d_E_i) ** (tau_d_E_i/(tau_r_E_i-tau_d_E_i))

    #conductances for the E cells
    #AMPA
    syn_eqs_E_e = Equations('''dg_e/dt = (invpeak_E_e*s_e - g_e)/tau_d_E_e : siemens
                    ds_e/dt = -s_e/tau_r_E_e : siemens''')
    #GABA
    syn_eqs_E_i = Equations('''dg_i/dt = (invpeak_E_i*s_i - g_i)/tau_d_E_i : siemens
                    ds_i/dt = -s_i/tau_r_E_i : siemens''')

    #conductances for the I cells
    #AMPA
    syn_eqs_I_e = Equations('''dg_e/dt = (invpeak_I_e*s_e - g_e)/tau_d_I_e : siemens
                    ds_e/dt = -s_e/tau_r_I_e : siemens''')
    #GABA
    syn_eqs_I_i = Equations('''dg_i/dt = (invpeak_I_i*s_i - g_i)/tau_d_I_i : siemens
                    ds_i/dt = -s_i/tau_r_I_i : siemens''')

    ####
    peak_time_zero_mean = 40.0

    sigma_noise = 1.0*mV
    
    n_groups = 10 
    neurons_in_group_e = int(N_e/n_groups)
    neurons_in_group_i = int(N_i/n_groups)
    print('number of neurons in E/ I groups', neurons_in_group_e, neurons_in_group_i)
    N_bulk = int(N_e - N_e/n_groups)
    N_initial = int(N_e/n_groups)

    #print(N_bulk, N_initial)
    
    #set up sharp waves for E groups
    peak_time_zero_mean = 40.0

    time_step_array = 0.01 #ms
    defaultclock.dt = time_step_array*ms
    time_for_gaussian = np.arange(0.0, 100 + defaultclock.dt/ms, time_step_array)
    
    offset_bulk = 16
    offset_initial = 3.0
    
    peak_time_bulk = peak_time_zero_mean + offset_bulk
    peak_time_initial = peak_time_zero_mean - offset_initial
    

    temporal_sigma_bulk = 13.0 #16
    temporal_sigma_initial = 4.5 
    normalization_bulk = 1./np.sqrt(2.*np.pi*temporal_sigma_bulk**(2))
    normalization_initial = 1./np.sqrt(2.*np.pi*temporal_sigma_initial**(2))

    amplitude_CV_bulk = CV_bulk 
    amplitude_CV_initial = CV_initial
    
    mean_intensity_bulk = amplitude_bulk
    mean_intensity_initial = amplitude_initial

    E_cells_bulk = np.arange(int(N_e - N_e/n_groups))
    E_cells_initial = np.arange(int(N_e/n_groups))
    #np.random.seed()
    shuffled_E_cells_bulk = np.random.permutation(E_cells_bulk)
    shuffled_E_cells_bulk = shuffled_E_cells_bulk[0:n_excited_bulk]
                                
    shuffled_E_cells_initial = np.random.permutation(E_cells_initial)
    shuffled_E_cells_initial = shuffled_E_cells_initial[0:n_excited_initial]

    active_neurons_bulk = np.zeros(int(N_e - N_e/n_groups))
    active_neurons_bulk[shuffled_E_cells_bulk] = 1.0
                                
    active_neurons_initial = np.zeros(int(N_e/n_groups))
    active_neurons_initial[shuffled_E_cells_initial] = 1.0


    array_gaussian_bulk = []
    amplitude_bulk = np.random.normal(mean_intensity_bulk, amplitude_CV_bulk*mean_intensity_bulk , int(N_e - N_e/n_groups))
    amplitude_bulk[np.where(amplitude_bulk<=0)[0]] = 0      
                            
    array_gaussian_initial = []
    amplitude_initial = np.random.normal(mean_intensity_initial, amplitude_CV_initial*mean_intensity_initial , int(N_e/n_groups))
    amplitude_initial[np.where(amplitude_initial<=0)[0]] = 0

    amplitude_bulk *= active_neurons_bulk
    amplitude_initial *= active_neurons_initial

    for j in np.arange(0, len(time_for_gaussian)):
        random_activation_bulk = np.zeros(len(amplitude_bulk))
        random_activation_bulk = (amplitude_bulk)*(1./normalization_bulk)*(gaussian_form(time_for_gaussian[j], temporal_sigma_bulk, peak_time_bulk))
        array_gaussian_bulk.append(random_activation_bulk) 
                                
        random_activation_initial = np.zeros(len(amplitude_initial))
        random_activation_initial = (amplitude_initial)*(1./normalization_initial)*(gaussian_form(time_for_gaussian[j], temporal_sigma_initial, peak_time_initial))
        array_gaussian_initial.append(random_activation_initial) 

        
    g_chr_bulk = TimedArray(array_gaussian_bulk, dt = time_step_array*ms)
    g_chr_initial = TimedArray(array_gaussian_initial, dt = time_step_array*ms)



    eqs_e_bulk = (Equations('''dV/dt = (gl*(E_rest - V) + (g_e + g_chr_bulk(t,i)*nS)*(E_e - V) + g_i*(E_i - V) +I_app_e)/C + sigma_noise*sqrt(2/(C/gl))*xi : volt (unless refractory) \
       I_app_e: ampere''') + syn_eqs_E_e + syn_eqs_E_i)
    
    eqs_e_initial = (Equations('''dV/dt = (gl*(E_rest - V) + (g_e + g_chr_initial(t,i)*nS)*(E_e - V) + g_i*(E_i - V) +I_app_e)/C + sigma_noise*sqrt(2/(C/gl))*xi : volt (unless refractory) \
       I_app_e: ampere''') + syn_eqs_E_e + syn_eqs_E_i)


    
    group_e_bulk = NeuronGroup(N_bulk,  eqs_e_bulk, threshold='V>=-50*mV', reset='V=E_rest', refractory= 2.0*ms, method = 'euler')
    group_e_initial = NeuronGroup(N_initial,  eqs_e_initial, threshold='V>=-50*mV', reset='V=E_rest', refractory= 2.0*ms, method = 'euler')
    
  
    eqs_i = (Equations('''dV/dt = (gl_i*(E_rest_i - V) + (g_e )*(E_e - V) + g_i*(E_I_i - V))/C_i + sigma_noise*sqrt(2/(C_i/gl_i))*xi: volt (unless refractory)''')+ syn_eqs_I_e + syn_eqs_I_i )
    

    group_i = NeuronGroup(N_i, eqs_i, threshold='V>=-52*mV', reset='V=E_rest_i', refractory= 1.0*ms, method = 'euler')
    

    #synapses, e.g. from E to I
    #kick sizes are g_peak
    syn_ee_bulk = Synapses(group_e_bulk, group_e_bulk, on_pre='s_e += g_peak_E_e', delay = 1.0*ms)
    syn_ee_initial = Synapses(group_e_initial, group_e_initial, on_pre='s_e += g_peak_E_e', delay = 1.0*ms)
    
    syn_ii = Synapses(group_i, group_i, on_pre='s_i += g_peak_I_i', delay = 1.0*ms)

    syn_ei_bulk = Synapses(group_e_bulk, group_i, on_pre='s_e += g_peak_I_e', delay = 1.0*ms)
    syn_ie_bulk = Synapses(group_i, group_e_bulk, on_pre='s_i += g_peak_E_i', delay = 1.0*ms)
    
    syn_ei_initial = Synapses(group_e_initial, group_i, on_pre='s_e += g_peak_I_e', delay = 1.0*ms)
    syn_ie_initial = Synapses(group_i, group_e_initial, on_pre='s_i += g_peak_E_i', delay = 1.0*ms)
   
    #seed()

    #random connectivity
    syn_ee_bulk.connect(p = 197./N_bulk)
    syn_ee_initial.connect(p = 197./N_initial)
    syn_ii.connect(p = 40./N_i)

    
    structured_EI_indicator = True
    if structured_EI_indicator:
        print('++++++++++setting up EI/ IE connectivity++++++++++') 
        #structured E-to-I connectivity
        
        def I_indices(group_index):
            return np.arange(group_index*neurons_in_group_i, (group_index+1)*neurons_in_group_i)
        
        def E_indices(group_index):
            return np.arange(group_index*neurons_in_group_e, (group_index+1)*neurons_in_group_e)
    
        #initial E-to-I connectivity
        k_initial = 1
        for k_connect in np.arange(0,n_groups):
            if k_connect != k_initial:
                #print('corresponding I group index', k_connect)
                for l in I_indices(k_connect):
                        syn_ei_initial.connect(i=np.arange(N_initial), j=l, n=1)
                
        #bulk E-to-I connecitivity
        #group E_k connects to all groups of I cells, but not I_k+1
        for k in np.arange(0, n_groups-1):
            #print('### E group index ###', k)
            for k_connect in np.arange(0,n_groups):
                if k_connect != k+2:
                    #print('corresponding I group index', k_connect)
                    for l in I_indices(k_connect):
                        syn_ei_bulk.connect(i=np.arange(k*neurons_in_group_e, (k+1)*neurons_in_group_e), j=l, n=1)
                    

        #structured I-to-E connectivity
        print('###setting up I-to-E connectivity###')
        
        #iniital I-to-E connectivity
        #print('### I group index ###', k)
        k_connect_initial = 0
        for l in I_indices(k_connect_initial):
            syn_ie_initial.connect(i= l, j= np.arange(N_initial), n = 1) #group I_0 connects to group E_initial
                
        #bulk I-to-E connectivity
        for k in np.arange(0, n_groups-1):
            k_connect = k+1 #index shift because we have two neuron groups: group I_1 connects to group E_0 in the bulk group
            for l in I_indices(k_connect):
                syn_ie_bulk.connect(i= l, j= np.arange((k_connect-1)*neurons_in_group_e, (k_connect)*neurons_in_group_e), n = 1)

        
    
    #np.random.seed()
    #seed()
    group_e_bulk.V = np.random.normal(E_rest/mV, 0.1, N_bulk)*mV
    group_e_initial.V = np.random.normal(E_rest/mV, 0.1, N_initial)*mV
    group_i.V = np.random.normal(E_rest_i/mV, 0.1, N_i)*mV
    group_e_bulk.I_app_e = 0.0*nA
    group_e_initial.I_app_e = 0.0*nA

    
    if spike_computation:
        M_e_bulk = SpikeMonitor(group_e_bulk)
        M_e_initial = SpikeMonitor(group_e_initial)
        M_i = SpikeMonitor(group_i)

    LFP_e_bulk = PopulationRateMonitor(group_e_bulk)
    LFP_e_initial = PopulationRateMonitor(group_e_initial)
    LFP_i = PopulationRateMonitor(group_i)


    duration = 0.1*second

    #seed() #different Poisson noise

    run(duration, report = 'text')
    
    if spike_computation:
        spike_trains_E = M_e_bulk.spike_trains()
        spike_trains_E_initial = M_e_initial.spike_trains()
        spike_trains_I = M_i.spike_trains()
        number_of_spikes_E_bulk = np.zeros(N_bulk)
        number_of_spikes_E_initial = np.zeros(N_initial)
        number_of_spikes_I = np.zeros(N_i)
        delta_t = 50.0
        t_min = peak_time_zero_mean - delta_t
        t_max = peak_time_zero_mean + delta_t

        #spike_times_E = {}
        for neuron in np.arange(N_bulk):
            spike_times_neuron_E = spike_trains_E[neuron]/ms
            number_of_spikes_E_bulk[neuron] = len(spike_times_neuron_E)
            #print(spike_times_neuron_E)
            
        #added 16.11.21
        for neuron in np.arange(N_initial):
            spike_times_neuron_E_initial = spike_trains_E_initial[neuron]/ms
            number_of_spikes_E_initial[neuron] = len(spike_times_neuron_E_initial)
            
        number_of_spikes_E_total = np.hstack([number_of_spikes_E_bulk, number_of_spikes_E_initial])


        #spike_times_I = {}
        for neuron in np.arange(N_i):
            spike_times_neuron_I = spike_trains_I[neuron]/ms
            number_of_spikes_I[neuron] = len(spike_times_neuron_I)

        #print number_of_spikes_E, number_of_spikes_I
        mean_spikes_E_bulk, mean_spikes_I = np.around(mean(number_of_spikes_E_bulk),1), np.around(mean(number_of_spikes_I),1)

        mean_spikes_E_nonsilent_bulk, mean_spikes_I_nonsilent =  np.around(np.mean(number_of_spikes_E_bulk[np.where(number_of_spikes_E_bulk != 0)[0]]),3), np.around(np.mean(number_of_spikes_I[np.where(number_of_spikes_I != 0)[0]]),1)
        total_spikes_E_bulk, total_spikes_I =  np.around(np.sum(number_of_spikes_E_bulk[np.where(number_of_spikes_E_bulk != 0)[0]]),1) ,np.around(np.sum(number_of_spikes_I[np.where(number_of_spikes_I != 0)[0]]),1)
        
        mean_spikes_E_nonsilent_total =  np.around(np.mean(number_of_spikes_E_total[np.where(number_of_spikes_E_total != 0)[0]]), 3)
        silent_percentage_of_E_population_bulk =  len(np.where(number_of_spikes_E_bulk == 0)[0])/ np.float(N_bulk)
        silent_percentage_of_I_population =  len(np.where(number_of_spikes_I == 0)[0])/ np.float(N_i)
        

    #####################
    #plotting results
    #####################
    resolution_LFP = 0.5*ms #0.2 is standard value
    population_rate_e_bulk = LFP_e_bulk.smooth_rate(window='gaussian', width=resolution_LFP)/Hz #'flat'
    population_rate_e_initial = LFP_e_initial.smooth_rate(window='gaussian', width=resolution_LFP)/Hz #'flat'
    population_rate_i = LFP_i.smooth_rate(window='gaussian', width=resolution_LFP)/Hz

    #compute PSD of rates between 40 and 60 ms
    t_min_rate = int(0/time_step_array) #40, 61
    t_max_rate = int(101/time_step_array)

    time_rate_i = LFP_i.t/ms
    sampling_frequency_i = 1000./(time_rate_i[1] - time_rate_i[0]) #Hz
    freq_i, spec_i = signal.welch(population_rate_i[t_min_rate:t_max_rate], sampling_frequency_i, nperseg = len(time_rate_i[t_min_rate:t_max_rate])/5., nfft = 5*len(time_rate_i[t_min_rate:t_max_rate]), window = 'hann', scaling = 'density')
    freq_max_i = freq_i[np.argmax(spec_i)]

    time_rate_e = LFP_e_bulk.t/ms
    sampling_frequency_e = 1000./(time_rate_e[1] - time_rate_e[0])
    freq_e, spec_e = signal.welch(population_rate_e_bulk[t_min_rate:t_max_rate], sampling_frequency_e, nperseg = len(time_rate_e[t_min_rate:t_max_rate])/5., nfft = 5*len(time_rate_e[t_min_rate:t_max_rate]), window = 'hann', scaling = 'density')
    freq_max_e = freq_e[np.argmax(spec_e)]
    
    print ("mean number of spikes per neuron for non-silent E (bulk), I population:", mean_spikes_E_nonsilent_bulk, mean_spikes_I_nonsilent)
    print ("number of silent E neurons (bulk):", len(np.where(number_of_spikes_E_bulk == 0)[0]), "percentage", silent_percentage_of_E_population_bulk)
    print ("number of silent I neurons:", len(np.where(number_of_spikes_I == 0)[0]), "percentage", silent_percentage_of_I_population)
    print ("number of firing E neurons (bulk)", len(np.where(number_of_spikes_E_bulk != 0)[0]), 'excited E neurons (bulk):', len(shuffled_E_cells_bulk))
    print ("mean number of spikes per firing E neuron:", mean_spikes_E_nonsilent_bulk)
    print ("mean number of spikes per firing I neuron:", mean_spikes_I_nonsilent)
    print ("total E spikes (bulk):", total_spikes_E_bulk, M_e_bulk.num_spikes)
    print ("total E spikes (initial):", M_e_initial.num_spikes)
    print("E spikes from bulk and initial", M_e_bulk.num_spikes + M_e_initial.num_spikes)
    print ("total I spikes",  total_spikes_I, M_i.num_spikes)
    print('Global oscillation frequencies (I/E (bulk)):', freq_max_i,",",  freq_max_e)


    if plotting:
        
        fig = pl.figure(0, figsize = (7.5, 4.0), dpi = 300)
        ax0 = fig.add_subplot(311) 
        ax0.annotate('B', xy=(-0.12, 1.2), xycoords='axes fraction', fontsize = 24.0) #36.0

        #print ("plotting time-dependent excitatory conductances")
        array_gaussian_plotting_bulk  = np.asarray(array_gaussian_bulk)
        array_gaussian_plotting_initial  = np.asarray(array_gaussian_initial)
      
        for index_cell, j in enumerate(shuffled_E_cells_bulk):
            ax0.plot(time_for_gaussian, array_gaussian_plotting_bulk[:, j], '-', linewidth = 0.5, alpha = 0.8, color = 'k')
            
        for index_cell, j in enumerate(shuffled_E_cells_initial):
            ax0.plot(time_for_gaussian, array_gaussian_plotting_initial[:, j], '-', linewidth = 0.5, alpha = 0.8, color = 'm')

        pl.axvline(x = peak_time_zero_mean + offset_bulk, color = 'r', linestyle = '--', linewidth = 1.0)
        pl.axvline(x = peak_time_zero_mean - offset_initial, color = 'k', linestyle = '--', linewidth = 1.0)
        
        pl.ylabel(r'$g_{\text{ext}}~[\text{nS}]$', fontsize = 12.0)

        pl.tick_params(labelsize=12.0)
        ax0.xaxis.set_major_formatter(fmt)
        ax0.yaxis.set_major_formatter(fmt)
        #ax0.set_xticks(np.arange(20, 100, step=10))
        pl.xlim(20, 100)
        ax0.xaxis.set_ticklabels([])

        ax1 = fig.add_subplot(312) #412
        ax1.plot(M_e_bulk.t/ms, M_e_bulk.i + N_initial, 'r.', markersize = 2.0)
        ax1.plot(M_e_initial.t/ms, M_e_initial.i, 'm.', markersize = 2.0)
        
        for k in np.arange(n_groups):
            pl.axhline(y = k*neurons_in_group_e, color = 'r', linestyle = '--', linewidth = 0.3)
        
        #pl.xlim(0, duration/ms)
        pl.xticks(np.arange(0, duration/ms, step=10)) #step = 5
        #pl.axvline(x = 50, color = 'k', linestyle = '--', linewidth = 3.0)
        pl.axvline(x = peak_time_zero_mean + offset_bulk, color = 'r', linestyle = '--', linewidth = 1.0)
        pl.axvline(x = peak_time_zero_mean - offset_initial, color = 'k', linestyle = '--', linewidth = 1.0)
        pl.ylabel('E cell', fontsize = 12.0)

        pl.tick_params(labelsize=12.0)
        ax1.xaxis.set_major_formatter(fmt)
        ax1.yaxis.set_major_formatter(fmt)
        pl.xlim(20, 100)
        ax1.set_xticks(np.arange(20, 100, step=10))
        ax1.xaxis.set_ticklabels([])


        ax2 = fig.add_subplot(313)
        ax2.plot(M_i.t/ms, M_i.i, 'b.', markersize = 2.0)
        
        for k in np.arange(n_groups):
            pl.axhline(y = k*neurons_in_group_i, color = 'b', linestyle = '--', linewidth = 0.3)
            
        #pl.axhline(y = N_i, color = 'b', linestyle = '--', linewidth = 1.0)
        pl.axvline(x = peak_time_zero_mean + offset_bulk, color = 'r', linestyle = '--', linewidth = 1.0)
        pl.axvline(x = peak_time_zero_mean - offset_initial, color = 'k', linestyle = '--', linewidth = 1.0)
        
        pl.ylabel('I cell', fontsize = 12.0)
        pl.xlabel(r'$t~(\text{ms})$', fontsize = 12.0)

        pl.xlim(20, 100)
        ax2.set_xticks(np.arange(20, 100, step=10))
        pl.tick_params(labelsize=12.0)
        ax2.xaxis.set_major_formatter(fmt)
        ax2.yaxis.set_major_formatter(fmt)
        pl.tight_layout(pad = 1.0)
        fig.savefig("/home/wilhelm/Schreibtisch/Fig9B.png", bbox_inches='tight')
        #pl.show()
        
        fig = pl.figure(1, figsize = (7.5, 1.0), dpi = 300)
        #plots for how many neurons are spiking
        ax6 = fig.add_subplot(121)
        ax6.annotate('C', xy=(-0.12, 1.1), xycoords='axes fraction', fontsize= 24.0) #(-0.12, 1.1), 36.0

        bins_E = np.arange(0, max(number_of_spikes_E_bulk) + 2)
        pl.hist(number_of_spikes_E_total, bins= bins_E, histtype = 'step', color = 'r', linewidth = 5.0, align = 'left', log = True)
        pl.axvline(x =mean_spikes_E_nonsilent_total, color = 'r', linewidth = 5.0, linestyle = '--')
        pl.ylabel('no. of E cells', fontsize = 12.0)
        pl.xlabel('spikes', fontsize = 12.0)
        pl.xticks(np.arange(0, np.max(bins_E), step=1)) #step = 4
        pl.tick_params(labelsize=12.0)
        ax6.xaxis.get_major_formatter()._usetex = False
        ax6.yaxis.get_major_formatter()._usetex = False
        ax6.xaxis.set_major_formatter(fmt)
        ax6.yaxis.set_major_formatter(fmt)

        ax7 = fig.add_subplot(122)
        bins_I = np.arange(0, max(number_of_spikes_I) + 2) 
        pl.hist(number_of_spikes_I, bins= bins_I, histtype = 'step', color = 'b', linewidth = 5.0, align = 'left', log = False)
        pl.axvline(x = mean_spikes_I_nonsilent, color = 'b', linewidth = 5.0, linestyle = '--')
        pl.ylabel('no. of I cells', fontsize = 12.0, labelpad = -0.5)
        pl.xlabel('spikes', fontsize = 12.0)
        #pl.xticks(bins_I)
        pl.xticks(np.arange(0, np.max(bins_I), step=2)) #step =4
        pl.tick_params(labelsize=12.0)
        ax7.xaxis.get_major_formatter()._usetex = False
        ax7.yaxis.get_major_formatter()._usetex = False
        ax7.xaxis.set_major_formatter(fmt)
        ax7.yaxis.set_major_formatter(fmt)

        ax7.yaxis.set_ticklabels([])
        positions = [100]
        ax7.set_yticks([100])
        ax7.yaxis.set_ticklabels(positions)
        pl.tight_layout(pad = 1.0)
        fig.savefig("/home/wilhelm/Schreibtisch/Fig9C.png", bbox_inches='tight')
        pl.show()

    print('===============================')
    return [1]

for MC_simulation in np.arange(1):

    #amplitude_initial, CV_initial, n_excited_initial, amplitude_bulk, CV_bulk, n_excited_bulk
    
    result = spike_generation(plot_indicator = True, amplitude_initial = 20., CV_initial = 0.1, n_excited_initial = 50, amplitude_bulk = 24., CV_bulk = 0.1, n_excited_bulk = 400)

