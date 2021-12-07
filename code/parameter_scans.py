#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 7 2021

@author: wbraun
"""

#############
#############
#This script generates data for Figures 4, 6 and 8. It uses the multiprocessing toolbox and either requires a cluster or a larger desktop computer (more than 10 threads) to finish in reasonable time.
#Script is setup so that its execution will generate data for Fig. 8. It can easily be adapted (lines 1040- 1125) to generate the data for all parameter plots in the manuscript.
#Code for model 1 starts on line 52, for model 1 with feedforward drive to basket cells on line 283, for model 2 on line 518 and code for model 3 on line 748.
#Some computing architectures only work with 'numpy' code generation target set below.
#############
#############


import multiprocessing
from brian2 import *
#from brian2 import seed
import numpy as np
import time
import os
#clear_cache('cython')

#set_device('cpp_standalone', build_on_run = False)

#print "version of brian2", brian2.__version__
#print('version of brian2=', brian2.__version__)
# prefs.codegen.target = 'auto'


prefs.codegen.target = 'numpy'
#prefs.codegen.target = 'cython'

#clear_cache('cython')
#BrianLogger.log_level_debug()


from scipy import signal

import scipy.stats as stats
import scipy.sparse as sparse


printing = False

def gaussian_form(t, sigma, mu):
    return (1./(np.sqrt(2.*np.pi*sigma**(2))))*np.exp(-(t-mu)**(2)/(2.0*sigma**(2)))



#model 1 (Figs. 3 and 4)
def spike_generation_1(params):

    param1, param2 = params[0], params[1]
    n_e = param1
    mean_amplitude = param2

    start_scope()
    seed()
    N_e = 12000
    N_i = 200

    #interneurons
    E_rest_i = -65.*mV
    E_I_i = -75.*mV
    C_i = 100.0*pF
    gl_i = 10.0*nS

    #E population
    E_e = 0.*mV
    E_i = -68.*mV
    E_rest = -67.*mV
    C = 275*pF
    gl = 25.*nS

    #synaptic parameters
    #E-synapses
    #"AMPA on pyramidal cells"
    tau_r_E_e = 0.5*ms
    tau_d_E_e = 1.8*ms
    g_peak_E_e = 0.9*nS

    #The following two parameters are crucial for the generation of high frequency oscillations.
    #"GABA on CA1 pyramidal cells"
    tau_r_E_i = 0.4*ms 
    tau_d_E_i = 2.0*ms 
    g_peak_E_i = 9.0*nS 

    #I-synapses
    #"AMPA on interneurons"
    tau_r_I_e = 0.5*ms
    tau_d_I_e = 1.2*ms
    g_peak_I_e = 3.0*nS 

    #"GABA on interneurons"
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
    peak_time_zero = 50.0
    peak_time_zero_mean = 50.0
    #intensities for sharp waves
    #mean_intensity = 10.0
    #mean_intensity_i = 2.0

    #this time step is also used for the setup of g_chr and the simulation time step
    time_step_array = 0.01 #ms #0.01 is standard
    defaultclock.dt = time_step_array*ms
    ##time from 0 to 100 ms in steps of time_step_array ms
    time_for_gaussian = np.arange(0.0, 100 + defaultclock.dt/ms, time_step_array)

    #print time_for_gaussian

    #value at t = 50 ms
    temporal_sigma = 10.0 #sigma_g in manuscript (Eq. 6)
    normalization = 1./np.sqrt(2.*np.pi*temporal_sigma**(2))

    #CV of amplitude distribution fixed
    amplitude_CV = 0.5 #


    I_app_i = 0.0*nA
    sigma_noise = 1.0*mV


    eqs_e = (Equations('dV/dt = (gl*(E_rest - V) + (g_e + g_chr(t,i)*nS)*(E_e - V) + g_i*(E_i - V) +I_app_e)/C  + sigma_noise*sqrt(2/(C/gl))*xi: volt (unless refractory) \
            I_app_e: ampere') + syn_eqs_E_e + syn_eqs_E_i)

    #standard
    group_e = NeuronGroup(N_e,  eqs_e, threshold='V>=-50*mV', reset='V=E_rest', refractory= 2.0*ms, method = 'euler')


    eqs_i = (Equations('dV/dt = (gl_i*(E_rest_i - V) + (g_e)*(E_e - V) + g_i*(E_I_i - V) + I_app_i)/C_i + sigma_noise*sqrt(2/(C_i/gl_i))*xi: volt (unless refractory)') + syn_eqs_I_e + syn_eqs_I_i)

    #standard
    group_i = NeuronGroup(N_i, eqs_i, threshold='V>=-52*mV', reset='V=E_rest_i', refractory= 1.0*ms, method = 'euler') #previously: refractory = 2*ms



    #synapses, e.g. from E to I
    #kick sizes are g_peak
    syn_ee = Synapses(group_e, group_e, on_pre='s_e += g_peak_E_e', delay = 1.0*ms)
    syn_ii = Synapses(group_i, group_i, on_pre='s_i += g_peak_I_i', delay = 1.0*ms)

    syn_ei = Synapses(group_e, group_i, on_pre='s_e += g_peak_I_e', delay = 1.0*ms)
    syn_ie = Synapses(group_i, group_e, on_pre='s_i += g_peak_E_i', delay = 1.0*ms)

    M_e = SpikeMonitor(group_e)
    M_i = SpikeMonitor(group_i)
    LFP_e = PopulationRateMonitor(group_e)
    LFP_i = PopulationRateMonitor(group_i)

    duration = 0.1*second

    ###
    mean_intensity = mean_amplitude

    #choose a subset of excitatory cells randomly
    np.random.seed()
    E_cells = np.arange(N_e)
    shuffled_E_cells = np.random.permutation(E_cells)
    #shuffled_E_cells = E_cells
    shuffled_E_cells = shuffled_E_cells[0:int(n_e)]
    #print shuffled_E_cells

    active_neurons = np.zeros(N_e)
    active_neurons[shuffled_E_cells] = 1.0
    array_gaussian = []

    numpy.random.seed()
    #set up sharp wave input
    amplitude = np.random.normal(mean_intensity, amplitude_CV*mean_intensity, N_e)
    amplitude *= active_neurons

    for time_index in np.arange(0, len(time_for_gaussian)):
        random_activation = np.zeros(len(amplitude))
        random_activation = (amplitude)*(1./normalization)*(gaussian_form(time_for_gaussian[time_index], temporal_sigma, peak_time_zero_mean))
        array_gaussian.append(random_activation)


    g_chr = TimedArray(array_gaussian, dt = time_step_array*ms)
    ###

    seed()
    np.random.seed()
    syn_ee.connect(p = 197./N_e)
    syn_ii.connect(p = 0.2)
    syn_ei.connect(p = 0.1)
    syn_ie.connect(p = 0.1)

    seed()
    np.random.seed()
    group_e.V = np.random.normal(E_rest/mV, 0.1, N_e)*mV
    group_i.V = np.random.normal(E_rest_i/mV, 0.1, N_i)*mV
    group_e.I_app_e = 0.0*nA


    run(duration)
    #spike count computations, added 170719
    spike_trains_E = M_e.spike_trains()
    spike_trains_I = M_i.spike_trains()
    number_of_spikes_E = np.zeros(N_e)
    number_of_spikes_I = np.zeros(N_i)
    delta_t = 50.0
    t_min = peak_time_zero_mean - delta_t
    t_max = peak_time_zero_mean + delta_t

    spike_times_E = {}
    for neuron in np.arange(N_e):
        spike_times_neuron_E = spike_trains_E[neuron]/ms
        #only count spikes in certain temporal range
        spike_times_neuron_E = spike_times_neuron_E[np.where(spike_times_neuron_E>=t_min)[0]]
        spike_times_neuron_E = spike_times_neuron_E[np.where(spike_times_neuron_E<=t_max)[0]]
        #print(spike_times_neuron_E)
        number_of_spikes_E[neuron] = len(spike_times_neuron_E[np.where(spike_times_neuron_E)[0]])
        spike_times_E[neuron] = spike_times_neuron_E

        #print spike_trains_E[neuron]/ms, number_of_spikes_E[neuron]

    spike_times_I = {}
    for neuron in np.arange(N_i):
        spike_times_neuron_I = spike_trains_I[neuron]/ms
        spike_times_neuron_I = spike_times_neuron_I[np.where(spike_times_neuron_I>=t_min)[0]]
        spike_times_neuron_I = spike_times_neuron_I[np.where(spike_times_neuron_I<=t_max)[0]]
        number_of_spikes_I[neuron] = len(spike_times_neuron_I[np.where(spike_times_neuron_I)[0]])

    mean_spikes_E_nonsilent, mean_spikes_I_nonsilent =  np.around(np.mean(number_of_spikes_E[np.where(number_of_spikes_E != 0)[0]]),1), np.around(np.mean(number_of_spikes_I[np.where(number_of_spikes_I != 0)[0]]),1)

    silent_percentage_of_E_population =  len(np.where(number_of_spikes_E == 0)[0])/ np.float(N_e)
    silent_percentage_of_I_population =  len(np.where(number_of_spikes_I == 0)[0])/ np.float(N_i)

    ############
    #spectrum computations

    resolution_LFP = 0.2*ms
    population_rate_e = LFP_e.smooth_rate(window='gaussian', width=resolution_LFP)/Hz
    population_rate_i = LFP_i.smooth_rate(window='gaussian', width=resolution_LFP)/Hz

    t_min_rate = int(0/time_step_array)
    t_max_rate = int(101/time_step_array)


    time_rate_i = LFP_i.t/ms
    sampling_frequency_i = 1000./(time_rate_i[1] - time_rate_i[0]) #Hz
    factor_spec = 5.
    freq_i, spec_i = signal.welch(population_rate_i[t_min_rate:t_max_rate], sampling_frequency_i, nperseg = len(time_rate_i[t_min_rate:t_max_rate])/factor_spec, nfft = factor_spec*len(time_rate_i), window = 'hann', scaling = 'density')
    freq_max_i = freq_i[np.argmax(spec_i)]
    
    time_rate_e = LFP_e.t/ms
    sampling_frequency_e = 1000./(time_rate_e[1] - time_rate_e[0])
    freq_e, spec_e = signal.welch(population_rate_e[t_min_rate:t_max_rate], sampling_frequency_e, nperseg = len(time_rate_e[t_min_rate:t_max_rate])/factor_spec, nfft = factor_spec*len(time_rate_e), window = 'hann', scaling = 'density')
    freq_max_e = freq_e[np.argmax(spec_e)]

    return [freq_max_i, freq_max_e, mean_spikes_I_nonsilent, mean_spikes_E_nonsilent, 1.-silent_percentage_of_I_population, 1.-silent_percentage_of_E_population]

  
#model 1 with feedforward excitatory drive to I cells
def spike_generation_1_FFDrive_I(params):
    param1, param2 = params[0], params[1]
    n_e = param1
    mean_amplitude = param2
    
    start_scope()
    seed()
    N_e = 12000 
    N_i = 200 #1200 

    #interneurons
    E_rest_i = -65.*mV
    E_I_i = -75.*mV
    C_i = 100.0*pF
    gl_i = 10.0*nS

    #E population
    E_e = 0.*mV
    E_i = -68.*mV 
    E_rest = -67.*mV
    C = 275*pF
    gl = 25.*nS

    #synaptic parameters
    #E-synapses
    #"AMPA on pyramidal cells"
    tau_r_E_e = 0.5*ms
    tau_d_E_e = 1.8*ms
    g_peak_E_e = 0.9*nS

    #"GABA on CA1 pyramidal cells"
    tau_r_E_i = 0.4*ms 
    tau_d_E_i = 2.0*ms
    g_peak_E_i = 9.0*nS 

    #I-synapses
    #"AMPA on interneurons"
    tau_r_I_e = 0.5*ms 
    tau_d_I_e = 1.2*ms 
    g_peak_I_e = 3.0*nS

    #"GABA on interneurons"
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
    peak_time_zero = 50.0
    peak_time_zero_mean = 50.0
    #intensities for sharp waves
    #mean_intensity = 10.0
    mean_intensity_i = 20.0 

    #this time step is also used for the setup of g_chr and the simulation time step
    time_step_array = 0.01 #ms, 0.01 is standard
    defaultclock.dt = time_step_array*ms
    ##time from 0 to 100 ms in steps of time_step_array ms
    time_for_gaussian = np.arange(0.0, 100 + defaultclock.dt/ms, time_step_array)

    #print time_for_gaussian

    #value at t = 50 ms
    temporal_sigma = 10.0 
    normalization = 1./np.sqrt(2.*np.pi*temporal_sigma**(2))

    #CV of amplitude distribution fixed
    amplitude_CV = 0.5
    

    I_app_i = 0.0*nA
    sigma_noise = 1.0*mV

    eqs_e = (Equations('dV/dt = (gl*(E_rest - V) + (g_e + g_chr(t,i)*nS )*(E_e - V) + g_i*(E_i - V) +I_app_e)/C  + sigma_noise*sqrt(2/(C/gl))*xi: volt (unless refractory) \
            I_app_e: ampere') + syn_eqs_E_e + syn_eqs_E_i)

    group_e = NeuronGroup(N_e,  eqs_e, threshold='V>=-50*mV', reset='V=E_rest', refractory= 2.0*ms, method = 'euler')
    #Sharp wave input to basket cells
    eqs_i = (Equations('dV/dt = (gl_i*(E_rest_i - V) + (g_e + g_chr_i(t,i)*nS)*(E_e - V) + g_i*(E_I_i - V) + I_app_i)/C_i + sigma_noise*sqrt(2/(C_i/gl_i))*xi: volt (unless refractory)') + syn_eqs_I_e + syn_eqs_I_i)

    group_i = NeuronGroup(N_i, eqs_i, threshold='V>=-52*mV', reset='V=E_rest_i', refractory= 1.0*ms, method = 'euler')


    #synapses, e.g. from E to I
    #kick sizes are g_peak
    syn_ee = Synapses(group_e, group_e, on_pre='s_e += g_peak_E_e', delay = 1.0*ms)
    syn_ii = Synapses(group_i, group_i, on_pre='s_i += g_peak_I_i', delay = 1.0*ms)

    syn_ei = Synapses(group_e, group_i, on_pre='s_e += g_peak_I_e', delay = 1.0*ms)
    syn_ie = Synapses(group_i, group_e, on_pre='s_i += g_peak_E_i', delay = 1.0*ms) #1.0*ms is standard

    M_e = SpikeMonitor(group_e)
    M_i = SpikeMonitor(group_i)
    LFP_e = PopulationRateMonitor(group_e)
    LFP_i = PopulationRateMonitor(group_i)

    duration = 0.1*second

    ###
    mean_intensity = mean_amplitude
    
    #choose a subset of excitatory cells randomly
    np.random.seed()
    E_cells = np.arange(N_e)
    shuffled_E_cells = np.random.permutation(E_cells)
    #shuffled_E_cells = E_cells
    shuffled_E_cells = shuffled_E_cells[0:int(n_e)]
    #print shuffled_E_cells

    active_neurons = np.zeros(N_e)
    active_neurons[shuffled_E_cells] = 1.0
    array_gaussian = []

    numpy.random.seed()
    #set up sharp wave input
    amplitude = np.random.normal(mean_intensity, amplitude_CV*mean_intensity, N_e) #standard, used always
    amplitude *= active_neurons
    #amplitude[np.where(amplitude<=0)[0]] = 0 #not normally used
    
    #sharp wave input to basket cells
    array_gaussian_i = []
    #amplitude_i = np.random.normal(mean_intensity_i, amplitude_CV*mean_intensity_i, N_i)
    amplitude_i = mean_intensity_i*np.ones(N_i) #used in December 2020 for control simulations

    for time_index in np.arange(0, len(time_for_gaussian)):
        random_activation = np.zeros(len(amplitude))
        random_activation = (amplitude)*(1./normalization)*(gaussian_form(time_for_gaussian[time_index], temporal_sigma, peak_time_zero_mean)) #standard
        array_gaussian.append(random_activation)
        
        #used in December 2020
        random_activation_i = np.zeros(len(amplitude_i))
        random_activation_i = (amplitude_i)*(1./normalization)*(gaussian_form(time_for_gaussian[time_index], temporal_sigma, peak_time_zero_mean))
        array_gaussian_i.append(random_activation_i)

    g_chr = TimedArray(array_gaussian, dt = time_step_array*ms)
    g_chr_i = TimedArray(array_gaussian_i, dt = time_step_array*ms)
    ###
    
    seed()
    np.random.seed()
    syn_ee.connect(p = 197./N_e)
    syn_ii.connect(p = 0.2)
    syn_ei.connect(p = 0.1)
    syn_ie.connect(p = 0.1)
    seed()
    np.random.seed()
    group_e.V = np.random.normal(E_rest/mV, 0.1, N_e)*mV
    group_i.V = np.random.normal(E_rest_i/mV, 0.1, N_i)*mV
    group_e.I_app_e = 0.0*nA
    
    run(duration)
    #spike count computations, added 170719
    spike_trains_E = M_e.spike_trains()
    spike_trains_I = M_i.spike_trains()
    number_of_spikes_E = np.zeros(N_e)
    number_of_spikes_I = np.zeros(N_i)
    delta_t = 50.0
    t_min = peak_time_zero_mean - delta_t
    t_max = peak_time_zero_mean + delta_t

    spike_times_E = {}
    for neuron in np.arange(N_e):
        spike_times_neuron_E = spike_trains_E[neuron]/ms
        #only count spikes in certain temporal range
        spike_times_neuron_E = spike_times_neuron_E[np.where(spike_times_neuron_E>=t_min)[0]]
        spike_times_neuron_E = spike_times_neuron_E[np.where(spike_times_neuron_E<=t_max)[0]]
        #print(spike_times_neuron_E)
        number_of_spikes_E[neuron] = len(spike_times_neuron_E[np.where(spike_times_neuron_E)[0]])
        #spike_times_E[neuron] = spike_times_neuron_E

        #print spike_trains_E[neuron]/ms, number_of_spikes_E[neuron]

    spike_times_I = {}
    for neuron in np.arange(N_i):
        spike_times_neuron_I = spike_trains_I[neuron]/ms
        spike_times_neuron_I = spike_times_neuron_I[np.where(spike_times_neuron_I>=t_min)[0]]
        spike_times_neuron_I = spike_times_neuron_I[np.where(spike_times_neuron_I<=t_max)[0]]
        number_of_spikes_I[neuron] = len(spike_times_neuron_I[np.where(spike_times_neuron_I)[0]])

    #print number_of_spikes_E, number_of_spikes_I
    #mean_spikes_E, mean_spikes_I = np.around(mean(number_of_spikes_E),1), np.around(mean(number_of_spikes_I),1)

    mean_spikes_E_nonsilent, mean_spikes_I_nonsilent =  np.around(np.mean(number_of_spikes_E[np.where(number_of_spikes_E != 0)[0]]),1), np.around(np.mean(number_of_spikes_I[np.where(number_of_spikes_I != 0)[0]]),1)

    silent_percentage_of_E_population =  len(np.where(number_of_spikes_E == 0)[0])/ np.float(N_e)
    silent_percentage_of_I_population =  len(np.where(number_of_spikes_I == 0)[0])/ np.float(N_i)

    ############
    #spectrum computations

    resolution_LFP = 0.2*ms #1.0*ms
    population_rate_e = LFP_e.smooth_rate(window='gaussian', width=resolution_LFP)/Hz
    population_rate_i = LFP_i.smooth_rate(window='gaussian', width=resolution_LFP)/Hz


    t_min_rate = int(0/time_step_array)
    t_max_rate = int(101/time_step_array)

    time_rate_i = LFP_i.t/ms
    sampling_frequency_i = 1000./(time_rate_i[1] - time_rate_i[0]) #Hz
    factor_spec = 5.
    freq_i, spec_i = signal.welch(population_rate_i[t_min_rate:t_max_rate], sampling_frequency_i, nperseg = len(time_rate_i[t_min_rate:t_max_rate])/factor_spec, nfft = 5*len(time_rate_i), window = 'hann', scaling = 'density')
    freq_max_i = freq_i[np.argmax(spec_i)]
    
    time_rate_e = LFP_e.t/ms
    sampling_frequency_e = 1000./(time_rate_e[1] - time_rate_e[0])
    freq_e, spec_e = signal.welch(population_rate_e[t_min_rate:t_max_rate], sampling_frequency_e, nperseg = len(time_rate_e[t_min_rate:t_max_rate])/factor_spec, nfft = 5*len(time_rate_e), window = 'hann', scaling = 'density')
    freq_max_e = freq_e[np.argmax(spec_e)]

    return [freq_max_i, freq_max_e, mean_spikes_I_nonsilent, mean_spikes_E_nonsilent, 1.-silent_percentage_of_I_population, 1.-silent_percentage_of_E_population]


#model 2 (Figs. 5 and 6)
def spike_generation_2(params):

    param1, param2 = params[0], params[1]
    n_e = param1
    mean_amplitude = param2

    start_scope()
    seed()
    N_e = 12000
    N_i = 200

    #interneurons
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

    #synaptic parameters
    #E-synapses
    #"AMPA on pyramidal cells"
    tau_r_E_e = 0.5*ms
    tau_d_E_e = 1.8*ms
    g_peak_E_e = 0.9*nS

    #"GABA on CA1 pyramidal cells"
    tau_r_E_i = 0.4*ms 
    tau_d_E_i = 2.0*ms
    g_peak_E_i = 9.0*nS

    #I-synapses
    #"AMPA on interneurons"
    tau_r_I_e = 0.5*ms
    tau_d_I_e = 1.2*ms
    g_peak_I_e = 3.0*nS

    #"GABA on interneurons"
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
    peak_time_zero_mean = 50.0
    #temporal variation around mean
    sigma_t = 10.0
    np.random.seed()
    peak_time_zero =  np.random.normal(peak_time_zero_mean, sigma_t, N_e)


    #this time step is also used for the setup of g_chr and the simulation time step
    time_step_array = 0.01 #ms
    ##time from 0 to 100 ms in steps of time_step_array ms
    defaultclock.dt = time_step_array*ms
    time_for_gaussian = np.arange(0.0, 100+ defaultclock.dt/ms, time_step_array)

    #value at t = 50 ms- width of sharp wave, also called sigma_g in Eq. 6 in manuscript.
    temporal_sigma = 3.0
    normalization = 1./np.sqrt(2.*np.pi*temporal_sigma**(2))


    I_app_i = 0.0*nA
    sigma_noise = 1.0*mV


    eqs_e = (Equations('dV/dt = (gl*(E_rest - V) + (g_e + g_chr(t,i)*nS )*(E_e - V) + g_i*(E_i - V) +I_app_e)/C  + sigma_noise*sqrt(2/(C/gl))*xi: volt (unless refractory) \
            I_app_e: ampere') + syn_eqs_E_e + syn_eqs_E_i)


    group_e = NeuronGroup(N_e,  eqs_e, threshold='V>=-50*mV', reset='V=E_rest', refractory= 2.0*ms, method = 'euler')

    eqs_i = (Equations('dV/dt = (gl_i*(E_rest_i - V) + (g_e)*(E_e - V) + g_i*(E_I_i - V) + I_app_i)/C_i + sigma_noise*sqrt(2/(C_i/gl_i))*xi: volt (unless refractory)') + syn_eqs_I_e + syn_eqs_I_i)


    group_i = NeuronGroup(N_i, eqs_i, threshold='V>=-52*mV', reset='V=E_rest_i', refractory= 1.0*ms, method = 'euler')


    #synapses, e.g. from E to I
    #kick sizes are g_peak
    syn_ee = Synapses(group_e, group_e, on_pre='s_e += g_peak_E_e', delay = 1.0*ms)
    syn_ii = Synapses(group_i, group_i, on_pre='s_i += g_peak_I_i', delay = 1.0*ms)

    syn_ei = Synapses(group_e, group_i, on_pre='s_e += g_peak_I_e', delay = 1.0*ms)
    syn_ie = Synapses(group_i, group_e, on_pre='s_i += g_peak_E_i', delay = 1.0*ms)

    M_e = SpikeMonitor(group_e)
    M_i = SpikeMonitor(group_i)
    LFP_e = PopulationRateMonitor(group_e)
    LFP_i = PopulationRateMonitor(group_i)


    duration = 0.1*second


    ###
    mean_intensity = mean_amplitude
    np.random.seed()
    #choose a subset of excitatory cells randomly
    E_cells = np.arange(N_e)
    shuffled_E_cells = np.random.permutation(E_cells)
    #shuffled_E_cells = E_cells
    shuffled_E_cells = shuffled_E_cells[0:int(n_e)]
    #print shuffled_E_cells

    active_neurons = np.zeros(N_e)
    active_neurons[shuffled_E_cells] = 1.0
    array_gaussian = []


    amplitude = mean_intensity*np.ones(N_e)

    amplitude *= active_neurons


    for time_index in np.arange(0, len(time_for_gaussian)):
        random_activation = np.zeros(len(amplitude))
        random_activation = (amplitude)*(1./normalization)*(gaussian_form(time_for_gaussian[time_index], temporal_sigma, peak_time_zero))
        array_gaussian.append(random_activation)

    g_chr = TimedArray(array_gaussian, dt = time_step_array*ms)



    seed()
    syn_ee.connect(p = 197./N_e)
    syn_ii.connect(p = 0.2)
    syn_ei.connect(p = 0.1)
    syn_ie.connect(p = 0.1)




    np.random.seed()

    group_e.V = np.random.normal(E_rest/mV, 0.1, N_e)*mV
    group_i.V = np.random.normal(E_rest_i/mV, 0.1, N_i)*mV
    group_e.I_app_e = 0.0*nA


    run(duration)
    #spike count computations, added 170719
    spike_trains_E = M_e.spike_trains()
    spike_trains_I = M_i.spike_trains()
    number_of_spikes_E = np.zeros(N_e)
    number_of_spikes_I = np.zeros(N_i)
    delta_t = 50.0
    t_min = peak_time_zero_mean - delta_t
    t_max = peak_time_zero_mean + delta_t

    spike_times_E = {}
    for neuron in np.arange(N_e):
        spike_times_neuron_E = spike_trains_E[neuron]/ms
        #only count spikes in certain temporal range
        spike_times_neuron_E = spike_times_neuron_E[np.where(spike_times_neuron_E>=t_min)[0]]
        spike_times_neuron_E = spike_times_neuron_E[np.where(spike_times_neuron_E<=t_max)[0]]
        #print(spike_times_neuron_E)
        number_of_spikes_E[neuron] = len(spike_times_neuron_E[np.where(spike_times_neuron_E)[0]])
        spike_times_E[neuron] = spike_times_neuron_E

        #print spike_trains_E[neuron]/ms, number_of_spikes_E[neuron]

    spike_times_I = {}
    for neuron in np.arange(N_i):
        spike_times_neuron_I = spike_trains_I[neuron]/ms
        spike_times_neuron_I = spike_times_neuron_I[np.where(spike_times_neuron_I>=t_min)[0]]
        spike_times_neuron_I = spike_times_neuron_I[np.where(spike_times_neuron_I<=t_max)[0]]
        number_of_spikes_I[neuron] = len(spike_times_neuron_I[np.where(spike_times_neuron_I)[0]])


    mean_spikes_E_nonsilent, mean_spikes_I_nonsilent =  np.around(np.mean(number_of_spikes_E[np.where(number_of_spikes_E != 0)[0]]),1), np.around(np.mean(number_of_spikes_I[np.where(number_of_spikes_I != 0)[0]]),1)

    silent_percentage_of_E_population =  len(np.where(number_of_spikes_E == 0)[0])/ np.float(N_e)
    silent_percentage_of_I_population =  len(np.where(number_of_spikes_I == 0)[0])/ np.float(N_i)

    ############
    #spectrum computations

    resolution_LFP = 0.2*ms
    population_rate_e = LFP_e.smooth_rate(window='gaussian', width=resolution_LFP)/Hz
    population_rate_i = LFP_i.smooth_rate(window='gaussian', width=resolution_LFP)/Hz



    t_min_rate = int(0/time_step_array)
    t_max_rate = int(101/time_step_array)


    time_rate_i = LFP_i.t/ms
    sampling_frequency_i = 1000./(time_rate_i[1] - time_rate_i[0]) #Hz
    factor_spec = 5.
    freq_i, spec_i = signal.welch(population_rate_i[t_min_rate:t_max_rate], sampling_frequency_i, nperseg = len(time_rate_i[t_min_rate:t_max_rate])/factor_spec, nfft = factor_spec*len(time_rate_i), window = 'hann', scaling = 'density')


    freq_max_i = freq_i[np.argmax(spec_i)]
    time_rate_e = LFP_e.t/ms
    sampling_frequency_e = 1000./(time_rate_e[1] - time_rate_e[0])
    freq_e, spec_e = signal.welch(population_rate_e[t_min_rate:t_max_rate], sampling_frequency_e, nperseg = len(time_rate_e[t_min_rate:t_max_rate])/factor_spec, nfft = factor_spec*len(time_rate_e), window = 'hann', scaling = 'density')
    freq_max_e = freq_e[np.argmax(spec_e)]

    return [freq_max_i, freq_max_e, mean_spikes_I_nonsilent, mean_spikes_E_nonsilent, 1.-silent_percentage_of_I_population, 1.-silent_percentage_of_E_population]


#model 3 (Figs. 7 and 8)
def spike_generation_3_current(params):
    #added 230920
    #clear_cache('cython')
    #cache_dir = os.path.expanduser(f'~/.cython/brian-pid-{os.getpid()}')
    #prefs.codegen.runtime.cython.cache_dir = cache_dir
    #prefs.codegen.runtime.cython.multiprocess_safe = False

    #mu and sigma of lognormal or normal distribution
    param1, param2 = params[0], params[1]

    start_scope()
    seed()
    N_e = 12000
    N_i = 200

    #interneurons
    E_rest_i = -65.*mV
    E_I_i = -75.*mV
    C_i = 100.0*pF #100
    gl_i = 10.0*nS

    #E population
    E_e = 0.*mV
    E_i = -68.*mV 
    E_rest = -67.*mV
    C = 275.0*pF
    gl = 25.*nS

    #synaptic parameters
    #E-synapses
    #"AMPA on pyramidal cells"
    tau_r_E_e = 0.5*ms
    tau_d_E_e = 1.8*ms
    g_peak_E_e = 0.9*nS

    #"GABA on CA1 pyramidal cells"
    tau_r_E_i = 0.4*ms 
    tau_d_E_i = 2.0*ms 
    g_peak_E_i = 9.0*nS 

    #I-synapses
    #"AMPA on interneurons"
    tau_r_I_e = 0.5*ms 
    tau_d_I_e = 1.2*ms 
    g_peak_I_e = 3.0*nS 

    #"GABA on interneurons"
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


    #CA3 to CA1 synapses
    #generic AMPA time course (see Gasparini papers), but large g_peak
    tau_r_poisson_SC = 1.0*ms
    tau_d_poisson_SC = 2.0*ms
    g_peak_external_SC = 0.25*nS

    invpeak_E_poisson_SC = (tau_r_poisson_SC/tau_d_poisson_SC) ** (tau_d_poisson_SC/(tau_r_poisson_SC-tau_d_poisson_SC))
    syn_eqs_poisson_E_SC = Equations('''dg_e_poisson_SC/dt = (invpeak_E_poisson_SC*s_e_poisson_SC - g_e_poisson_SC)/tau_d_poisson_SC : siemens
            ds_e_poisson_SC/dt = -s_e_poisson_SC/tau_r_poisson_SC : siemens''')

    #120320, equations for dendritic current arriving in soma
    tau_r_dendritic = 1.0*ms #1.0*ms (standard), 0.1*ms
    tau_d_dendritic = 4.0*ms #4.0*ms #2.0*ms #4.0*ms is standard

    invpeak_dendritic = (tau_r_dendritic/tau_d_dendritic) ** (tau_d_dendritic/(tau_r_dendritic-tau_d_dendritic))
    eqs_dendritic = Equations('''dI_dendritic/dt = (invpeak_dendritic*s_dendritic - I_dendritic)/tau_d_dendritic : ampere
            ds_dendritic/dt = -s_dendritic/tau_r_dendritic : ampere''')

    ####
    peak_time_zero_mean = 50.0
    np.random.seed()


    #this time step is also used for the setup of g_chr and the simulation time step
    time_step_array = 0.01 #0.01 ms is default
    defaultclock.dt = time_step_array*ms
    ##time from 0 to 100 ms in steps of time_step_array ms
    temporal_sigma = 10.0 
    normalization = 1./np.sqrt(2.*np.pi*temporal_sigma**(2))
    time_for_gaussian = np.arange(0.0, 100+ defaultclock.dt/ms, time_step_array)

    dendritic_spike_threshold =  4 #more than 4 spikes are required to generate dendritic spike
    event_delay = 5.0*ms #dendritic refractory period
    FF_delay = 1.0*ms
    rec_delay = 1.0*ms

    I_app_i = 0.0*nA

    sigma_noise = 1.0*mV

    eqs_e = (Equations('''dV/dt = (gl*(E_rest - V) + (g_e + g_e_poisson_SC)*(E_e - V) + g_i*(E_i - V) + I_app_e + I_dendritic)/C  + sigma_noise*sqrt(2/(C/gl))*xi: volt (unless refractory) \
            I_app_e: ampere
            spike_counter: integer
            last_event : second''') + syn_eqs_E_e + syn_eqs_E_i + syn_eqs_poisson_E_SC + eqs_dendritic)


    group_e = NeuronGroup(N_e,  eqs_e, threshold='V>=-50*mV',  events = {'dendritic_spike': '(spike_counter > dendritic_spike_threshold) and V > E_i and (t>last_event+event_delay)'},  reset='V=E_rest', refractory= 2.0*ms, method = 'euler') 


    eqs_i = (Equations('''dV/dt = (gl_i*(E_rest_i - V) + (g_e)*(E_e - V) + g_i*(E_I_i - V) + I_app_i + I_gap)/C_i + sigma_noise*sqrt(2/(C_i/gl_i))*xi: volt (unless refractory) \
            I_gap: ampere''') + syn_eqs_I_e + syn_eqs_I_i)

    group_i = NeuronGroup(N_i, eqs_i, threshold='V>=-52*mV', reset='V=E_rest_i', refractory= 1.0*ms, method = 'euler')

    group_e.run_on_event('dendritic_spike', 'last_event = t')

    group_e.last_event = -1e9*ms  #generate no dendritic spikes at the beginning of the simulation

    n_excited = N_e 
    
    delayed_trigger = Synapses(group_e[0:n_excited], group_e[0:n_excited], 'w_syn: ampere', on_pre= 's_dendritic += w_syn', on_event='dendritic_spike', delay= FF_delay + 1.0*ms)
    #synapses, e.g. from E to I
    syn_ee = Synapses(group_e, group_e, on_pre='s_e += g_peak_E_e', delay = 1.0*ms)
    syn_ii = Synapses(group_i, group_i, on_pre='s_i += g_peak_I_i', delay = 1.0*ms)

    syn_ei = Synapses(group_e, group_i, on_pre='s_e += g_peak_I_e', delay = rec_delay)
    syn_ie = Synapses(group_i, group_e, on_pre='s_i += g_peak_E_i', delay = rec_delay-0.5*ms)

    M_e = SpikeMonitor(group_e)
    M_i = SpikeMonitor(group_i)
    LFP_e = PopulationRateMonitor(group_e)
    LFP_i = PopulationRateMonitor(group_i)

    duration = 0.1*second


    dendritic_window = 2.0*ms + defaultclock.dt 
    N_eCA3 = 15000 


    peak_rate = 8.0
    array_gaussian = []
    for time_index in np.arange(0, len(time_for_gaussian)):
        random_activation = peak_rate*(1./normalization)*(gaussian_form(time_for_gaussian[time_index], temporal_sigma, peak_time_zero_mean))
        array_gaussian.append(random_activation)

    rates_CA3 = TimedArray(array_gaussian*Hz, dt = time_step_array*ms)
    group_eCA3 = PoissonGroup(N_eCA3, rates= 'rates_CA3(t)')

    seed()
    S = Synapses(group_eCA3, group_e[0:n_excited], on_pre={'up': 'spike_counter += 1',  'down': 'spike_counter -= 1', 'normal': 's_e_poisson_SC += 3.0*g_peak_external_SC'}, delay={'up': 0*ms, 'down': dendritic_window, 'normal': FF_delay})

    seed()
    delayed_trigger.connect(j='i')
    syn_ee.connect(p = 197./N_e)
    syn_ii.connect(p = 0.2)
    syn_ei.connect(p = 0.1)
    syn_ie.connect(p = 0.1) 

    p_connect = 130.0/np.float(N_eCA3) #130 CA3 inputs on average per CA1 cell
    S.connect(p=p_connect)

    #peak conductances for d-spike induced somatic currents
    np.random.seed()
    mu_lognormal = param1
    sigma_lognormal = param2 

    #lognormal
    W = np.random.lognormal(mu_lognormal, sigma_lognormal, size = n_excited)
    
    #restrict W to values smaller than 4nA and set larger values to 0.
    #W = np.where(W<4.0, W, 0.0) #restrict W to values smaller than 4 nA
    
    #re-sampling, 26.05.21
    #for index_W in range(n_excited):
    #    while W[index_W]>2.0:
    #        W[index_W] = np.random.lognormal(mu_lognormal, sigma_lognormal, size = 1)

    #Gaussian
    #W = np.random.normal(mu_lognormal, sigma_lognormal, size = n_excited) 
    #W = np.where(W<10, W, 0) #restrict W to values smaller than 10 nA
    #W[np.where(W<0)[0]] = 0 #restrict W to positive values in case of Gaussian
    
    
    #mean- and standard-deviation matched Gaussian, 19.04.21
    def mean_lognormal(mu, sigma):
        return np.exp(mu + sigma**(2)/2.)

    def std_lognormal(mu, sigma):
        return np.sqrt(((np.exp(sigma**2))-1)*(np.exp(2*mu + sigma**(2))))
    
    #mu_gaussian = mean_lognormal(mu_lognormal, sigma_lognormal)
    #sigma_gaussian = std_lognormal(mu_lognormal, sigma_lognormal)
    #W = np.random.normal(mu_gaussian, sigma_gaussian, size = n_excited) #size = N_e

    weights = W*nA
    delayed_trigger.w_syn = weights.flatten()

    #new 160919
    np.random.seed()

    group_e.V = np.random.normal(E_rest/mV, 0.1, N_e)*mV
    group_e.spike_counter = 0
    group_i.V = np.random.normal(E_rest_i/mV, 0.1, N_i)*mV
    group_e.I_app_e = 0.0*nA 

    run(duration)

    #spike count computations, added 170719
    spike_trains_E = M_e.spike_trains()
    spike_trains_I = M_i.spike_trains()
    number_of_spikes_E = np.zeros(N_e)
    number_of_spikes_I = np.zeros(N_i)
    delta_t = 50.0
    t_min = peak_time_zero_mean - delta_t
    t_max = peak_time_zero_mean + delta_t

    spike_times_E = {}
    for neuron in np.arange(N_e):
        spike_times_neuron_E = spike_trains_E[neuron]/ms
        #only count spikes in certain temporal range
        #spike_times_neuron_E = spike_times_neuron_E[np.where(spike_times_neuron_E>=t_min)[0]] 
        #spike_times_neuron_E = spike_times_neuron_E[np.where(spike_times_neuron_E<=t_max)[0]]
        #print(spike_times_neuron_E)
        #number_of_spikes_E[neuron] = len(spike_times_neuron_E[np.where(spike_times_neuron_E)[0]]) #usually used
        number_of_spikes_E[neuron] = len(spike_times_neuron_E)
        #spike_times_E[neuron] = spike_times_neuron_E

        #print spike_trains_E[neuron]/ms, number_of_spikes_E[neuron]

    spike_times_I = {}
    for neuron in np.arange(N_i):
        spike_times_neuron_I = spike_trains_I[neuron]/ms
        #spike_times_neuron_I = spike_times_neuron_I[np.where(spike_times_neuron_I>=t_min)[0]]
        #spike_times_neuron_I = spike_times_neuron_I[np.where(spike_times_neuron_I<=t_max)[0]]
        #number_of_spikes_I[neuron] = len(spike_times_neuron_I[np.where(spike_times_neuron_I)[0]]) #usually used
        number_of_spikes_I[neuron] = len(spike_times_neuron_I)

    #print number_of_spikes_E, number_of_spikes_I
    #mean_spikes_E, mean_spikes_I = np.around(mean(number_of_spikes_E),1), np.around(mean(number_of_spikes_I),1)

    mean_spikes_E_nonsilent, mean_spikes_I_nonsilent =  np.around(np.mean(number_of_spikes_E[np.where(number_of_spikes_E != 0)[0]]),1), np.around(np.mean(number_of_spikes_I[np.where(number_of_spikes_I != 0)[0]]),1)
    #no rounding
    #mean_spikes_E_nonsilent, mean_spikes_I_nonsilent =  np.mean(number_of_spikes_E[np.where(number_of_spikes_E != 0)[0]]), np.mean(number_of_spikes_I[np.where(number_of_spikes_I != 0)[0]])

    silent_percentage_of_E_population =  len(np.where(number_of_spikes_E == 0)[0])/ np.float(N_e)
    silent_percentage_of_I_population =  len(np.where(number_of_spikes_I == 0)[0])/ np.float(N_i)

    ############
    #spectrum computations

    resolution_LFP = 0.2*ms #1.0*ms
    population_rate_e = LFP_e.smooth_rate(window='gaussian', width=resolution_LFP)/Hz
    population_rate_i = LFP_i.smooth_rate(window='gaussian', width=resolution_LFP)/Hz

    #standard
    t_min_rate = int(0/time_step_array)
    t_max_rate = int(101/time_step_array)

    #t_min_rate = int(40/time_step_array)
    #t_max_rate = int(61/time_step_array)

    time_rate_i = LFP_i.t/ms
    sampling_frequency_i = 1000./(time_rate_i[1] - time_rate_i[0]) #Hz
    factor_spec = 5.
    freq_i, spec_i = signal.welch(population_rate_i[t_min_rate:t_max_rate], sampling_frequency_i, nperseg = len(time_rate_i[t_min_rate:t_max_rate])/factor_spec, nfft = 5.*len(time_rate_i), window = 'hann', scaling = 'density')
    freq_max_i = freq_i[np.argmax(spec_i)]
    
    
    time_rate_e = LFP_e.t/ms
    sampling_frequency_e = 1000./(time_rate_e[1] - time_rate_e[0])
    freq_e, spec_e = signal.welch(population_rate_e[t_min_rate:t_max_rate], sampling_frequency_e, nperseg = len(time_rate_e[t_min_rate:t_max_rate])/factor_spec, nfft = 5.*len(time_rate_e), window = 'hann', scaling = 'density')
    freq_max_e = freq_e[np.argmax(spec_e)]

    return [freq_max_i, freq_max_e, mean_spikes_I_nonsilent, mean_spikes_E_nonsilent, 1.-silent_percentage_of_I_population, 1.-silent_percentage_of_E_population]



#######################################
#model 1 with and without feedforward drive to I cells
#######################################
#number of excited E cells n_E
#parameter_values_1 = np.arange(10, 1500, 30)
#mean intensity of sharp wave
#parameter_values_2 = np.arange(10., 159., 3)

#######################################
#model 2
#######################################
#number of excited E cells n_E
#parameter_values_1 = np.arange(10, 3000, 60)
#mean intensity of sharp wave
#parameter_values_2 = np.arange(10., 159., 3)

#######################################
#model 3_current
#######################################
#mu and sigma of lognormal distribution
parameter_values_1 = np.arange(-1.0, 1.0, 0.04) #standard
parameter_values_2 = np.arange(0.01, 2.0, 0.04) #standard

#060520, 081120: mu and sigma of Gaussian
#parameter_values_1 = np.arange(0.0, 5.0, 0.1)
#parameter_values_2 = np.arange(0.08, 5.0, 0.1)
#######################################


frequency_I = np.zeros((len(parameter_values_1), len(parameter_values_2)))
frequency_E = np.zeros((len(parameter_values_1), len(parameter_values_2)))
std_frequency_I = np.zeros((len(parameter_values_1), len(parameter_values_2)))
std_frequency_E = np.zeros((len(parameter_values_1), len(parameter_values_2)))

#mean spike counts of non-silent populations
mean_count_I = np.zeros((len(parameter_values_1), len(parameter_values_2)))
mean_count_E = np.zeros((len(parameter_values_1), len(parameter_values_2)))

#fraction of non-silent population in ripple window
non_silent_I = np.zeros((len(parameter_values_1), len(parameter_values_2)))
non_silent_E = np.zeros((len(parameter_values_1), len(parameter_values_2)))

#start n_MC processes on n_threads
n_threads = 30 #30 usually
for h, param1_in in  enumerate(parameter_values_1):
    for j, param2_in in enumerate(parameter_values_2):
        print('=====================')
        print(h,j)

        #prepare parameter for pool function below, this controls how many simulations each thread will run
        params_pool = []
        for k in range(n_threads): #3*n_threads simulations in total
            params_pool.append([param1_in, param2_in])

        if __name__ == '__main__':
            result_list = []
            pool = multiprocessing.Pool(processes=n_threads)
            #result_list= pool.map(spike_generation_1, params_pool) #run scan for model 1
            #result_list= pool.map(spike_generation_1_FFDrive_I, params_poo) #run scan for model 1 with feedforward drive to basket cells
            #result_list= pool.map(spike_generation_2, params_pool) #run scan for model 2
            result_list= pool.map(spike_generation_3_current, params_pool) #run scan for model 3
            pool.close()


        result_list = np.asarray(result_list)

        frequency_I[h,j] = np.mean(result_list[:,0])
        frequency_E[h,j] = np.mean(result_list[:,1])
        std_frequency_I[h,j] = np.std(result_list[:,0])
        std_frequency_E[h,j] = np.std(result_list[:,1])

        #spike counts, added 170719
        mean_count_I[h,j] = np.mean(result_list[:,2])
        mean_count_E[h,j] = np.mean(result_list[:,3])
        #added 120819
        non_silent_I[h,j] = np.mean(result_list[:,4])
        non_silent_E[h,j] = np.mean(result_list[:,5])


        print('parameter 1, parameter 2/ mean frequencies', np.around(param1_in,4), np.around(param2_in,4), frequency_I[h,j], frequency_E[h,j])
        print('standard deviation of frequencies', std_frequency_I[h,j], std_frequency_E[h,j])

#----------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------
file_name = 'model3_standard.dat'

print('Now saving data...')
np.savetxt('fI_'+ file_name, frequency_I, delimiter=',',  fmt = '%10.3f')
np.savetxt('fE_' + file_name, frequency_E, delimiter=',',  fmt = '%10.3f')
np.savetxt('std_fI_' + file_name, std_frequency_I, delimiter=',',  fmt = '%10.3f')
np.savetxt('std_fE_' + file_name, std_frequency_E, delimiter=',',  fmt = '%10.3f')
np.savetxt('mean_count_I_' + file_name, mean_count_I, delimiter=',',  fmt = '%10.3f')
np.savetxt('mean_count_E_' + file_name, mean_count_E, delimiter=',',  fmt = '%10.3f')
np.savetxt('nonsilent_I_' + file_name, non_silent_I, delimiter=',',  fmt = '%10.3f')
np.savetxt('nonsilent_E_' + file_name, non_silent_E, delimiter=',',  fmt = '%10.3f')
