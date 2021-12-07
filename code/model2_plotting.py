#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 28 10:33:00 2019

@author: wbraun
"""

###
#from ripple_generation_new_deltat.ipynb
#Generates Fig.5


from brian2 import *
from brian2 import seed
import numpy as np
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

printing = True
single_runs = True

#sharp wave profiles
def gaussian_form(t, sigma, mu):
    return (1./(np.sqrt(2.*np.pi*sigma**(2))))*np.exp(-(t-mu)**(2)/(2.0*sigma**(2)))

def rectangular_form(t, sigma, mu):
    if np.abs(t-mu) <= sigma:
        x = 1.
    else:
        x = 0

    return x

def spike_generation(plot_indicator, sharp_wave_amplitude, n_excited):
    print('setting up network')
    spike_computation = True

    start_scope()
    #np.random.seed()
    seed(2)
    N_e = 12000 #10000  #9416
    N_i = 200 # 1200 #1000 #471

    plotting = plot_indicator
    spike_computation = True

    #interneurons
    # I_app = 0.6*nA
    E_rest_i = -65.*mV
    E_I_i = -75.*mV
    C_i = 100.0*pF
    gl_i = 10.0*nS

    #E population
    E_e = 0.*mV
    E_i = -68.*mV #inhibitory reversal potential
    E_rest = -67.*mV #resting membrane potential
    C = 275.0*pF 
    gl = 25.*nS

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
    g_peak_I_e = 3.0*nS 

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
    peak_time_zero_mean = 50.0
    #temporal variation around
    peak_time_zero =  np.random.normal(peak_time_zero_mean, 10.0, N_e) #50.0
    mean_intensity = sharp_wave_amplitude 

    #this time step is also used for the setup of g_chr and the simulation time step
    time_step_array = 0.01 #ms
    defaultclock.dt = time_step_array*ms
    ##time from 0 to 100 ms in steps of time_step_array ms
    time_for_gaussian = np.arange(0.0, 100 + defaultclock.dt/ms, time_step_array)

    #print time_for_gaussian

    #value at peak time.
    temporal_sigma = 3.0 
    normalization = 1./np.sqrt(2.*np.pi*temporal_sigma**(2))


    #choose a subset of excitatory cells randomly
    E_cells = np.arange(N_e)
    #np.random.seed()
    shuffled_E_cells = np.random.permutation(E_cells)
    shuffled_E_cells = shuffled_E_cells[0:n_excited]


    active_neurons = np.zeros(N_e)
    active_neurons[shuffled_E_cells] = 1.0

    # pl.plot(peak_time_zero[np.where(active_neurons>0)[0]], shuffled_E_cells, 'x')
    # pl.show()

    array_gaussian = []
    #drive all pyramidal cells in the same way.
    amplitude = mean_intensity*np.ones(N_e)

    amplitude *= active_neurons


    for j in np.arange(0, len(time_for_gaussian)):
        random_activation = np.zeros(len(amplitude))
        random_activation = (amplitude)*(1./normalization)*(gaussian_form(time_for_gaussian[j], temporal_sigma, peak_time_zero)) #+ np.random.normal(0.0, 1.0)
        array_gaussian.append(random_activation)


    g_chr = TimedArray(array_gaussian, dt = time_step_array*ms)

    I_app_i = 0.00*nA

    sigma_noise = 1.0*mV

    eqs_e = (Equations('dV/dt = (gl*(E_rest - V) + (g_e + g_chr(t,i)*nS)*(E_e - V) + g_i*(E_i - V) +I_app_e)/C  + sigma_noise*sqrt(2/(C/gl))*xi : volt (unless refractory) \
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

    #numpy.random.seed()
    #seed()
    syn_ee.connect(p = 197./N_e)
    syn_ii.connect(p = 0.2)
    syn_ei.connect(p = 0.1) 
    syn_ie.connect(p = 0.1)

    #np.random.seed()
    #seed()
    group_e.V = np.random.normal(E_rest/mV, 0.1, N_e)*mV
    group_i.V = np.random.normal(E_rest_i/mV, 0.1, N_i)*mV
    group_e.I_app_e = 0.0*nA

    if spike_computation:
        M_e = SpikeMonitor(group_e)
        M_i = SpikeMonitor(group_i)

    LFP_e = PopulationRateMonitor(group_e)
    LFP_i = PopulationRateMonitor(group_i)

    #recording_set = np.random.randint(0, N_e, size = 20)
    #recording_set_i = np.random.randint(0, N_i, size = 10)
    #voltage_e = StateMonitor(group_e, 'V', record= recording_set)
    #voltage_i = StateMonitor(group_i, 'V', record= recording_set_i)

    duration = 0.1*second

    run(duration)

    #####################
    #plotting results
    #####################

    resolution_LFP = 0.5*ms #0.2 is standard value
    population_rate_e = LFP_e.smooth_rate(window='gaussian', width=resolution_LFP)/Hz #'flat'
    population_rate_i = LFP_i.smooth_rate(window='gaussian', width=resolution_LFP)/Hz

    t_min_rate = int(0/time_step_array)
    t_max_rate = int(101/time_step_array)
    
    time_rate_i = LFP_i.t/ms
    sampling_frequency_i = 1000./(time_rate_i[1] - time_rate_i[0]) #Hz
    freq_i, spec_i = signal.welch(population_rate_i[t_min_rate:t_max_rate], sampling_frequency_i, nperseg = len(time_rate_i)/5, nfft = 5*len(time_rate_i), window = 'hann', scaling = 'density')
    freq_max_i = freq_i[np.argmax(spec_i)]

    time_rate_e = LFP_e.t/ms
    sampling_frequency_e = 1000./(time_rate_e[1] - time_rate_e[0])
    freq_e, spec_e = signal.welch(population_rate_e[t_min_rate:t_max_rate], sampling_frequency_e, nperseg = len(time_rate_e)/5, nfft = 5*len(time_rate_e), window = 'hann', scaling = 'density')
    freq_max_e = freq_e[np.argmax(spec_e)]


    if spike_computation:
        spike_trains_E = M_e.spike_trains()
        spike_trains_I = M_i.spike_trains()
        number_of_spikes_E = np.zeros(N_e)
        number_of_spikes_I = np.zeros(N_i)
        delta_t = 50.0
        t_min = peak_time_zero_mean - delta_t
        t_max = peak_time_zero_mean + delta_t

        #spike_times_E = {}
        for neuron in np.arange(N_e):
            spike_times_neuron_E = spike_trains_E[neuron]/ms
            #only count spikes in certain temporal range
            #spike_times_neuron_E = spike_times_neuron_E[np.where(spike_times_neuron_E>=t_min)[0]]
            #spike_times_neuron_E = spike_times_neuron_E[np.where(spike_times_neuron_E<=t_max)[0]]
            #print('---')
            #print(spike_times_neuron_E)
            number_of_spikes_E[neuron] = len(spike_times_neuron_E)
            #print(number_of_spikes_E[neuron])
            #spike_times_E[neuron] = spike_times_neuron_E

            #print spike_trains_E[neuron]/ms, number_of_spikes_E[neuron]

        #spike_times_I = {}
        for neuron in np.arange(N_i):
            spike_times_neuron_I = spike_trains_I[neuron]/ms
            #spike_times_neuron_I = spike_times_neuron_I[np.where(spike_times_neuron_I>=t_min)[0]]
            #spike_times_neuron_I = spike_times_neuron_I[np.where(spike_times_neuron_I<=t_max)[0]]
            #spike_times_I[neuron] = spike_times_neuron_I
            #print neuron, len(spike_times_neuron_I)
            #print spike_times_I[neuron]

            number_of_spikes_I[neuron] = len(spike_times_neuron_I)
            #print spike_trains_I,  number_of_spikes_I[neuron]

        #print number_of_spikes_E, number_of_spikes_I
        mean_spikes_E, mean_spikes_I = np.around(mean(number_of_spikes_E),1), np.around(mean(number_of_spikes_I),1)

        mean_spikes_E_nonsilent, mean_spikes_I_nonsilent =  np.around(np.mean(number_of_spikes_E[np.where(number_of_spikes_E != 0)[0]]),1), np.around(np.mean(number_of_spikes_I[np.where(number_of_spikes_I != 0)[0]]),1)
        total_spikes_E, total_spikes_I =  np.around(np.sum(number_of_spikes_E[np.where(number_of_spikes_E != 0)[0]]),1) ,np.around(np.sum(number_of_spikes_I[np.where(number_of_spikes_I != 0)[0]]),1)

        silent_percentage_of_E_population =  len(np.where(number_of_spikes_E == 0)[0])/ np.float(N_e)
        silent_percentage_of_I_population =  len(np.where(number_of_spikes_I == 0)[0])/ np.float(N_i)


    if printing:
        print ("mean number of spikes per neuron for non-silent E, I population:", mean_spikes_E_nonsilent, mean_spikes_I_nonsilent)
        print ("number of silent E neurons:", len(np.where(number_of_spikes_E == 0)[0]), "percentage", silent_percentage_of_E_population)
        print ("number of silent I neurons:", len(np.where(number_of_spikes_I == 0)[0]), "percentage", silent_percentage_of_I_population)
        print ("number of firing E neurons:", len(np.where(number_of_spikes_E != 0)[0]), 'excited E neurons:', len(shuffled_E_cells))
        print ("mean number of spikes per firing E neuron:", mean_spikes_E_nonsilent)
        print ("mean number of spikes per firing I neuron:", mean_spikes_I_nonsilent)
        print ("total E spikes:", total_spikes_E)
        print ("total I spikes",  total_spikes_I)

    if plotting:

        fig = pl.figure(0, figsize = (7.5, 5.0), dpi = 400)
        ax0 = fig.add_subplot(411)
        ax0.annotate('A', xy=(-0.12, 1.0), xycoords='axes fraction', fontsize= 24.0)

        array_gaussian_plotting  = np.asarray(array_gaussian)
      
        for index_cell, j in enumerate(shuffled_E_cells):
            ax0.plot(time_for_gaussian, array_gaussian_plotting[:, j], '-', linewidth = 0.5, alpha = 0.8, color = 'k')
    
        pl.axvline(x = 50, color = 'r', linestyle = '--', linewidth = 3.0)
        pl.axhline(y = sharp_wave_amplitude, color = 'r', linestyle = '--', linewidth = 1.0)
        pl.ylabel(r'$g_{\text{ext}}~[\text{nS}]$', fontsize = 12.0)

        pl.tick_params(labelsize=12.0)
        ax0.xaxis.set_major_formatter(fmt)
        ax0.yaxis.set_major_formatter(fmt)
        ax0.set_xticks(np.arange(0, duration/ms, step=10))
        pl.xlim(0, duration/ms)
        ax0.xaxis.set_ticklabels([])

        ax1 = fig.add_subplot(412)
        ax1.plot(M_e.t/ms, M_e.i, 'r.', markersize = 2.0)
       
        pl.axvline(x = 50, color = 'k', linestyle = '--', linewidth = 1.0)
        pl.ylabel('E cell', fontsize = 12.0)

        pl.title("avg. number of spikes for E/ I: " + str(mean_spikes_E_nonsilent) + '/ ' +  str(mean_spikes_I_nonsilent) + ' ' + ", fraction active E (I) neurons: " + str(np.around(1.-silent_percentage_of_E_population,4)) + " (" + str(np.around(1.-silent_percentage_of_I_population,3)) + ")" , fontsize = 12.0)
        pl.tick_params(labelsize = 12.0)
        ax1.xaxis.set_major_formatter(fmt)
        ax1.yaxis.set_major_formatter(fmt)
        pl.xlim(0, duration/ms)
        ax1.set_xticks(np.arange(0, duration/ms, step=10))
        ax1.xaxis.set_ticklabels([])

        ax2 = fig.add_subplot(413)
        ax2.plot(M_i.t/ms, M_i.i, 'b.', markersize = 2.0)
        #pl.xlim(0, duration/ms)
        pl.axvline(x = 50, color = 'k', linestyle = '--', linewidth = 1.0)
        pl.ylabel('I cell', fontsize = 12.0)
        pl.tick_params(labelsize = 12.0)
        ax2.xaxis.set_major_formatter(fmt)
        ax2.yaxis.set_major_formatter(fmt)
        pl.xlim(0, duration/ms)
        ax2.set_xticks(np.arange(0, duration/ms, step=10))
        ax2.xaxis.set_ticklabels([])

       
        ax3 = fig.add_subplot(414)
        rate_e = (N_e*population_rate_e)/1000.
        rate_i = (N_i*population_rate_i)/1000.
        ax3.plot(LFP_e.t/ms, rate_e, '-r')
        ax3.plot(LFP_i.t/ms, rate_i, '-b')
        #pl.xlim(0, duration/ms)
        pl.axvline(x = 50, color = 'k', linestyle = '--', linewidth = 1.0)
        pl.ylabel(r'$A_{E},A_{I}~[\text{kHz}]$', fontsize = 12.0)
        ax3.annotate(r'$f_{\text{E}}=$'+ str(np.around(freq_max_e,5)) + " Hz", xy=(2, 0.62*np.max(rate_i)), color = 'r',  fontsize = 12.0)
        ax3.annotate(r'$f_{\text{I}}=$'+ str(np.around(freq_max_i,5)) + " Hz", xy=(2, 0.22*np.max(rate_i)), color = 'b', fontsize = 12.0)
        pl.xlabel(r'$t~(\text{ms})$', fontsize = 12.0)

        pl.xlim(0, 100)
        ax3.set_xticks(np.arange(0, duration/ms, step=10))
        ax3.xaxis.set_ticklabels(np.arange(0, duration/ms, step=10))
        pl.tick_params(labelsize = 12.0)
        ax3.xaxis.set_major_formatter(fmt)
        ax3.yaxis.set_major_formatter(fmt)
        #pl.show()
        pl.tight_layout(pad = 1.0)
        fig.savefig("/home/wilhelm/Schreibtisch/Fig5A.png", bbox_inches='tight', dpi= 400)
        pl.show()

        #exit()

        fig = pl.figure(1, figsize = (7.5, 1.5), dpi = 300)
        #plots for how many neurons are spiking
        ax6 = fig.add_subplot(121)
        ax6.annotate('B', xy=(-0.3, 1.1), xycoords='axes fraction', fontsize= 24.0)

        bins_E = np.arange(0, max(number_of_spikes_E) + 2)
        pl.hist(number_of_spikes_E, bins= bins_E, histtype = 'step', color = 'r', linewidth = 5.0, align = 'left', log = True)
        pl.axvline(x =mean_spikes_E_nonsilent, color = 'r', linewidth = 5.0, linestyle = '--')
        pl.ylabel('no. of E cells', fontsize = 12.0)
        pl.xlabel('spikes', fontsize = 12.0)
        pl.xticks(np.arange(0, np.max(bins_E), step=1)) #step = 2
        pl.tick_params(labelsize = 12.0)
        ax6.xaxis.get_major_formatter()._usetex = False
        ax6.yaxis.get_major_formatter()._usetex = False

        ax7 = fig.add_subplot(122)
        bins_I = np.arange(0, max(number_of_spikes_I) + 2)
        pl.hist(number_of_spikes_I, bins= bins_I, histtype = 'step', color = 'b', linewidth = 5.0, align = 'left', log = True)
        pl.axvline(x = mean_spikes_I_nonsilent, color = 'b', linewidth = 5.0, linestyle = '--')

        pl.ylabel('no. of I cells', fontsize = 12.0)
        pl.xlabel('spikes', fontsize = 12.0)
        #pl.xticks(bins_I)
        pl.xticks(np.arange(0, np.max(bins_I), step=2)) #step =4
        pl.tick_params(labelsize = 12.0)
        ax7.xaxis.get_major_formatter()._usetex = False
        ax7.yaxis.get_major_formatter()._usetex = False
        pl.tight_layout(pad = 1.0)
        fig.savefig("/home/wilhelm/Schreibtisch/Fig5B.png", bbox_inches='tight', dpi = 400)
        pl.show()
        
        
    if printing:
        print('generated ripple and spike times')
        #return spike_times_E, spike_times_I
        print('Global oscillation frequencies (I/E):', freq_max_i,",",  freq_max_e)
        print('Mean spikes around ripple for active subpopulations (I/E)', mean_spikes_I_nonsilent,",",  mean_spikes_E_nonsilent)
    return [freq_max_i, freq_max_e]

if single_runs:
    n_reps = 1
    for k in np.arange(n_reps):
        result = spike_generation(plot_indicator = True, sharp_wave_amplitude = 83.0, n_excited = 1580)
