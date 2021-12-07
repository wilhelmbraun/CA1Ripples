# #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 5 2019

@author: wbraun
"""

###from ripple_generation_apical_parameters.ipynb
#Generates Fig.7

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

#sharp wave profiles
def gaussian_form(t, sigma, mu):
    return (1./(np.sqrt(2.*np.pi*sigma**(2))))*np.exp(-(t-mu)**(2)/(2.0*sigma**(2)))

def rectangular_form(t, sigma, mu):
    if np.abs(t-mu) <= sigma:
        x = 1.
    else:
        x = 0

    return x

def spike_generation(plot_indicator, N_eCA3, dendritic_spike_threshold, event_delay, dendritic_window , peak_rate, p_IE, mu_LN, sigma_LN):
    plotting = plot_indicator
    spike_computation = True
    printing = True
    time_step_array = 0.01 #ms
    defaultclock.dt = time_step_array*ms
    print('===============================')
    print('parameters:', 'number of CA3 neurons', N_eCA3, 'dendr. spike threshold', dendritic_spike_threshold, 'delay betw. dendr. spikes', event_delay, 'dendr. integ. window', dendritic_window, 'rate', peak_rate, 'p_IE', p_IE)
    print('===============================')

    start_scope()
    #np.random.seed()
    seed(0)
    N_e = 12000
    N_eCA3 = N_eCA3
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

  

    #CA3 to CA1 synapses, parameters from Sayer et al. 1990, J Neurosci- need to adapt otherwise too many E spikes
    tau_r_poisson_SC = 1.0*ms # 4.0*ms
    tau_d_poisson_SC = 2.0*ms # 23.0*ms
    g_peak_external_SC =  0.25*nS

    invpeak_E_poisson_SC = (tau_r_poisson_SC/tau_d_poisson_SC) ** (tau_d_poisson_SC/(tau_r_poisson_SC-tau_d_poisson_SC))
    syn_eqs_poisson_E_SC = Equations('''dg_e_poisson_SC/dt = (invpeak_E_poisson_SC*s_e_poisson_SC - g_e_poisson_SC)/tau_d_poisson_SC : siemens
            ds_e_poisson_SC/dt = -s_e_poisson_SC/tau_r_poisson_SC : siemens''')
    
    #120320, equations for dendritic current arriving in soma
    tau_r_dendritic = 1.0*ms
    tau_d_dendritic = 4.0*ms #4.0*ms is standard

    invpeak_dendritic = (tau_r_dendritic/tau_d_dendritic) ** (tau_d_dendritic/(tau_r_dendritic-tau_d_dendritic))
    eqs_dendritic = Equations('''dI_dendritic/dt = (invpeak_dendritic*s_dendritic - I_dendritic)/tau_d_dendritic : ampere
            ds_dendritic/dt = -s_dendritic/tau_r_dendritic : ampere''')


    ####
    peak_time_zero_mean = 50.0
    dendritic_spike_threshold = dendritic_spike_threshold #5
    event_delay = event_delay #4.2*ms
    dendritic_window = dendritic_window + defaultclock.dt 
    
    I_app_i = 0.0*nA

    sigma_noise = 1.0*mV

    
    eqs_e = (Equations('''dV/dt = (gl*(E_rest - V) + (g_e + g_e_poisson_SC)*(E_e - V) + g_i*(E_i - V) +I_app_e + I_dendritic)/C + sigma_noise*sqrt(2/(C/gl))*xi : volt (unless refractory) \
       I_app_e: ampere
       spike_counter : integer
       last_event : second''') + syn_eqs_E_e + syn_eqs_E_i + syn_eqs_poisson_E_SC + eqs_dendritic)

    #with dendritic variable
    group_e = NeuronGroup(N_e,  eqs_e, threshold='V>=-50*mV', events = {'dendritic_spike': '(spike_counter > dendritic_spike_threshold) and V > E_i and (t>last_event+event_delay)'},  reset='V=E_rest', refractory= 2.0*ms, method = 'euler')

    temporal_sigma = 10.0 
    normalization = 1./np.sqrt(2.*np.pi*temporal_sigma**(2))
    
    time_for_gaussian = np.arange(0.0, 100+ defaultclock.dt/ms,time_step_array)
    peak_rate = peak_rate
    array_gaussian = []
    for time_index in np.arange(0, len(time_for_gaussian)):
        random_activation = peak_rate*(1./normalization)*(gaussian_form(time_for_gaussian[time_index], temporal_sigma, peak_time_zero_mean))
        array_gaussian.append(random_activation)
        

    rates_CA3 = TimedArray(array_gaussian*Hz, dt = time_step_array*ms)

    group_eCA3 = PoissonGroup(N_eCA3, rates= 'rates_CA3(t)')

    eqs_i = (Equations('dV/dt = (gl_i*(E_rest_i - V) + (g_e)*(E_e - V) + g_i*(E_I_i - V) + I_app_i)/C_i + sigma_noise*sqrt(2/(C_i/gl_i))*xi: volt (unless refractory)') + syn_eqs_I_e + syn_eqs_I_i)

    group_i = NeuronGroup(N_i, eqs_i, threshold='V>=-52*mV', reset='V=E_rest_i', refractory= 1.0*ms, method = 'euler') 

    #set up dendritic events
    group_e.run_on_event('dendritic_spike', 'last_event = t')
    group_e.last_event = -1e9*ms  # initial value = "never"
    
    n_excited = N_e
    FF_delay = 1.0*ms
    #synapses for dendritic spike_counter variables, coincidence-based
    S = Synapses(group_eCA3, group_e[0:n_excited], on_pre={'up': 'spike_counter += 1',  'down': 'spike_counter -= 1', 'normal': 's_e_poisson_SC += 3.0*g_peak_external_SC'}, delay={'up': 0*ms, 'down': dendritic_window, 'normal': FF_delay})
    #apply kicks with a delay
    delayed_trigger = Synapses(group_e[0:n_excited], group_e[0:n_excited], 'w_syn: ampere', on_pre= 's_dendritic += w_syn', on_event='dendritic_spike', delay=FF_delay + 1.0*ms)
    delayed_trigger.connect(j='i')
    

    
    #parameters of lognormal distribution
    mu = mu_LN
    sigma = sigma_LN

    W = np.random.lognormal(mu, sigma, size = n_excited)
    weights = W*nA

    
    #synapses, e.g. from E to I
    #kick sizes are g_peak
    syn_ee = Synapses(group_e, group_e, on_pre='s_e += g_peak_E_e', delay = 1.0*ms)
    syn_ii = Synapses(group_i, group_i, on_pre='s_i += g_peak_I_i', delay = 1.0*ms)

    syn_ei = Synapses(group_e, group_i, on_pre='s_e += g_peak_I_e', delay = 1.0*ms)
    syn_ie = Synapses(group_i, group_e, on_pre='s_i += g_peak_E_i', delay = 0.5*ms)

    #seed()

    syn_ee.connect(p = 197./N_e)
    syn_ii.connect(p = 0.2)
    syn_ei.connect(p = 0.1)
    syn_ie.connect(p = p_IE)
    p_connect = 130.0/np.float(N_eCA3)
    S.connect(p=p_connect)
    delayed_trigger.w_syn = weights.flatten()

    #np.random.seed()
    #seed()
    group_e.V = np.random.normal(E_rest/mV, 0.1, N_e)*mV
    group_e.spike_counter = 0 
    group_i.V = np.random.normal(E_rest_i/mV, 0.1, N_i)*mV
    group_e.I_app_e = 0.0*nA 

    recording_set = np.random.randint(0, N_e, size = 5)
    recording_set_i = np.random.randint(0, N_i, size = 5)
    
    if spike_computation:
        M_e = SpikeMonitor(group_e)
        M_eCA3 = SpikeMonitor(group_eCA3)
        M_i = SpikeMonitor(group_i)

    LFP_e = PopulationRateMonitor(group_e)
    LFP_i = PopulationRateMonitor(group_i)

    LFP_eCA3 = PopulationRateMonitor(group_eCA3)

   
    voltage_e = StateMonitor(group_e, 'V', record= recording_set)
    voltage_i = StateMonitor(group_i, 'V', record= recording_set_i)

    fast_input_e = StateMonitor(group_e, 'spike_counter', record = recording_set)

    duration = 0.1*second

    #seed()

    run(duration)

    #####################
    #plotting results
    #####################

    resolution_LFP = 0.5*ms #0.2 is standard value
    population_rate_e = LFP_e.smooth_rate(window='gaussian', width=resolution_LFP)/Hz #'flat'
    population_rate_i = LFP_i.smooth_rate(window='gaussian', width=resolution_LFP)/Hz
    population_rate_eCA3 = LFP_eCA3.smooth_rate(window='flat', width=1.0*ms)/Hz

    t_min_rate = int(0/time_step_array) #40, 61
    t_max_rate = int(101/time_step_array)

    time_rate_i = LFP_i.t/ms
    sampling_frequency_i = 1000./(time_rate_i[1] - time_rate_i[0]) #Hz
    freq_i, spec_i = signal.welch(population_rate_i[t_min_rate:t_max_rate], sampling_frequency_i, nperseg = len(time_rate_i[t_min_rate:t_max_rate])/5., nfft = 5*len(time_rate_i[t_min_rate:t_max_rate]), window = 'hann', scaling = 'density')
    freq_max_i = freq_i[np.argmax(spec_i)]

    time_rate_e = LFP_e.t/ms
    sampling_frequency_e = 1000./(time_rate_e[1] - time_rate_e[0])
    freq_e, spec_e = signal.welch(population_rate_e[t_min_rate:t_max_rate], sampling_frequency_e, nperseg = len(time_rate_e[t_min_rate:t_max_rate])/5., nfft = 5*len(time_rate_e[t_min_rate:t_max_rate]), window = 'hann', scaling = 'density')
    freq_max_e = freq_e[np.argmax(spec_e)]

    if spike_computation:
        spike_trains_E = M_e.spike_trains()
        spike_trains_I = M_i.spike_trains()
        number_of_spikes_E = np.zeros(N_e)
        number_of_spikes_I = np.zeros(N_i)
        delta_t = 50.0 #20
        t_min = peak_time_zero_mean - delta_t
        t_max = peak_time_zero_mean + delta_t

        spike_times_E = {}
        for neuron in np.arange(N_e):
            spike_times_neuron_E = spike_trains_E[neuron]/ms
            #only count spikes in certain temporal range
            spike_times_neuron_E = spike_times_neuron_E[np.where(spike_times_neuron_E>=t_min)[0]]
            spike_times_neuron_E = spike_times_neuron_E[np.where(spike_times_neuron_E<=t_max)[0]]
            #print(neuron, '---')
            #print(spike_times_neuron_E)
            number_of_spikes_E[neuron] = len(spike_times_neuron_E[np.where(spike_times_neuron_E)[0]])
            #print(number_of_spikes_E[neuron])
            #spike_times_E[neuron] = spike_times_neuron_E
            #print spike_trains_E[neuron]/ms, number_of_spikes_E[neuron]

        spike_times_I = {}
        for neuron in np.arange(N_i):
            spike_times_neuron_I = spike_trains_I[neuron]/ms
            spike_times_neuron_I = spike_times_neuron_I[np.where(spike_times_neuron_I>=t_min)[0]]
            spike_times_neuron_I = spike_times_neuron_I[np.where(spike_times_neuron_I<=t_max)[0]]
            spike_times_I[neuron] = spike_times_neuron_I
            #print neuron, len(spike_times_neuron_I)
            #print spike_times_I[neuron]

            number_of_spikes_I[neuron] = len(spike_times_neuron_I[np.where(spike_times_neuron_I)[0]])
            #print spike_trains_I,  number_of_spikes_I[neuron]

        #print number_of_spikes_E, number_of_spikes_I
        #mean_spikes_E, mean_spikes_I = np.around(mean(number_of_spikes_E),1), np.around(mean(number_of_spikes_I),1)

        mean_spikes_E_nonsilent, mean_spikes_I_nonsilent =  np.around(np.mean(number_of_spikes_E[np.where(number_of_spikes_E != 0)[0]]),3), np.around(np.mean(number_of_spikes_I[np.where(number_of_spikes_I != 0)[0]]),3)
        total_spikes_E, total_spikes_I =  np.around(np.sum(number_of_spikes_E[np.where(number_of_spikes_E != 0)[0]]),1) ,np.around(np.sum(number_of_spikes_I[np.where(number_of_spikes_I != 0)[0]]),1)

        silent_percentage_of_E_population =  len(np.where(number_of_spikes_E == 0)[0])/ np.float(N_e)
        silent_percentage_of_I_population =  len(np.where(number_of_spikes_I == 0)[0])/ np.float(N_i)


    if printing:
        print ("mean number of spikes per neuron for non-silent E, I population:", mean_spikes_E_nonsilent, mean_spikes_I_nonsilent)
        print ("number of silent E neurons:", len(np.where(number_of_spikes_E == 0)[0]), "percentage", silent_percentage_of_E_population)
        print ("number of silent I neurons:", len(np.where(number_of_spikes_I == 0)[0]), "percentage", silent_percentage_of_I_population)
        print ("number of firing E neurons:", len(np.where(number_of_spikes_E != 0)[0]))
        print ("mean number of spikes per firing E neuron:", mean_spikes_E_nonsilent)
        print ("mean number of spikes per firing I neuron:", mean_spikes_I_nonsilent)
        print ("total E spikes:", total_spikes_E)
        print ("total I spikes",  total_spikes_I)

    if plotting:
        
        
        fig = pl.figure(0, figsize = (7.5, 5.0), dpi = 400)
        ax0 = fig.add_subplot(411)
        ax0.annotate('A', xy=(-0.12, 1.05), xycoords='axes fraction', fontsize= 24.0) #36.0
        ax0.plot(M_eCA3.t/ms, M_eCA3.i, 'k.', markersize = 2.0)
        pl.ylabel('CA3 E cell', fontsize = 12.0)       
        pl.axvline(x = 50, color = 'r', linestyle = '--', linewidth = 3.0)
        pl.tick_params(labelsize = 12.0)
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
        rate_eCA3 = (N_eCA3*population_rate_eCA3)/1000.
        ax3.plot(LFP_e.t/ms, rate_e, '-r')
        ax3.plot(LFP_i.t/ms, rate_i, '-b')
        ax3.plot(LFP_eCA3.t/ms, rate_eCA3, '-k')
        #rate_sum_CA3 = peak_rate*N_eCA3
        #pl.axhline(y = rate_sum_CA3/10000., color = 'k', linestyle = '--', linewidth = 1.0)
        pl.axvline(x = 50, color = 'k', linestyle = '--', linewidth = 1.0)
        pl.ylabel(r'$A_{E},A_{I}~[\text{kHz}]$', fontsize = 12.0)
        ax3.annotate(r'$f_{E}=$'+ str(np.around(freq_max_e,1)) + " Hz", xy=(10, 0.55*np.max(rate_i)), color = 'r', fontsize = 12.0) 
        ax3.annotate(r'$f_{I}=$'+ str(np.around(freq_max_i,1)) + " Hz", xy=(10, 0.2*np.max(rate_i)), color = 'b', fontsize = 12.0)
        pl.xlabel(r'$t~(\text{ms})$', fontsize = 12.0)

        pl.xlim(0, 100)
        ax3.set_xticks(np.arange(0, duration/ms, step=10))
        ax3.xaxis.set_ticklabels(np.arange(0, duration/ms, step=10))
        pl.tick_params(labelsize = 12.0)
        ax3.xaxis.set_major_formatter(fmt)
        ax3.yaxis.set_major_formatter(fmt)
        #pl.show()
        pl.tight_layout(pad = 1.0)
        fig.savefig("/home/wilhelm/Schreibtisch/Fig7A.png", bbox_inches='tight', dpi= 400)
        pl.show()
        
        #exit()

        fig = pl.figure(1, figsize = (7.5, 1.5), dpi = 300)
        #plots for how many neurons are spiking
        ax6 = fig.add_subplot(121)
        ax6.annotate('B', xy=(-0.3, 1.1), xycoords='axes fraction', fontsize= 24.0) #36.0

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
        fig.savefig("/home/wilhelm/Schreibtisch/Fig7B.png", bbox_inches='tight', dpi = 400)
        pl.show()

  

        
    print('Global oscillation frequencies (I/E):', freq_max_i,",",  freq_max_e)
    print('Mean spikes around ripple for active subpopulations (I/E)', mean_spikes_I_nonsilent,",",  mean_spikes_E_nonsilent)
    print('===============================')
    return [freq_max_i, freq_max_e]

M = 1
for MC_simulation in np.arange(M):
    N_eCA3 = 15000
    dendritic_window = 2.0 
    p = 0.1
    mu_in = 0.0 
    sigma_in = 0.75
    result = spike_generation(plot_indicator = True, N_eCA3 = N_eCA3, dendritic_spike_threshold = 4, event_delay = 5.0*ms, dendritic_window = dendritic_window*ms, peak_rate = 8.0, p_IE= p, mu_LN = mu_in, sigma_LN = sigma_in)
