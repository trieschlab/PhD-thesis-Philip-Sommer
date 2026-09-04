# energy-dependent simulation functions

# import predefined functions
import Functions.analysis_functions as af
import Functions.synaptic_input_functions as sif

# import packages for simulation and calculation
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import pickle
import os
os.environ["CC"] = "gcc"
os.environ["CXX"] = "g++"
from brian2 import *
# brian2 is unstable with python 3.12. In case of errors either use:
#prefs.codegen.target = "numpy"  # or



################################ parameter functions ################################

def energy_dependent_sim_parameters(parameter_mode='CTR', membrane_noise=0.0, s_E=0.85):
    # return default parameters for the energy-dependent LIF model
    
    # input
    # parameter_mode selects parameter set ('CTR' control or 'FR' food restriction)
    # membrane_noise is the membrane noise amplitude in mV/sqrt(ms)
    # s_E is the energy supply fraction of rho_E_target
    
    # output
    # C_m membrane capacitance in pF
    # V_rest initial resting potential in mV
    # R_m membrane resistance in MOhm
    # g_L leak conductance in nS
    # V_thresh spike threshold in mV
    # V_reset reset potential in mV
    # Delta_T_ad AdEx exponential slope factor in mV
    # tau_w_ad AdEx adaptation current time constant in ms
    # a_ad AdEx subthreshold adaptation conductance in nS
    # b_ad AdEx spike-triggered adaptation increment in nA
    # rate_window firing-rate estimation window in ms
    # tau_Ca decay time constant of activity trace in ms
    # A_Ca spike-triggered increment of activity trace in Hz
    # E_e excitatory reversal potential in mV
    # E_i inhibitory reversal potential in mV
    # tau_e excitatory synaptic decay constant in ms
    # tau_i inhibitory synaptic decay constant in ms
    # membrane_noise membrane noise amplitude in mV/sqrt(ms)
    # rho_E_target energy balance target
    # s_E energy supply fraction of rho_E_target that can be supplied
    # E_target_ATP normalization factor for ATP consumption
    # r_target target firing rate in Hz
    # tau_r_smooth is the time constants to low-pass filter/smooth the firing rate 
    # tau_r_deriv is the time constants of the firing rate derivative control term
    # tau_E_smooth is the time constants to low-pass filter/smooth the energy availability
    # dt_sim numerical integration time step in ms
    # dt_energy energy-controller update interval in ms
    # tau_w_scale synaptic scaling time constant in ms
    # tau_V_rest resting potential adaptation time constant in ms
    # tau_R_m membrane resistance adaptation time constant in ms
    
    # E_HK housekeeping energy consumption in ATP/s
    # V_Na sodium reversal potential in V
    # V_K potassium reversal potential in V
    # V_h h-channel reversal potential in V
    # alpha_atp relative sodium conductance factor
    # e_c elementary charge in Coulomb
    # A_R_m effective membrane resistance scaling factor
    # A_overlap_AP sodium-potassium current overlap factor for spikes
    # A_overlap_syn sodium-potassium current overlap factor for synapses
    # d_axon axon diameter in m
    # l_axon axon length in m
    # d_soma soma diameter in m
    # d_dendrite dendrite diameter in m
    # l_dendrite dendrite length in m
    # c_m_area membrane capacitance per area in F/m^2
    # V_peak spike peak voltage in mV
    # N_boutons number of synaptic boutons
    # E_glu_rec energy cost of vesicle recycling in ATP
    # E_Ca_ex energy cost of calcium extrusion in ATP

    # define neuron parameters
    C_m = 100.0 # pF
    
    if parameter_mode == 'CTR':
        V_rest = -72.0 # mV
        R_m = 93.0 # MOhm
    elif parameter_mode == 'FR':
        V_rest = -66.0 # mV
        R_m = 113.0 # MOhm

    g_L = 1e3 / R_m # nS
    V_thresh = -50.0 # mV
    V_reset = V_rest # mV
    
    # AdExp parameters
    Delta_T_ad = 2 # mV
    tau_w_ad = 500 # ms
    a_ad = 4 # nS
    b_ad = 0.02 # nA
    
    rate_window = 1000.0 # ms
    tau_Ca = 1000.0 # ms
    A_Ca = 1.0 # Hz

    E_e = 0.0 # mV
    E_i = -80.0 # mV
    tau_e = 3.0 # ms
    tau_i = 5.0 # ms

    rho_E_target = 1.0 # 1
    E_target_ATP = 1.6e9 # ATP/s
    r_target = 1.0 # Hz

    tau_r_smooth = 10000 # ms
    tau_E_smooth = 10000 # ms
    tau_r_deriv = 2500.0 # ms

    dt_sim = 0.1 # ms
    dt_energy = 1 # ms
    
    tau_V_rest = 50000.0 # ms
    tau_R_m = tau_V_rest / 2.5 # ms
    tau_w_scale = tau_R_m / 5 # ms


    E_HK = 5.38e8 # ATP/s
    
    V_Na = 0.05 # V
    V_K = -0.1 # V
    V_h = -0.043 # V

    alpha_atp = 0.05 # 1
    e_c = 1.602e-19 # C
    A_R_m = 0.91 # 1

    A_overlap_AP = 1.24 # 1
    A_overlap_syn = 1.1 # 1

    d_axon = 3e-7 # m
    l_axon = 4e-2 # m

    d_soma = 2.5e-5 # m

    d_dendrite = 3 * d_axon # m
    l_dendrite = l_axon / 9.0 # m

    c_m_area = 1e-2 # F/m^2

    V_peak = 25.0 # mV

    N_boutons = 8000.0 # 1

    E_glu_rec = 1.468e4 # ATP
    E_Ca_ex = 1.2e4 # ATP

    return C_m, V_rest, R_m, g_L, V_thresh, V_reset, Delta_T_ad, tau_w_ad, a_ad, b_ad, rate_window, tau_Ca, A_Ca, E_e, E_i, tau_e, tau_i, membrane_noise, rho_E_target, s_E, E_target_ATP, r_target, tau_r_smooth, tau_r_deriv, tau_E_smooth, dt_sim, dt_energy, tau_w_scale, tau_V_rest, tau_R_m, E_HK, V_Na, V_K, V_h, alpha_atp, e_c, A_R_m, A_overlap_AP, A_overlap_syn, d_axon, l_axon, d_soma, d_dendrite, l_dendrite, c_m_area, V_peak, N_boutons, E_glu_rec, E_Ca_ex

def energy_parameters_brian(T, N_e, N_i, spike_times_e, spike_times_i, w_e, w_i, C_m, V_rest, R_m, g_L, V_thresh, V_reset, Delta_T_ad, tau_w_ad, a_ad, b_ad, rate_window, tau_Ca, A_Ca, E_e, E_i, tau_e, tau_i, membrane_noise, rho_E_target, s_E, E_target_ATP, r_target, tau_r_smooth, tau_r_deriv, tau_E_smooth, dt_sim, dt_energy, tau_w_scale, tau_V_rest, tau_R_m, E_HK, V_Na, V_K, V_h, alpha_atp, e_c, A_R_m, A_overlap_AP, A_overlap_syn, d_axon, l_axon, d_soma, d_dendrite, l_dendrite, c_m_area, V_peak, N_boutons, E_glu_rec, E_Ca_ex):
    # transforms parameters for brian2
    
    # input
    # T simulation duration in ms
    # N_e number of excitatory synapses
    # N_i number of inhibitory synapses
    # spike_times_e excitatory spike dictionary {synapse_index: spike_times_ms}
    # spike_times_i inhibitory spike dictionary {synapse_index: spike_times_ms}
    # w_e excitatory baseline synaptic weights in nS
    # w_i inhibitory baseline synaptic weights in nS
    # C_m membrane capacitance in pF
    # V_rest initial resting potential in mV
    # R_m membrane resistance in MOhm
    # g_L leak conductance in nS
    # V_thresh spike threshold in mV
    # V_reset reset potential in mV
    # Delta_T_ad AdEx exponential slope factor in mV
    # tau_w_ad AdEx adaptation current time constant in ms
    # a_ad AdEx subthreshold adaptation conductance in nS
    # b_ad AdEx spike-triggered adaptation increment in nA
    # rate_window rate estimation window in ms
    # tau_Ca decay constant of activity trace in ms
    # A_Ca spike-triggered increment of activity trace in Hz
    # E_e excitatory reversal potential in mV
    # E_i inhibitory reversal potential in mV
    # tau_e excitatory synaptic decay constant in ms
    # tau_i inhibitory synaptic decay constant in ms
    # membrane_noise membrane noise amplitude in mV/sqrt(ms)
    # rho_E_target energy balance target
    # s_E energy supply fraction of rho_E_target that can be supplied
    # E_target_ATP normalization factor for ATP consumption
    # r_target target firing rate in Hz
    # tau_r_smooth is the time constants to low-pass filter/smooth the firing rate 
    # tau_r_deriv is the time constants of the firing rate derivative control term
    # tau_E_smooth is the time constants to low-pass filter/smooth the energy availability
    # dt_sim numerical integration time step in ms
    # dt_energy energy-controller update interval in ms
    # tau_w_scale synaptic scaling time constant in ms
    # tau_V_rest resting potential adaptation time constant in ms
    # tau_R_m membrane resistance adaptation time constant in ms
    # E_HK housekeeping energy consumption in ATP/s
    # V_Na sodium reversal potential in V
    # V_K potassium reversal potential in V
    # V_h h-channel reversal potential in V
    # alpha_atp relative sodium conductance factor
    # e_c elementary charge in Coulomb
    # A_R_m effective membrane resistance scaling factor
    # A_overlap_AP sodium-potassium current overlap factor for spikes
    # A_overlap_syn sodium-potassium current overlap factor for synapses
    # d_axon axon diameter in m
    # l_axon axon length in m
    # d_soma soma diameter in m
    # d_dendrite dendrite diameter in m
    # l_dendrite dendrite length in m
    # c_m_area membrane capacitance per area in F/m^2
    # V_peak spike peak voltage in mV
    # N_boutons number of synaptic boutons
    # E_glu_rec energy cost of vesicle recycling in ATP
    # E_Ca_ex energy cost of calcium extrusion in ATP
    
    # output
    # all parameters converted to brian2 units

    T_b = T*ms

    E_e_b = E_e*mV
    E_i_b = E_i*mV

    w_e_b = w_e*nS
    w_i_b = w_i*nS

    tau_e_b = tau_e*ms
    tau_i_b = tau_i*ms

    C_m_b = C_m*pF
    g_L_b = g_L*nS

    V_rest_b = V_rest*mV
    V_thresh_b = V_thresh*mV
    V_reset_b = V_reset*mV

    Delta_T_ad_b = Delta_T_ad*mV
    tau_w_ad_b = tau_w_ad*ms
    a_ad_b = a_ad*nS
    b_ad_b = b_ad*nA

    rate_window_b = rate_window*ms

    tau_Ca_b = tau_Ca*ms
    A_Ca_b = A_Ca*Hz

    membrane_noise_b = membrane_noise*mV/sqrt(ms)#*volt/sqrt(second)
    
    E_target_ATP_b = E_target_ATP*Hz
    r_target_b = r_target*Hz
    
    tau_r_smooth_b = tau_r_smooth*ms
    tau_r_deriv_b = tau_r_deriv*ms
    tau_E_smooth_b = tau_E_smooth*ms

    dt_sim_b = dt_sim*ms
    dt_energy_b = dt_energy*ms

    tau_w_scale_b = tau_w_scale*ms
    tau_V_rest_b = tau_V_rest*ms
    tau_R_m_b = tau_R_m*ms
    
    E_HK_b = E_HK * Hz
    
    V_Na_b = V_Na*volt
    V_K_b = V_K*volt
    V_h_b = V_h*volt

    e_c_b = e_c*coulomb

    d_axon_b = d_axon*meter
    l_axon_b = l_axon*meter

    d_soma_b = d_soma*meter

    d_dendrite_b = d_dendrite*meter
    l_dendrite_b = l_dendrite*meter

    c_m_area_b = c_m_area*farad/meter**2

    V_peak_b = V_peak*mV

    S_axon = pi*d_axon_b*l_axon_b
    S_soma = pi*d_soma_b**2
    S_dendrite = pi*d_dendrite_b*l_dendrite_b

    S_m_b = S_axon + S_soma + S_dendrite
    

    return T_b, N_e, N_i, spike_times_e, spike_times_i, w_e_b, w_i_b, C_m_b, V_rest_b,  g_L_b, V_thresh_b, V_reset_b, Delta_T_ad_b, tau_w_ad_b, a_ad_b, b_ad_b, rate_window_b, tau_Ca_b, A_Ca_b, E_e_b, E_i_b, tau_e_b, tau_i_b, membrane_noise_b, rho_E_target, s_E, E_target_ATP_b, r_target_b, tau_r_smooth_b, tau_r_deriv_b, tau_E_smooth_b, dt_sim_b, dt_energy_b, tau_w_scale_b, tau_V_rest_b, tau_R_m_b, E_HK_b, V_Na_b, V_K_b, V_h_b, alpha_atp, e_c_b, A_R_m, A_overlap_AP, A_overlap_syn, S_m_b, c_m_area_b, V_peak_b, N_boutons, E_glu_rec, E_Ca_ex

################################ energy-dependent LIF ################################

def energy_dependent_LIF_P_energy_PD_rate_control(T, N_e, N_i, spike_times_e, spike_times_i, w_e, w_i, C_m, V_rest, g_L, V_thresh, V_reset, rate_window, tau_Ca, A_Ca, E_e, E_i, tau_e, tau_i, membrane_noise, rho_E_target, s_E, E_target_ATP, r_target, tau_r_smooth, tau_r_deriv, tau_E_smooth, dt_sim, dt_plasticity, tau_w_scale, tau_V_rest, tau_R_m, E_HK, V_Na, V_K, V_h, alpha_atp, e_c, A_R_m, A_overlap_AP, A_overlap_syn, S_m, c_m_area, V_peak, N_boutons, E_glu_rec, E_Ca_ex):
    # simulation of energy-dependent leaky integrate-and-fire neuron
    # input
    # T simulation duration in ms
    # N_e number of excitatory synapses
    # N_i number of inhibitory synapses
    # spike_times_e excitatory spike dictionary {synapse_index: spike_times_ms}
    # spike_times_i inhibitory spike dictionary {synapse_index: spike_times_ms}
    # w_e baseline excitatory synaptic weights in nS
    # w_i baseline inhibitory synaptic weights in nS
    # C_m membrane capacitance in pF
    # V_rest initial resting potential in mV
    # g_L initial leak conductance in nS
    # V_thresh spike threshold in mV
    # V_reset reset potential in mV
    # rate_window rate estimation window in ms
    # tau_Ca decay constant of activity trace in ms
    # A_Ca spike-triggered increment of activity trace in Hz
    # E_e excitatory reversal potential in mV
    # E_i inhibitory reversal potential in mV
    # tau_e excitatory synaptic decay constant in ms
    # tau_i inhibitory synaptic decay constant in ms
    # membrane_noise membrane noise amplitude in mV/sqrt(ms)
    # rho_E_target energy balance target
    # s_E energy supply fraction of rho_E_target that can be supplied
    # E_target_ATP normalization factor for ATP consumption in ATP/s
    # r_target target firing rate in Hz
    # tau_r_smooth is the time constants to low-pass filter/smooth the firing rate 
    # tau_r_deriv is the time constants of the firing rate derivative control term
    # tau_E_smooth is the time constants to low-pass filter/smooth the energy availability
    # dt_sim numerical integration time step in ms
    # dt_plasticity plasticity-controller update interval in ms
    # tau_w_scale synaptic scaling time constant in ms
    # tau_V_rest resting potential adaptation time constant in ms
    # tau_R_m membrane resistance adaptation time constant in ms
    # E_HK housekeeping energy consumption in ATP/s
    # V_Na sodium reversal potential in V
    # V_K potassium reversal potential in V
    # V_h h-channel reversal potential in V
    # alpha_atp relative sodium conductance factor
    # e_c elementary charge in Coulomb
    # A_R_m effective membrane resistance scaling factor
    # A_overlap_AP sodium-potassium current overlap factor for spikes
    # A_overlap_syn sodium-potassium current overlap factor for synapses
    # S_m total membrane surface area in m^2
    # c_m_area membrane capacitance per area in F/m^2
    # V_peak spike peak voltage in mV
    # N_boutons number of synaptic boutons
    # E_glu_rec energy cost of vesicle recycling in ATP
    # E_Ca_ex energy cost of calcium extrusion in ATP
    
    # output
    # time simulation time in s
    # V_m membrane voltage trace in mV
    # spike_times_post postsynaptic spike times in ms
    # rate_estimate_post postsynaptic firing-rate estimate in Hz
    # I_syn_e excitatory synaptic current trace in nA
    # I_syn_i inhibitory synaptic current trace in nA
    # g_e excitatory synaptic conductance trace in nS
    # g_i inhibitory synaptic conductance trace in nS
    # E_cons_dyn normalized energy consumption trace
    # E_prod_dyn normalized energy production trace
    # rho_E instantaneous energy balance
    # rho_E_smooth smoothed energy balance
    # rho_E_P_control_dyn proportional energy-control signal
    # r_Ca calcium-based activity trace in Hz
    # r_Ca_dyn_smooth smoothed calcium-based firing-rate estimate in Hz
    # dr_Ca_dyn_smooth derivative of smoothed calcium-based firing-rate estimate in Hz/s
    # r_PD_control_dyn proportional-derivative firing-rate control signal
    # w_syn_e_dyn dynamic mean excitatory synaptic weight in nS
    # V_rest_dyn dynamic resting potential trace in mV
    # R_m_dyn dynamic membrane resistance trace in MOhm


    start_scope()
    defaultclock.dt = dt_sim

    N = 1
    V_thresh_spike_detection = V_thresh
    
    # convert spike_times_e and spike_times_i to SpikeGeneratorGroup format
    indices_e, times_e = zip(*[(i, spike) for i, spikes in spike_times_e.items() for spike in spikes])
    indices_i, times_i = zip(*[(i, spike) for i, spikes in spike_times_i.items() for spike in spikes])
    G_e = SpikeGeneratorGroup(N_e, indices_e, times_e)
    G_i = SpikeGeneratorGroup(N_i, indices_i, times_i)
    
    eqs='''
    dV_m/dt=(g_L_dyn*(V_rest_dyn-V_m)+I_syn_e+I_syn_i)/C_m+membrane_noise*xi : volt
    
    I_syn_e=g_e*(E_e-V_m) : amp
    I_syn_i=g_i*(E_i-V_m) : amp

    dg_e/dt=-g_e/tau_e : siemens
    dg_i/dt=-g_i/tau_i : siemens

    E_HK_dyn : Hz
    E_RP_dyn : Hz
    E_AP_dyn : Hz
    E_ST_dyn : Hz
    E_glu_dyn : Hz
    E_Ca_dyn : Hz
    E_tot_dyn : Hz

    E_cons_dyn : 1
    E_prod_dyn : 1
    rho_E : 1
    rho_E_smooth : 1
    rho_E_P_control_dyn : 1
    
    dCa/dt=-Ca/tau_Ca : Hz
    r_Ca_dyn_smooth : Hz
    dr_Ca_dyn_smooth : Hz/second
    r_PD_control_dyn : 1
    
    w_scale_dyn : 1
    V_rest_dyn : volt
    R_m_dyn : ohm
    g_L_dyn : siemens
    '''
    
    reset='''
    V_m=V_reset
    Ca+=A_Ca
    '''
    
    neuron=NeuronGroup(N,eqs,threshold='V_m>V_thresh_spike_detection',reset=reset,refractory=1*ms,method='euler')

    neuron.V_m=V_rest
    
    neuron.rho_E=rho_E_target 
    neuron.rho_E_smooth = rho_E_target 
    neuron.rho_E_P_control_dyn = 0.0

    neuron.Ca=r_target
    neuron.r_Ca_dyn_smooth = r_target
    neuron.dr_Ca_dyn_smooth = 0*Hz/second
    neuron.r_PD_control_dyn = 0.0

    neuron.R_m_dyn=1/g_L
    neuron.g_L_dyn=g_L
    neuron.V_rest_dyn=V_rest
    neuron.w_scale_dyn=1.0
    
    exc_syn=Synapses(G_e,neuron,'w_base:siemens',on_pre='g_e+=w_base*w_scale_dyn_post') #w_scale_dyn')
    exc_syn.connect()
    exc_syn.w_base=w_e

    inh_syn=Synapses(G_i,neuron,'w:siemens',on_pre='g_i+=w')
    inh_syn.connect()
    inh_syn.w=w_i

    neuron.namespace.update(locals())
    
    neuron.run_regularly('''
    on = int(t > 10*second)
    on_w = int(t > 10*second)
    
    E_HK_dyn=E_HK

    R_eff=A_R_m*R_m_dyn
    g_Na=(9*alpha_atp/R_eff)*((V_rest_dyn-V_h)/(9*(1+alpha_atp)*(V_rest_dyn-V_h)+12*(V_K-V_rest_dyn)+8*alpha_atp*(V_Na-V_rest_dyn)))
    g_h=(4/R_eff)*((3*(V_K-V_rest_dyn)+2*alpha_atp*(V_Na-V_rest_dyn))/(9*(1+alpha_atp)*(V_rest_dyn-V_h)+12*(V_K-V_rest_dyn)+8*alpha_atp*(V_Na-V_rest_dyn)))
    E_RP_dyn=((g_Na*(V_Na-V_rest_dyn)/3)+(g_h*(V_h-V_rest_dyn)/4))/e_c
    
    Delta_V=V_peak-V_thresh
    E_AP_dyn=Ca*A_overlap_AP*c_m_area*S_m*Delta_V/(3*e_c)

    I_e_pos=clip(I_syn_e,0*amp,inf*amp)
    E_ST_dyn=I_e_pos*A_overlap_syn/(3*e_c)

    p_ves=0.6*exp(-Ca/Hz/5)
    E_glu_dyn=Ca*N_boutons*p_ves*E_glu_rec
    E_Ca_dyn=Ca*N_boutons*p_ves*E_Ca_ex

    E_tot_dyn=E_HK_dyn+E_RP_dyn+E_AP_dyn+E_ST_dyn+E_glu_dyn+E_Ca_dyn
    E_cons_dyn=E_tot_dyn/E_target_ATP

    E_prod_dyn=s_E*rho_E_target 

    rho_E=E_prod_dyn/E_cons_dyn
    rho_E_smooth = rho_E_smooth + dt * (rho_E - rho_E_smooth) / tau_E_smooth
    rho_E_P_control_dyn = (rho_E_target  - rho_E_smooth) 
    
    dr_Ca_dyn_smooth = (Ca - r_Ca_dyn_smooth) / tau_r_smooth
    r_PD_control_dyn = (r_target - r_Ca_dyn_smooth - tau_r_deriv * dr_Ca_dyn_smooth) / Hz
    r_Ca_dyn_smooth = r_Ca_dyn_smooth + dt * dr_Ca_dyn_smooth
    
    V_rest_dyn = V_rest_dyn - on * dt * V_rest_dyn * r_PD_control_dyn / tau_V_rest
    R_m_dyn = R_m_dyn + on * dt * R_m_dyn * r_PD_control_dyn / tau_R_m
    g_L_dyn = 1/R_m_dyn

    w_scale_dyn=w_scale_dyn - on_w * dt * w_scale_dyn * rho_E_P_control_dyn / tau_w_scale
    ''',dt=dt_plasticity)
    
    spikes=SpikeMonitor(neuron)
    rate_monitor = PopulationRateMonitor(neuron)
    post_states = StateMonitor(neuron, ('V_m', 'I_syn_e', 'I_syn_i', 'g_e', 'g_i', 'Ca', 'rho_E', 'E_prod_dyn', 'E_cons_dyn', 'V_rest_dyn', 'R_m_dyn', 'w_scale_dyn', 'E_tot_dyn', 'E_RP_dyn', 'E_AP_dyn', 'E_ST_dyn', 'E_glu_dyn', 'E_Ca_dyn', 'rho_E_smooth', 'rho_E_P_control_dyn', 'r_Ca_dyn_smooth', 'dr_Ca_dyn_smooth', 'r_PD_control_dyn'), record=True) 
    
    run(T, report='text')
    
    time = post_states.t / second
    V_m = post_states.V_m[0] / mV
    spike_times_post = spikes.t / ms
    rate_estimate_post = rate_monitor.smooth_rate(window='gaussian', width=rate_window) / Hz
    I_syn_e = post_states.I_syn_e[0] / nA
    I_syn_i = post_states.I_syn_i[0] / nA
    g_e = post_states.g_e[0] / nS
    g_i = post_states.g_i[0] / nS
    E_cons_dyn = post_states.E_cons_dyn[0] / 1
    E_prod_dyn = post_states.E_prod_dyn[0] / 1
    rho_E = post_states.rho_E[0] / 1
    rho_E_smooth = post_states.rho_E_smooth[0] / 1
    rho_E_P_control_dyn = post_states.rho_E_P_control_dyn[0] / 1
    r_Ca = post_states.Ca[0] / Hz
    r_Ca_dyn_smooth = post_states.r_Ca_dyn_smooth[0] / Hz
    dr_Ca_dyn_smooth = post_states.dr_Ca_dyn_smooth[0] / (Hz/second)
    r_PD_control_dyn = post_states.r_PD_control_dyn[0] / 1
    w_syn_e_dyn = post_states.w_scale_dyn[0]*np.mean(w_e)*1e9/siemens
    V_rest_dyn = post_states.V_rest_dyn[0] / mV
    R_m_dyn = post_states.R_m_dyn[0] / Mohm

    return time, V_m, spike_times_post, rate_estimate_post, I_syn_e, I_syn_i, g_e, g_i, E_cons_dyn, E_prod_dyn, rho_E, rho_E_smooth, rho_E_P_control_dyn, r_Ca, r_Ca_dyn_smooth, dr_Ca_dyn_smooth, r_PD_control_dyn, w_syn_e_dyn, V_rest_dyn, R_m_dyn
    

################################ energy-dependent AdExp ################################

def energy_dependent_AdExp_P_energy_PD_rate_control(T, N_e, N_i, spike_times_e, spike_times_i, w_e, w_i, C_m, V_rest, g_L, V_thresh, V_reset, Delta_T_ad, tau_w_ad, a_ad, b_ad, rate_window, tau_Ca, A_Ca, E_e, E_i, tau_e, tau_i, membrane_noise, rho_E_target, s_E, E_target_ATP, r_target, tau_r_smooth, tau_r_deriv, tau_E_smooth, dt_sim, dt_plasticity, tau_w_scale, tau_V_rest, tau_R_m, E_HK, V_Na, V_K, V_h, alpha_atp, e_c, A_R_m, A_overlap_AP, A_overlap_syn, S_m, c_m_area, V_peak, N_boutons, E_glu_rec, E_Ca_ex):
    # simulation of energy-dependent adaptive exponential integrate-and-fire neuron
    # input
    # T simulation duration in ms
    # N_e number of excitatory synapses
    # N_i number of inhibitory synapses
    # spike_times_e excitatory spike dictionary {synapse_index: spike_times_ms}
    # spike_times_i inhibitory spike dictionary {synapse_index: spike_times_ms}
    # w_e baseline excitatory synaptic weights in nS
    # w_i baseline inhibitory synaptic weights in nS
    # C_m membrane capacitance in pF
    # V_rest initial resting potential in mV
    # g_L initial leak conductance in nS
    # V_thresh spike threshold in mV
    # V_reset reset potential in mV
    # Delta_T_ad AdEx exponential slope factor in mV
    # tau_w_ad AdEx adaptation current time constant in ms
    # a_ad AdEx subthreshold adaptation conductance in nS
    # b_ad AdEx spike-triggered adaptation increment in nA
    # rate_window rate estimation window in ms
    # tau_Ca decay constant of activity trace in ms
    # A_Ca spike-triggered increment of activity trace in Hz
    # E_e excitatory reversal potential in mV
    # E_i inhibitory reversal potential in mV
    # tau_e excitatory synaptic decay constant in ms
    # tau_i inhibitory synaptic decay constant in ms
    # membrane_noise membrane noise amplitude in mV/sqrt(ms)
    # rho_E_target energy balance target
    # s_E energy supply fraction of rho_E_target that can be supplied
    # E_target_ATP normalization factor for ATP consumption in ATP/s
    # r_target target firing rate in Hz
    # tau_r_smooth is the time constants to low-pass filter/smooth the firing rate 
    # tau_r_deriv is the time constants of the firing rate derivative control term
    # tau_E_smooth is the time constants to low-pass filter/smooth the energy availability
    # dt_sim numerical integration time step in ms
    # dt_plasticity plasticity-controller update interval in ms
    # tau_w_scale synaptic scaling time constant in ms
    # tau_V_rest resting potential adaptation time constant in ms
    # tau_R_m membrane resistance adaptation time constant in ms
    # E_HK housekeeping energy consumption in ATP/s
    # V_Na sodium reversal potential in V
    # V_K potassium reversal potential in V
    # V_h h-channel reversal potential in V
    # alpha_atp relative sodium conductance factor
    # e_c elementary charge in Coulomb
    # A_R_m effective membrane resistance scaling factor
    # A_overlap_AP sodium-potassium current overlap factor for spikes
    # A_overlap_syn sodium-potassium current overlap factor for synapses
    # S_m total membrane surface area in m^2
    # c_m_area membrane capacitance per area in F/m^2
    # V_peak spike peak voltage in mV
    # N_boutons number of synaptic boutons
    # E_glu_rec energy cost of vesicle recycling in ATP
    # E_Ca_ex energy cost of calcium extrusion in ATP
    
    
    # output
    # time simulation time in s
    # V_m membrane voltage trace in mV
    # w_ad adaptation current trace in nA
    # spike_times_post postsynaptic spike times in ms
    # rate_estimate_post postsynaptic firing-rate estimate in Hz
    # I_syn_e excitatory synaptic current trace in nA
    # I_syn_i inhibitory synaptic current trace in nA
    # g_e excitatory synaptic conductance trace in nS
    # g_i inhibitory synaptic conductance trace in nS
    # E_cons_dyn normalized energy consumption trace
    # E_prod_dyn normalized energy production trace
    # rho_E instantaneous energy balance
    # rho_E_smooth smoothed energy balance
    # rho_E_P_control_dyn proportional energy-control signal
    # r_Ca calcium-based activity trace in Hz
    # r_Ca_dyn_smooth smoothed calcium-based firing-rate estimate in Hz
    # dr_Ca_dyn_smooth derivative of smoothed calcium-based firing-rate estimate in Hz/s
    # r_PD_control_dyn proportional-derivative firing-rate control signal
    # w_syn_e_dyn dynamic mean excitatory synaptic weight in nS
    # V_rest_dyn dynamic resting potential trace in mV
    # R_m_dyn dynamic membrane resistance trace in MOhm


    start_scope()
    defaultclock.dt = dt_sim

    N = 1

    V_T_ad = V_thresh - 2.5 * Delta_T_ad
    V_thresh_spike_detection = V_thresh + 2.5 * Delta_T_ad
    
    # convert spike_times_e and spike_times_i to SpikeGeneratorGroup format
    indices_e, times_e = zip(*[(i, spike) for i, spikes in spike_times_e.items() for spike in spikes])
    indices_i, times_i = zip(*[(i, spike) for i, spikes in spike_times_i.items() for spike in spikes])
    G_e = SpikeGeneratorGroup(N_e, indices_e, times_e)
    G_i = SpikeGeneratorGroup(N_i, indices_i, times_i)
    
    eqs='''
    dV_m/dt=(g_L_dyn*(V_rest_dyn-V_m)+g_L_dyn*Delta_T_ad*exp((V_m-V_T_ad)/Delta_T_ad)-w_ad+I_syn_e+I_syn_i)/C_m+membrane_noise*xi : volt
    dw_ad/dt=(a_ad*(V_m-V_rest_dyn)-w_ad)/tau_w_ad : amp

    I_syn_e=g_e*(E_e-V_m) : amp
    I_syn_i=g_i*(E_i-V_m) : amp

    dg_e/dt=-g_e/tau_e : siemens
    dg_i/dt=-g_i/tau_i : siemens

    E_HK_dyn : Hz
    E_RP_dyn : Hz
    E_AP_dyn : Hz
    E_ST_dyn : Hz
    E_glu_dyn : Hz
    E_Ca_dyn : Hz
    E_tot_dyn : Hz

    E_cons_dyn : 1
    E_prod_dyn : 1
    rho_E : 1
    rho_E_smooth : 1
    rho_E_P_control_dyn : 1
    
    dCa/dt=-Ca/tau_Ca : Hz
    r_Ca_dyn_smooth : Hz
    dr_Ca_dyn_smooth : Hz/second
    r_PD_control_dyn : 1
    
    w_scale_dyn : 1
    V_rest_dyn : volt
    R_m_dyn : ohm
    g_L_dyn : siemens
    '''
    
    reset='''
    V_m=V_reset
    w_ad+=b_ad
    Ca+=A_Ca
    '''
    
    neuron=NeuronGroup(N,eqs,threshold='V_m>V_thresh_spike_detection',reset=reset,refractory=1*ms,method='euler')

    neuron.V_m=V_rest
    neuron.w_ad=0*nA
    
    neuron.rho_E=rho_E_target 
    neuron.rho_E_smooth = rho_E_target 
    neuron.rho_E_P_control_dyn = 0.0

    neuron.Ca=r_target
    neuron.r_Ca_dyn_smooth = r_target
    neuron.dr_Ca_dyn_smooth = 0*Hz/second
    neuron.r_PD_control_dyn = 0.0

    neuron.R_m_dyn=1/g_L
    neuron.g_L_dyn=g_L
    neuron.V_rest_dyn=V_rest
    neuron.w_scale_dyn=1.0
    
    exc_syn=Synapses(G_e,neuron,'w_base:siemens',on_pre='g_e+=w_base*w_scale_dyn_post') #w_scale_dyn')
    exc_syn.connect()
    exc_syn.w_base=w_e

    inh_syn=Synapses(G_i,neuron,'w:siemens',on_pre='g_i+=w')
    inh_syn.connect()
    inh_syn.w=w_i

    neuron.namespace.update(locals())
    
    neuron.run_regularly('''
    on = int(t > 10*second)
    on_w = int(t > 10*second)
    
    E_HK_dyn=E_HK

    R_eff=A_R_m*R_m_dyn
    g_Na=(9*alpha_atp/R_eff)*((V_rest_dyn-V_h)/(9*(1+alpha_atp)*(V_rest_dyn-V_h)+12*(V_K-V_rest_dyn)+8*alpha_atp*(V_Na-V_rest_dyn)))
    g_h=(4/R_eff)*((3*(V_K-V_rest_dyn)+2*alpha_atp*(V_Na-V_rest_dyn))/(9*(1+alpha_atp)*(V_rest_dyn-V_h)+12*(V_K-V_rest_dyn)+8*alpha_atp*(V_Na-V_rest_dyn)))
    E_RP_dyn=((g_Na*(V_Na-V_rest_dyn)/3)+(g_h*(V_h-V_rest_dyn)/4))/e_c
    
    Delta_V=V_peak-V_thresh
    E_AP_dyn=Ca*A_overlap_AP*c_m_area*S_m*Delta_V/(3*e_c)

    I_e_pos=clip(I_syn_e,0*amp,inf*amp)
    E_ST_dyn=I_e_pos*A_overlap_syn/(3*e_c)

    p_ves=0.6*exp(-Ca/Hz/5)
    E_glu_dyn=Ca*N_boutons*p_ves*E_glu_rec
    E_Ca_dyn=Ca*N_boutons*p_ves*E_Ca_ex

    E_tot_dyn=E_HK_dyn+E_RP_dyn+E_AP_dyn+E_ST_dyn+E_glu_dyn+E_Ca_dyn
    E_cons_dyn=E_tot_dyn/E_target_ATP

    E_prod_dyn=s_E*rho_E_target 

    rho_E=E_prod_dyn/E_cons_dyn
    rho_E_smooth = rho_E_smooth + dt * (rho_E - rho_E_smooth) / tau_E_smooth
    rho_E_P_control_dyn = (rho_E_target  - rho_E_smooth) 
    
    dr_Ca_dyn_smooth = (Ca - r_Ca_dyn_smooth) / tau_r_smooth
    r_PD_control_dyn = (r_target - r_Ca_dyn_smooth - tau_r_deriv * dr_Ca_dyn_smooth) / Hz
    r_Ca_dyn_smooth = r_Ca_dyn_smooth + dt * dr_Ca_dyn_smooth
    
    V_rest_dyn = V_rest_dyn - on * dt * V_rest_dyn * r_PD_control_dyn / tau_V_rest
    R_m_dyn = R_m_dyn + on * dt * R_m_dyn * r_PD_control_dyn / tau_R_m
    g_L_dyn = 1/R_m_dyn

    w_scale_dyn=w_scale_dyn - on_w * dt * w_scale_dyn * rho_E_P_control_dyn / tau_w_scale
    ''',dt=dt_plasticity)
    
    spikes=SpikeMonitor(neuron)
    rate_monitor = PopulationRateMonitor(neuron)
    post_states = StateMonitor(neuron, ('V_m', 'w_ad', 'I_syn_e', 'I_syn_i', 'g_e', 'g_i', 'Ca', 'rho_E', 'E_prod_dyn', 'E_cons_dyn', 'V_rest_dyn', 'R_m_dyn', 'w_scale_dyn', 'E_tot_dyn', 'E_RP_dyn', 'E_AP_dyn', 'E_ST_dyn', 'E_glu_dyn', 'E_Ca_dyn', 'rho_E_smooth', 'rho_E_P_control_dyn', 'r_Ca_dyn_smooth', 'dr_Ca_dyn_smooth', 'r_PD_control_dyn'), record=True) 
    
    run(T, report='text')
    
    time = post_states.t / second
    V_m = post_states.V_m[0] / mV
    w_ad = post_states.w_ad[0] / nA
    spike_times_post = spikes.t / ms
    rate_estimate_post = rate_monitor.smooth_rate(window='gaussian', width=rate_window) / Hz
    I_syn_e = post_states.I_syn_e[0] / nA
    I_syn_i = post_states.I_syn_i[0] / nA
    g_e = post_states.g_e[0] / nS
    g_i = post_states.g_i[0] / nS
    E_cons_dyn = post_states.E_cons_dyn[0] / 1
    E_prod_dyn = post_states.E_prod_dyn[0] / 1
    rho_E = post_states.rho_E[0] / 1
    rho_E_smooth = post_states.rho_E_smooth[0] / 1
    rho_E_P_control_dyn = post_states.rho_E_P_control_dyn[0] / 1
    r_Ca = post_states.Ca[0] / Hz
    r_Ca_dyn_smooth = post_states.r_Ca_dyn_smooth[0] / Hz
    dr_Ca_dyn_smooth = post_states.dr_Ca_dyn_smooth[0] / (Hz/second)
    r_PD_control_dyn = post_states.r_PD_control_dyn[0] / 1
    w_syn_e_dyn = post_states.w_scale_dyn[0]*np.mean(w_e)*1e9/siemens
    V_rest_dyn = post_states.V_rest_dyn[0] / mV
    R_m_dyn = post_states.R_m_dyn[0] / Mohm

    return time, V_m, w_ad, spike_times_post, rate_estimate_post, I_syn_e, I_syn_i, g_e, g_i, E_cons_dyn, E_prod_dyn, rho_E, rho_E_smooth, rho_E_P_control_dyn, r_Ca, r_Ca_dyn_smooth, dr_Ca_dyn_smooth, r_PD_control_dyn, w_syn_e_dyn, V_rest_dyn, R_m_dyn
    
################################ reversed energy-dependent LIF ################################

def energy_dependent_LIF_reverse(T, N_e, N_i, spike_times_e, spike_times_i, w_e, w_i, C_m, V_rest, g_L, V_thresh, V_reset, rate_window, tau_Ca, A_Ca, E_e, E_i, tau_e, tau_i, membrane_noise, rho_E_target, s_E, E_target_ATP, r_target, tau_r_smooth, tau_r_deriv, tau_E_smooth, dt_sim, dt_plasticity, tau_w_scale, tau_V_rest, tau_R_m, E_HK, V_Na, V_K, V_h, alpha_atp, e_c, A_R_m, A_overlap_AP, A_overlap_syn, S_m, c_m_area, V_peak, N_boutons, E_glu_rec, E_Ca_ex):
    # simulation of reversed energy-dependent leaky integrate-and-fire neuron
    
    # input
    # T is the simulation duration in ms
    # N_e is the number of excitatory synapses
    # N_i is the number of inhibitory synapses
    # spike_times_e is a dictionary of excitatory spike times {synapse_index: spike_times_ms}
    # spike_times_i is a dictionary of inhibitory spike times {synapse_index: spike_times_ms}
    # w_e is an array of baseline excitatory synaptic weights in nS
    # w_i is an array of baseline inhibitory synaptic weights in nS
    # C_m is the membrane capacitance in pF
    # V_rest is the initial resting potential in mV
    # g_L is the initial leak conductance in nS
    # V_thresh is the spike threshold in mV
    # V_reset is the reset potential in mV
    # rate_window is the rate-estimation smoothing window in ms
    # tau_Ca is the decay time constant of the calcium/activity trace in ms
    # A_Ca is the spike-triggered increment of the calcium/activity trace in Hz
    # E_e is the excitatory reversal potential in mV
    # E_i is the inhibitory reversal potential in mV
    # tau_e is the excitatory synaptic decay time constant in ms
    # tau_i is the inhibitory synaptic decay time constant in ms
    # membrane_noise is the membrane noise amplitude in mV/sqrt(ms)
    # rho_E_target is the normalized energy-balance target
    # s_E is the energy-supply fraction of rho_E_target
    # E_target_ATP is the ATP normalization factor in ATP/s
    # r_target is the target firing rate in Hz
    # tau_r_smooth is the time constants to low-pass filter/smooth the firing rate 
    # tau_r_deriv is the time constants of the firing rate derivative control term
    # tau_E_smooth is the time constants to low-pass filter/smooth the energy availability
    # dt_sim is the numerical integration time step in ms
    # dt_plasticity is the plasticity-controller update interval in ms
    # tau_w_scale is the synaptic scaling time constant in ms
    # tau_V_rest is the resting-potential adaptation time constant in ms
    # tau_R_m is the membrane-resistance adaptation time constant in ms
    # E_HK is the housekeeping energy consumption in ATP/s
    # V_Na is the sodium reversal potential in V
    # V_K is the potassium reversal potential in V
    # V_h is the h-channel reversal potential in V
    # alpha_atp is the relative sodium conductance factor
    # e_c is the elementary charge in Coulomb
    # A_R_m is the effective membrane-resistance scaling factor
    # A_overlap_AP is the sodium-potassium current-overlap factor for spikes
    # A_overlap_syn is the sodium-potassium current-overlap factor for synapses
    # S_m is the total membrane surface area in m^2
    # c_m_area is the membrane capacitance per area in F/m^2
    # V_peak is the spike peak voltage in mV
    # N_boutons is the number of synaptic boutons
    # E_glu_rec is the vesicle-recycling energy cost in ATP
    # E_Ca_ex is the calcium-extrusion energy cost in ATP
    
    # output
    # time is the simulation time in s
    # V_m is the membrane voltage trace in mV
    # spike_times_post are the postsynaptic spike times in ms
    # rate_estimate_post is the postsynaptic firing-rate estimate in Hz
    # I_syn_e is the excitatory synaptic current trace in nA
    # I_syn_i is the inhibitory synaptic current trace in nA
    # g_e is the excitatory synaptic conductance trace in nS
    # g_i is the inhibitory synaptic conductance trace in nS
    # E_cons_dyn is the normalized energy-consumption trace
    # E_prod_dyn is the normalized energy-production trace
    # rho_E is the instantaneous energy balance
    # rho_E_smooth is the smoothed energy balance
    # rho_E_P_control_dyn is the proportional energy-control signal
    # r_Ca is the calcium-based activity trace in Hz
    # r_Ca_dyn_smooth is the smoothed calcium-based firing-rate estimate in Hz
    # dr_Ca_dyn_smooth is the derivative of the smoothed calcium-based firing-rate estimate in Hz/s
    # r_PD_control_dyn is the proportional-derivative firing-rate control signal
    # w_syn_e_dyn is the dynamic mean excitatory synaptic weight in nS
    # V_rest_dyn is the dynamic resting-potential trace in mV
    # R_m_dyn is the dynamic membrane-resistance trace in MOhm
    

    start_scope()
    defaultclock.dt = dt_sim

    N = 1

    V_thresh_spike_detection = V_thresh
    
    # convert spike_times_e and spike_times_i to SpikeGeneratorGroup format
    spike_list_e = [(i, spike) for i, spikes in spike_times_e.items() for spike in spikes]
    spike_list_i = [(i, spike) for i, spikes in spike_times_i.items() for spike in spikes]

    if len(spike_list_e) > 0:
        indices_e, times_e = zip(*spike_list_e)
    else:
        indices_e = np.asarray([], dtype=int)
        times_e = np.asarray([])*ms

    if len(spike_list_i) > 0:
        indices_i, times_i = zip(*spike_list_i)
    else:
        indices_i = np.asarray([], dtype=int)
        times_i = np.asarray([])*ms

    G_e = SpikeGeneratorGroup(N_e, indices_e, times_e)
    G_i = SpikeGeneratorGroup(N_i, indices_i, times_i)
    
    eqs='''
    dV_m/dt=(g_L_dyn*(V_rest_dyn-V_m)+I_syn_e+I_syn_i)/C_m+membrane_noise*xi : volt
    
    I_syn_e=g_e*(E_e-V_m) : amp
    I_syn_i=g_i*(E_i-V_m) : amp

    dg_e/dt=-g_e/tau_e : siemens
    dg_i/dt=-g_i/tau_i : siemens

    E_HK_dyn : Hz
    E_RP_dyn : Hz
    E_AP_dyn : Hz
    E_ST_dyn : Hz
    E_glu_dyn : Hz
    E_Ca_dyn : Hz
    E_tot_dyn : Hz

    E_cons_dyn : 1
    E_prod_dyn : 1
    rho_E : 1
    rho_E_smooth : 1
    rho_E_P_control_dyn : 1
    
    dCa/dt=-Ca/tau_Ca : Hz
    r_Ca_dyn_smooth : Hz
    dr_Ca_dyn_smooth : Hz/second
    r_PD_control_dyn : 1
    
    w_scale_dyn : 1
    V_rest_dyn : volt
    R_m_dyn : ohm
    g_L_dyn : siemens
    '''
    
    reset='''
    V_m=V_reset
    Ca+=A_Ca
    '''
    
    neuron = NeuronGroup(N, eqs, threshold='V_m>V_thresh_spike_detection', reset=reset, refractory=1*ms, method='euler')

    neuron.V_m = V_rest
    
    neuron.rho_E = rho_E_target 
    neuron.rho_E_smooth = rho_E_target 
    neuron.rho_E_P_control_dyn = 0.0

    neuron.Ca = r_target
    neuron.r_Ca_dyn_smooth = r_target
    neuron.dr_Ca_dyn_smooth = 0*Hz/second
    neuron.r_PD_control_dyn = 0.0

    neuron.R_m_dyn = 1/g_L
    neuron.g_L_dyn = g_L
    neuron.V_rest_dyn = V_rest
    neuron.w_scale_dyn = 1.0
    
    exc_syn = Synapses(G_e, neuron, 'w_base:siemens', on_pre='g_e+=w_base*w_scale_dyn_post')
    exc_syn.connect()
    exc_syn.w_base = w_e

    inh_syn = Synapses(G_i, neuron, 'w:siemens', on_pre='g_i+=w')
    inh_syn.connect()
    inh_syn.w = w_i

    neuron.namespace.update(locals())
    
    neuron.run_regularly('''
    on = int(t > 1*second)
    on_w = int(t > 1*second)
    
    E_HK_dyn=E_HK

    R_eff=A_R_m*R_m_dyn
    g_Na=(9*alpha_atp/R_eff)*((V_rest_dyn-V_h)/(9*(1+alpha_atp)*(V_rest_dyn-V_h)+12*(V_K-V_rest_dyn)+8*alpha_atp*(V_Na-V_rest_dyn)))
    g_h=(4/R_eff)*((3*(V_K-V_rest_dyn)+2*alpha_atp*(V_Na-V_rest_dyn))/(9*(1+alpha_atp)*(V_rest_dyn-V_h)+12*(V_K-V_rest_dyn)+8*alpha_atp*(V_Na-V_rest_dyn)))
    E_RP_dyn=((g_Na*(V_Na-V_rest_dyn)/3)+(g_h*(V_h-V_rest_dyn)/4))/e_c
    
    Delta_V=V_peak-V_thresh
    E_AP_dyn=Ca*A_overlap_AP*c_m_area*S_m*Delta_V/(3*e_c)

    I_e_pos=clip(I_syn_e,0*amp,inf*amp)
    E_ST_dyn=I_e_pos*A_overlap_syn/(3*e_c)

    p_ves=0.6*exp(-Ca/Hz/5)
    E_glu_dyn=Ca*N_boutons*p_ves*E_glu_rec
    E_Ca_dyn=Ca*N_boutons*p_ves*E_Ca_ex

    E_tot_dyn=E_HK_dyn+E_RP_dyn+E_AP_dyn+E_ST_dyn+E_glu_dyn+E_Ca_dyn
    E_cons_dyn=E_tot_dyn/E_target_ATP

    E_prod_dyn=s_E*rho_E_target 

    rho_E=E_prod_dyn/E_cons_dyn
    rho_E_smooth = rho_E_smooth + dt * (rho_E - rho_E_smooth) / tau_E_smooth
    rho_E_P_control_dyn = (rho_E_target  - rho_E_smooth) 
    
    dr_Ca_dyn_smooth = (Ca - r_Ca_dyn_smooth) / tau_r_smooth
    r_PD_control_dyn = (r_target - r_Ca_dyn_smooth - tau_r_deriv * dr_Ca_dyn_smooth) / Hz
    r_Ca_dyn_smooth = r_Ca_dyn_smooth + dt * dr_Ca_dyn_smooth
    
    V_rest_dyn = V_rest_dyn - on * dt * V_rest_dyn * rho_E_P_control_dyn / tau_V_rest
    R_m_dyn = R_m_dyn + on * dt * R_m_dyn * rho_E_P_control_dyn / tau_R_m
    g_L_dyn = 1/R_m_dyn

    w_scale_dyn = w_scale_dyn + on_w * dt * w_scale_dyn * r_PD_control_dyn / tau_w_scale
    ''', dt=dt_plasticity)
    
    spikes = SpikeMonitor(neuron)
    rate_monitor = PopulationRateMonitor(neuron)
    post_states = StateMonitor(neuron, ('V_m', 'I_syn_e', 'I_syn_i', 'g_e', 'g_i', 'Ca', 'rho_E', 'E_prod_dyn', 'E_cons_dyn', 'V_rest_dyn', 'R_m_dyn', 'w_scale_dyn', 'E_tot_dyn', 'E_RP_dyn', 'E_AP_dyn', 'E_ST_dyn', 'E_glu_dyn', 'E_Ca_dyn', 'rho_E_smooth', 'rho_E_P_control_dyn', 'r_Ca_dyn_smooth', 'dr_Ca_dyn_smooth', 'r_PD_control_dyn'), record=True) 
    
    run(T, report='text')
    
    time = post_states.t / second
    V_m = post_states.V_m[0] / mV
    spike_times_post = spikes.t / ms
    rate_estimate_post = rate_monitor.smooth_rate(window='gaussian', width=rate_window) / Hz
    I_syn_e = post_states.I_syn_e[0] / nA
    I_syn_i = post_states.I_syn_i[0] / nA
    g_e = post_states.g_e[0] / nS
    g_i = post_states.g_i[0] / nS
    E_cons_dyn = post_states.E_cons_dyn[0] / 1
    E_prod_dyn = post_states.E_prod_dyn[0] / 1
    rho_E = post_states.rho_E[0] / 1
    rho_E_smooth = post_states.rho_E_smooth[0] / 1
    rho_E_P_control_dyn = post_states.rho_E_P_control_dyn[0] / 1
    r_Ca = post_states.Ca[0] / Hz
    r_Ca_dyn_smooth = post_states.r_Ca_dyn_smooth[0] / Hz
    dr_Ca_dyn_smooth = post_states.dr_Ca_dyn_smooth[0] / (Hz/second)
    r_PD_control_dyn = post_states.r_PD_control_dyn[0] / 1
    w_syn_e_dyn = post_states.w_scale_dyn[0]*np.mean(w_e)*1e9/siemens
    V_rest_dyn = post_states.V_rest_dyn[0] / mV
    R_m_dyn = post_states.R_m_dyn[0] / Mohm

    return time, V_m, spike_times_post, rate_estimate_post, I_syn_e, I_syn_i, g_e, g_i, E_cons_dyn, E_prod_dyn, rho_E, rho_E_smooth, rho_E_P_control_dyn, r_Ca, r_Ca_dyn_smooth, dr_Ca_dyn_smooth, r_PD_control_dyn, w_syn_e_dyn, V_rest_dyn, R_m_dyn

################################ reversed energy-dependent AdExp ################################


def energy_dependent_AdExp_reverse(T, N_e, N_i, spike_times_e, spike_times_i, w_e, w_i, C_m, V_rest, g_L, V_thresh, V_reset, Delta_T_ad, tau_w_ad, a_ad, b_ad, rate_window, tau_Ca, A_Ca, E_e, E_i, tau_e, tau_i, membrane_noise, rho_E_target, s_E, E_target_ATP, r_target, tau_r_smooth, tau_r_deriv, tau_E_smooth, dt_sim, dt_plasticity, tau_w_scale, tau_V_rest, tau_R_m, E_HK, V_Na, V_K, V_h, alpha_atp, e_c, A_R_m, A_overlap_AP, A_overlap_syn, S_m, c_m_area, V_peak, N_boutons, E_glu_rec, E_Ca_ex):
    # simulation of reversed energy-dependent adaptive exponential integrate-and-fire neuron
    
    # input
    # T is the simulation duration in ms
    # N_e is the number of excitatory synapses
    # N_i is the number of inhibitory synapses
    # spike_times_e is a dictionary of excitatory spike times {synapse_index: spike_times_ms}
    # spike_times_i is a dictionary of inhibitory spike times {synapse_index: spike_times_ms}
    # w_e is an array of baseline excitatory synaptic weights in nS
    # w_i is an array of baseline inhibitory synaptic weights in nS
    # C_m is the membrane capacitance in pF
    # V_rest is the initial resting potential in mV
    # g_L is the initial leak conductance in nS
    # V_thresh is the spike threshold in mV
    # V_reset is the reset potential in mV
    # Delta_T_ad is the AdExp exponential slope factor in mV
    # tau_w_ad is the AdExp adaptation-current time constant in ms
    # a_ad is the AdExp subthreshold adaptation conductance in nS
    # b_ad is the AdExp spike-triggered adaptation increment in nA
    # rate_window is the rate-estimation smoothing window in ms
    # tau_Ca is the decay time constant of the calcium/activity trace in ms
    # A_Ca is the spike-triggered increment of the calcium/activity trace in Hz
    # E_e is the excitatory reversal potential in mV
    # E_i is the inhibitory reversal potential in mV
    # tau_e is the excitatory synaptic decay time constant in ms
    # tau_i is the inhibitory synaptic decay time constant in ms
    # membrane_noise is the membrane noise amplitude in mV/sqrt(ms)
    # rho_E_target is the normalized energy-balance target
    # s_E is the energy-supply fraction of rho_E_target
    # E_target_ATP is the ATP normalization factor in ATP/s
    # r_target is the target firing rate in Hz
    # tau_r_smooth is the time constant for smoothing the calcium-based firing-rate estimate in ms
    # tau_r_deriv is the derivative-control time constant in ms
    # tau_E_smooth is the time constant for smoothing energy availability in ms
    # dt_sim is the numerical integration time step in ms
    # dt_plasticity is the plasticity-controller update interval in ms
    # tau_w_scale is the synaptic scaling time constant in ms
    # tau_V_rest is the resting-potential adaptation time constant in ms
    # tau_R_m is the membrane-resistance adaptation time constant in ms
    # E_HK is the housekeeping energy consumption in ATP/s
    # V_Na is the sodium reversal potential in V
    # V_K is the potassium reversal potential in V
    # V_h is the h-channel reversal potential in V
    # alpha_atp is the relative sodium conductance factor
    # e_c is the elementary charge in Coulomb
    # A_R_m is the effective membrane-resistance scaling factor
    # A_overlap_AP is the sodium-potassium current-overlap factor for spikes
    # A_overlap_syn is the sodium-potassium current-overlap factor for synapses
    # S_m is the total membrane surface area in m^2
    # c_m_area is the membrane capacitance per area in F/m^2
    # V_peak is the spike peak voltage in mV
    # N_boutons is the number of synaptic boutons
    # E_glu_rec is the vesicle-recycling energy cost in ATP
    # E_Ca_ex is the calcium-extrusion energy cost in ATP
    
    # output
    # time is the simulation time in s
    # V_m is the membrane voltage trace in mV
    # w_ad is the adaptation current trace in nA
    # spike_times_post are the postsynaptic spike times in ms
    # rate_estimate_post is the postsynaptic firing-rate estimate in Hz
    # I_syn_e is the excitatory synaptic current trace in nA
    # I_syn_i is the inhibitory synaptic current trace in nA
    # g_e is the excitatory synaptic conductance trace in nS
    # g_i is the inhibitory synaptic conductance trace in nS
    # E_cons_dyn is the normalized energy-consumption trace
    # E_prod_dyn is the normalized energy-production trace
    # rho_E is the instantaneous energy balance
    # rho_E_smooth is the smoothed energy balance
    # rho_E_P_control_dyn is the proportional energy-control signal
    # r_Ca is the calcium-based activity trace in Hz
    # r_Ca_dyn_smooth is the smoothed calcium-based firing-rate estimate in Hz
    # dr_Ca_dyn_smooth is the derivative of the smoothed calcium-based firing-rate estimate in Hz/s
    # r_PD_control_dyn is the proportional-derivative firing-rate control signal
    # w_syn_e_dyn is the dynamic mean excitatory synaptic weight in nS
    # V_rest_dyn is the dynamic resting-potential trace in mV
    # R_m_dyn is the dynamic membrane-resistance trace in MOhm
    

    start_scope()
    defaultclock.dt = dt_sim

    N = 1

    V_T_ad = V_thresh - 2.5 * Delta_T_ad
    V_thresh_spike_detection = V_thresh + 2.5 * Delta_T_ad
    
    # convert spike_times_e and spike_times_i to SpikeGeneratorGroup format
    indices_e, times_e = zip(*[(i, spike) for i, spikes in spike_times_e.items() for spike in spikes])
    indices_i, times_i = zip(*[(i, spike) for i, spikes in spike_times_i.items() for spike in spikes])
    G_e = SpikeGeneratorGroup(N_e, indices_e, times_e)
    G_i = SpikeGeneratorGroup(N_i, indices_i, times_i)
    
    eqs='''
    dV_m/dt=(g_L_dyn*(V_rest_dyn-V_m)+g_L_dyn*Delta_T_ad*exp((V_m-V_T_ad)/Delta_T_ad)-w_ad+I_syn_e+I_syn_i)/C_m+membrane_noise*xi : volt
    dw_ad/dt=(a_ad*(V_m-V_rest_dyn)-w_ad)/tau_w_ad : amp

    I_syn_e=g_e*(E_e-V_m) : amp
    I_syn_i=g_i*(E_i-V_m) : amp

    dg_e/dt=-g_e/tau_e : siemens
    dg_i/dt=-g_i/tau_i : siemens

    E_HK_dyn : Hz
    E_RP_dyn : Hz
    E_AP_dyn : Hz
    E_ST_dyn : Hz
    E_glu_dyn : Hz
    E_Ca_dyn : Hz
    E_tot_dyn : Hz

    E_cons_dyn : 1
    E_prod_dyn : 1
    rho_E : 1
    rho_E_smooth : 1
    rho_E_P_control_dyn : 1
    
    dCa/dt=-Ca/tau_Ca : Hz
    r_Ca_dyn_smooth : Hz
    dr_Ca_dyn_smooth : Hz/second
    r_PD_control_dyn : 1
    
    w_scale_dyn : 1
    V_rest_dyn : volt
    R_m_dyn : ohm
    g_L_dyn : siemens
    '''
    
    reset='''
    V_m=V_reset
    w_ad+=b_ad
    Ca+=A_Ca
    '''
    
    neuron=NeuronGroup(N,eqs,threshold='V_m>V_thresh_spike_detection',reset=reset,refractory=1*ms,method='euler')

    neuron.V_m=V_rest
    neuron.w_ad=0*nA
    
    neuron.rho_E=rho_E_target 
    neuron.rho_E_smooth = rho_E_target 
    neuron.rho_E_P_control_dyn = 0.0

    neuron.Ca=r_target
    neuron.r_Ca_dyn_smooth = r_target
    neuron.dr_Ca_dyn_smooth = 0*Hz/second
    neuron.r_PD_control_dyn = 0.0

    neuron.R_m_dyn=1/g_L
    neuron.g_L_dyn=g_L
    neuron.V_rest_dyn=V_rest
    neuron.w_scale_dyn=1.0
    
    exc_syn=Synapses(G_e,neuron,'w_base:siemens',on_pre='g_e+=w_base*w_scale_dyn_post') #w_scale_dyn')
    exc_syn.connect()
    exc_syn.w_base=w_e

    inh_syn=Synapses(G_i,neuron,'w:siemens',on_pre='g_i+=w')
    inh_syn.connect()
    inh_syn.w=w_i

    neuron.namespace.update(locals())
    
    neuron.run_regularly('''
    on = int(t > 1*second)
    on_w = int(t > 1*second)
    
    E_HK_dyn=E_HK

    R_eff=A_R_m*R_m_dyn
    g_Na=(9*alpha_atp/R_eff)*((V_rest_dyn-V_h)/(9*(1+alpha_atp)*(V_rest_dyn-V_h)+12*(V_K-V_rest_dyn)+8*alpha_atp*(V_Na-V_rest_dyn)))
    g_h=(4/R_eff)*((3*(V_K-V_rest_dyn)+2*alpha_atp*(V_Na-V_rest_dyn))/(9*(1+alpha_atp)*(V_rest_dyn-V_h)+12*(V_K-V_rest_dyn)+8*alpha_atp*(V_Na-V_rest_dyn)))
    E_RP_dyn=((g_Na*(V_Na-V_rest_dyn)/3)+(g_h*(V_h-V_rest_dyn)/4))/e_c
    
    Delta_V=V_peak-V_thresh
    E_AP_dyn=Ca*A_overlap_AP*c_m_area*S_m*Delta_V/(3*e_c)

    I_e_pos=clip(I_syn_e,0*amp,inf*amp)
    E_ST_dyn=I_e_pos*A_overlap_syn/(3*e_c)

    p_ves=0.6*exp(-Ca/Hz/5)
    E_glu_dyn=Ca*N_boutons*p_ves*E_glu_rec
    E_Ca_dyn=Ca*N_boutons*p_ves*E_Ca_ex

    E_tot_dyn=E_HK_dyn+E_RP_dyn+E_AP_dyn+E_ST_dyn+E_glu_dyn+E_Ca_dyn
    E_cons_dyn=E_tot_dyn/E_target_ATP

    E_prod_dyn=s_E*rho_E_target 

    rho_E=E_prod_dyn/E_cons_dyn
    rho_E_smooth = rho_E_smooth + dt * (rho_E - rho_E_smooth) / tau_E_smooth
    rho_E_P_control_dyn = (rho_E_target  - rho_E_smooth) 
    
    dr_Ca_dyn_smooth = (Ca - r_Ca_dyn_smooth) / tau_r_smooth
    r_PD_control_dyn = (r_target - r_Ca_dyn_smooth - tau_r_deriv * dr_Ca_dyn_smooth) / Hz
    r_Ca_dyn_smooth = r_Ca_dyn_smooth + dt * dr_Ca_dyn_smooth
    
    V_rest_dyn = V_rest_dyn - on * dt * V_rest_dyn * rho_E_P_control_dyn / tau_V_rest
    R_m_dyn = R_m_dyn + on * dt * R_m_dyn * rho_E_P_control_dyn / tau_R_m
    g_L_dyn = 1/R_m_dyn

    w_scale_dyn=w_scale_dyn + on_w * dt * w_scale_dyn * r_PD_control_dyn / tau_w_scale
    ''',dt=dt_plasticity)
    # before
    #V_rest_dyn = V_rest_dyn - on * dt * V_rest_dyn * rho_E_P_control_dyn / tau_V_rest
    #R_m_dyn = R_m_dyn + on * dt * R_m_dyn * rho_E_P_control_dyn / tau_R_m
    
    spikes=SpikeMonitor(neuron)
    rate_monitor = PopulationRateMonitor(neuron)
    post_states = StateMonitor(neuron, ('V_m', 'w_ad', 'I_syn_e', 'I_syn_i', 'g_e', 'g_i', 'Ca', 'rho_E', 'E_prod_dyn', 'E_cons_dyn', 'V_rest_dyn', 'R_m_dyn', 'w_scale_dyn', 'E_tot_dyn', 'E_RP_dyn', 'E_AP_dyn', 'E_ST_dyn', 'E_glu_dyn', 'E_Ca_dyn', 'rho_E_smooth', 'rho_E_P_control_dyn', 'r_Ca_dyn_smooth', 'dr_Ca_dyn_smooth', 'r_PD_control_dyn'), record=True) 
    
    run(T, report='text')
    
    time = post_states.t / second
    V_m = post_states.V_m[0] / mV
    w_ad = post_states.w_ad[0] / nA
    spike_times_post = spikes.t / ms
    rate_estimate_post = rate_monitor.smooth_rate(window='gaussian', width=rate_window) / Hz
    I_syn_e = post_states.I_syn_e[0] / nA
    I_syn_i = post_states.I_syn_i[0] / nA
    g_e = post_states.g_e[0] / nS
    g_i = post_states.g_i[0] / nS
    E_cons_dyn = post_states.E_cons_dyn[0] / 1
    E_prod_dyn = post_states.E_prod_dyn[0] / 1
    rho_E = post_states.rho_E[0] / 1
    rho_E_smooth = post_states.rho_E_smooth[0] / 1
    rho_E_P_control_dyn = post_states.rho_E_P_control_dyn[0] / 1
    r_Ca = post_states.Ca[0] / Hz
    r_Ca_dyn_smooth = post_states.r_Ca_dyn_smooth[0] / Hz
    dr_Ca_dyn_smooth = post_states.dr_Ca_dyn_smooth[0] / (Hz/second)
    r_PD_control_dyn = post_states.r_PD_control_dyn[0] / 1
    w_syn_e_dyn = post_states.w_scale_dyn[0]*np.mean(w_e)*1e9/siemens
    V_rest_dyn = post_states.V_rest_dyn[0] / mV
    R_m_dyn = post_states.R_m_dyn[0] / Mohm

    return time, V_m, w_ad, spike_times_post, rate_estimate_post, I_syn_e, I_syn_i, g_e, g_i, E_cons_dyn, E_prod_dyn, rho_E, rho_E_smooth, rho_E_P_control_dyn, r_Ca, r_Ca_dyn_smooth, dr_Ca_dyn_smooth, r_PD_control_dyn, w_syn_e_dyn, V_rest_dyn, R_m_dyn
    
################################ variance controlled energy-dependent LIF ################################

def energy_dependent_LIF_variance(T, N_e, N_i, spike_times_e, spike_times_i, w_e, w_i, C_m, V_rest, g_L, V_thresh, V_reset, rate_window, tau_Ca, A_Ca, E_e, E_i, tau_e, tau_i, membrane_noise, rho_E_target, s_E, E_target_ATP, r_target, r_Ca_var_target, tau_r_smooth, tau_r_deriv, tau_dr_smooth, tau_E_smooth, dt_sim, dt_plasticity, tau_w_scale, tau_V_rest, tau_R_m, E_HK, V_Na, V_K, V_h, alpha_atp, e_c, A_R_m, A_overlap_AP, A_overlap_syn, S_m, c_m_area, V_peak, N_boutons, E_glu_rec, E_Ca_ex):
    # simulation of energy-dependent leaky integrate-and-fire neuron with variance control
    
    # input
    # T is the simulation duration in ms
    # N_e is the number of excitatory synapses
    # N_i is the number of inhibitory synapses
    # spike_times_e is a dictionary of excitatory spike times {synapse_index: spike_times_ms}
    # spike_times_i is a dictionary of inhibitory spike times {synapse_index: spike_times_ms}
    # w_e is an array of baseline excitatory synaptic weights in nS
    # w_i is an array of baseline inhibitory synaptic weights in nS
    # C_m is the membrane capacitance in pF
    # V_rest is the initial resting potential in mV
    # g_L is the initial leak conductance in nS
    # V_thresh is the spike threshold in mV
    # V_reset is the reset potential in mV
    # rate_window is the rate-estimation smoothing window in ms
    # tau_Ca is the decay time constant of the calcium/activity trace in ms
    # A_Ca is the spike-triggered increment of the calcium/activity trace in Hz
    # E_e is the excitatory reversal potential in mV
    # E_i is the inhibitory reversal potential in mV
    # tau_e is the excitatory synaptic decay time constant in ms
    # tau_i is the inhibitory synaptic decay time constant in ms
    # membrane_noise is the membrane noise amplitude in mV/sqrt(ms)
    # rho_E_target is the normalized energy-balance target
    # s_E is the energy-supply fraction of rho_E_target
    # E_target_ATP is the ATP normalization factor in ATP/s
    # r_target is the target firing rate in Hz
    # r_Ca_var_target is the target calcium-rate variance in Hz^2
    # tau_r_smooth is the time constant for smoothing the calcium-based firing-rate estimate and second moment in ms
    # tau_r_deriv is the derivative-control time constant in ms
    # tau_dr_smooth is the time constant for low-pass filtering the rate derivative in ms
    # tau_E_smooth is the time constant for smoothing energy availability in ms
    # dt_sim is the numerical integration time step in ms
    # dt_plasticity is the plasticity-controller update interval in ms
    # tau_w_scale is the synaptic scaling time constant in ms
    # tau_V_rest is the resting-potential adaptation time constant in ms
    # tau_R_m is the membrane-resistance adaptation time constant in ms
    # E_HK is the housekeeping energy consumption in ATP/s
    # V_Na is the sodium reversal potential in V
    # V_K is the potassium reversal potential in V
    # V_h is the h-channel reversal potential in V
    # alpha_atp is the relative sodium conductance factor
    # e_c is the elementary charge in Coulomb
    # A_R_m is the effective membrane-resistance scaling factor
    # A_overlap_AP is the sodium-potassium current-overlap factor for spikes
    # A_overlap_syn is the sodium-potassium current-overlap factor for synapses
    # S_m is the total membrane surface area in m^2
    # c_m_area is the membrane capacitance per area in F/m^2
    # V_peak is the spike peak voltage in mV
    # N_boutons is the number of synaptic boutons
    # E_glu_rec is the vesicle-recycling energy cost in ATP
    # E_Ca_ex is the calcium-extrusion energy cost in ATP
    
    # output
    # time is the simulation time in s
    # V_m is the membrane voltage trace in mV
    # spike_times_post are the postsynaptic spike times in ms
    # rate_estimate_post is the postsynaptic firing-rate estimate in Hz
    # I_syn_e is the excitatory synaptic current trace in nA
    # I_syn_i is the inhibitory synaptic current trace in nA
    # g_e is the excitatory synaptic conductance trace in nS
    # g_i is the inhibitory synaptic conductance trace in nS
    # E_cons_dyn is the normalized energy-consumption trace
    # E_prod_dyn is the normalized energy-production trace
    # rho_E is the instantaneous energy balance
    # rho_E_smooth is the smoothed energy balance
    # rho_E_P_control_dyn is the proportional energy-control signal
    # r_Ca is the calcium-based activity trace in Hz
    # r_Ca_dyn_smooth is the smoothed calcium-based firing-rate estimate in Hz
    # dr_Ca_dyn_smooth is the filtered derivative of the smoothed calcium-based firing-rate estimate in Hz/s
    # r_PD_control_dyn is the proportional-derivative firing-rate control signal
    # w_syn_e_dyn is the dynamic mean excitatory synaptic weight in nS
    # V_rest_dyn is the dynamic resting-potential trace in mV
    # R_m_dyn is the dynamic membrane-resistance trace in MOhm
    # r_Ca2_dyn_smooth is the smoothed calcium-rate second moment in Hz^2
    # r_Ca_var_dyn is the calcium-rate variance estimate in Hz^2
    # r_Ca_var_control_dyn is the dimensionless variance-control signal
    
    
    start_scope()
    defaultclock.dt = dt_sim

    N = 1

    V_thresh_spike_detection = V_thresh
    
    # convert spike_times_e and spike_times_i to SpikeGeneratorGroup format
    spike_list_e = [(i, spike) for i, spikes in spike_times_e.items() for spike in spikes]
    spike_list_i = [(i, spike) for i, spikes in spike_times_i.items() for spike in spikes]

    if len(spike_list_e) > 0:
        indices_e, times_e = zip(*spike_list_e)
    else:
        indices_e = np.asarray([], dtype=int)
        times_e = np.asarray([])*ms

    if len(spike_list_i) > 0:
        indices_i, times_i = zip(*spike_list_i)
    else:
        indices_i = np.asarray([], dtype=int)
        times_i = np.asarray([])*ms

    G_e = SpikeGeneratorGroup(N_e, indices_e, times_e)
    G_i = SpikeGeneratorGroup(N_i, indices_i, times_i)
    
    eqs = '''
    dV_m/dt = (g_L_dyn*(V_rest_dyn - V_m) + I_syn_e + I_syn_i)/C_m + membrane_noise*xi : volt

    I_syn_e = g_e*(E_e - V_m) : amp
    I_syn_i = g_i*(E_i - V_m) : amp

    dg_e/dt = -g_e/tau_e : siemens
    dg_i/dt = -g_i/tau_i : siemens

    E_HK_dyn : Hz
    E_RP_dyn : Hz
    E_AP_dyn : Hz
    E_ST_dyn : Hz
    E_glu_dyn : Hz
    E_Ca_dyn : Hz
    E_tot_dyn : Hz

    E_cons_dyn : 1
    E_prod_dyn : 1
    rho_E : 1
    rho_E_smooth : 1
    rho_E_P_control_dyn : 1

    dCa/dt = -Ca/tau_Ca : Hz
    r_Ca_dyn_smooth : Hz
    dr_Ca_dyn_raw : Hz/second
    dr_Ca_dyn_smooth : Hz/second
    r_PD_control_dyn : 1

    r_Ca2_dyn_smooth : Hz**2
    r_Ca_var_dyn : Hz**2
    r_Ca_var_control_dyn : 1

    w_scale_dyn : 1
    V_rest_dyn : volt
    R_m_dyn : ohm
    g_L_dyn : siemens
    '''

    reset = '''
    V_m = V_reset
    Ca += A_Ca
    '''

    neuron = NeuronGroup(N, eqs, threshold='V_m > V_thresh_spike_detection', reset=reset, refractory=1*ms, method='euler')

    neuron.V_m = V_rest

    neuron.rho_E = rho_E_target 
    neuron.rho_E_smooth = rho_E_target 
    neuron.rho_E_P_control_dyn = 0.0

    neuron.Ca = r_target
    neuron.r_Ca_dyn_smooth = r_target
    neuron.dr_Ca_dyn_raw = 0*Hz/second
    neuron.dr_Ca_dyn_smooth = 0*Hz/second
    neuron.r_PD_control_dyn = 0.0

    neuron.r_Ca2_dyn_smooth = (r_target)**2 + r_Ca_var_target
    neuron.r_Ca_var_dyn = r_Ca_var_target
    neuron.r_Ca_var_control_dyn = 0.0

    neuron.R_m_dyn = 1/g_L
    neuron.g_L_dyn = g_L
    neuron.V_rest_dyn = V_rest
    neuron.w_scale_dyn = 1.0

    exc_syn = Synapses(G_e, neuron, 'w_base : siemens', on_pre='g_e += w_base*w_scale_dyn_post')
    exc_syn.connect()
    exc_syn.w_base = w_e

    inh_syn = Synapses(G_i, neuron, 'w : siemens', on_pre='g_i += w')
    inh_syn.connect()
    inh_syn.w = w_i

    neuron.namespace.update(locals())

    neuron.run_regularly('''
    on = int(t > 1*second)
    on_w = int(t > 1*second)

    E_HK_dyn = E_HK

    R_eff = A_R_m*R_m_dyn
    g_Na = (9*alpha_atp/R_eff)*((V_rest_dyn - V_h)/(9*(1 + alpha_atp)*(V_rest_dyn - V_h) + 12*(V_K - V_rest_dyn) + 8*alpha_atp*(V_Na - V_rest_dyn)))
    g_h = (4/R_eff)*((3*(V_K - V_rest_dyn) + 2*alpha_atp*(V_Na - V_rest_dyn))/(9*(1 + alpha_atp)*(V_rest_dyn - V_h) + 12*(V_K - V_rest_dyn) + 8*alpha_atp*(V_Na - V_rest_dyn)))
    E_RP_dyn = ((g_Na*(V_Na - V_rest_dyn)/3) + (g_h*(V_h - V_rest_dyn)/4))/e_c

    Delta_V = V_peak - V_thresh
    E_AP_dyn = Ca*A_overlap_AP*c_m_area*S_m*Delta_V/(3*e_c)

    I_e_pos = clip(I_syn_e, 0*amp, inf*amp)
    E_ST_dyn = I_e_pos*A_overlap_syn/(3*e_c)

    p_ves = 0.6*exp(-Ca/Hz/5)
    E_glu_dyn = Ca*N_boutons*p_ves*E_glu_rec
    E_Ca_dyn = Ca*N_boutons*p_ves*E_Ca_ex

    E_tot_dyn = E_HK_dyn + E_RP_dyn + E_AP_dyn + E_ST_dyn + E_glu_dyn + E_Ca_dyn
    E_cons_dyn = E_tot_dyn/E_target_ATP

    E_prod_dyn = s_E*rho_E_target 

    rho_E = E_prod_dyn/E_cons_dyn
    rho_E_smooth = rho_E_smooth + dt*(rho_E - rho_E_smooth)/tau_E_smooth
    rho_E_P_control_dyn = rho_E_target  - rho_E_smooth

    dr_Ca_dyn_raw = (Ca - r_Ca_dyn_smooth)/tau_r_smooth
    dr_Ca_dyn_smooth = dr_Ca_dyn_smooth + dt*(dr_Ca_dyn_raw - dr_Ca_dyn_smooth)/tau_dr_smooth
    r_PD_control_dyn = (r_target - r_Ca_dyn_smooth - tau_r_deriv*dr_Ca_dyn_smooth)/Hz
    r_Ca_dyn_smooth = r_Ca_dyn_smooth + dt*dr_Ca_dyn_raw

    r_Ca2_dyn_smooth = r_Ca2_dyn_smooth + dt*(Ca**2 - r_Ca2_dyn_smooth)/tau_r_smooth
    r_Ca_var_dyn = r_Ca2_dyn_smooth - r_Ca_dyn_smooth**2
    r_Ca_var_control_dyn = (r_Ca_var_target - r_Ca_var_dyn)/Hz**2

    V_rest_dyn = V_rest_dyn - on*dt*V_rest_dyn*r_PD_control_dyn/tau_V_rest
    R_m_dyn = R_m_dyn + on*dt*R_m_dyn*r_Ca_var_control_dyn/tau_R_m
    g_L_dyn = 1/R_m_dyn

    w_scale_dyn = w_scale_dyn - on_w*dt*w_scale_dyn*rho_E_P_control_dyn/tau_w_scale
    ''', dt=dt_plasticity)

    spikes = SpikeMonitor(neuron)
    rate_monitor = PopulationRateMonitor(neuron)
    post_states = StateMonitor(neuron, ('V_m', 'I_syn_e', 'I_syn_i', 'g_e', 'g_i', 'Ca', 'rho_E', 'E_prod_dyn', 'E_cons_dyn', 'V_rest_dyn', 'R_m_dyn', 'w_scale_dyn', 'E_tot_dyn', 'E_RP_dyn', 'E_AP_dyn', 'E_ST_dyn', 'E_glu_dyn', 'E_Ca_dyn', 'rho_E_smooth', 'rho_E_P_control_dyn', 'r_Ca_dyn_smooth', 'dr_Ca_dyn_raw', 'dr_Ca_dyn_smooth', 'r_PD_control_dyn', 'r_Ca2_dyn_smooth', 'r_Ca_var_dyn', 'r_Ca_var_control_dyn'), record=True)

    run(T, report='text')

    time = post_states.t / second
    V_m = post_states.V_m[0] / mV
    spike_times_post = spikes.t / ms
    rate_estimate_post = rate_monitor.smooth_rate(window='gaussian', width=rate_window) / Hz
    I_syn_e = post_states.I_syn_e[0] / nA
    I_syn_i = post_states.I_syn_i[0] / nA
    g_e = post_states.g_e[0] / nS
    g_i = post_states.g_i[0] / nS
    E_cons_dyn = post_states.E_cons_dyn[0] / 1
    E_prod_dyn = post_states.E_prod_dyn[0] / 1
    rho_E = post_states.rho_E[0] / 1
    rho_E_smooth = post_states.rho_E_smooth[0] / 1
    rho_E_P_control_dyn = post_states.rho_E_P_control_dyn[0] / 1
    r_Ca = post_states.Ca[0] / Hz
    r_Ca_dyn_smooth = post_states.r_Ca_dyn_smooth[0] / Hz
    dr_Ca_dyn_smooth = post_states.dr_Ca_dyn_smooth[0] / (Hz/second)
    r_PD_control_dyn = post_states.r_PD_control_dyn[0] / 1
    w_syn_e_dyn = post_states.w_scale_dyn[0]*np.mean(w_e)*1e9/siemens
    V_rest_dyn = post_states.V_rest_dyn[0] / mV
    R_m_dyn = post_states.R_m_dyn[0] / Mohm
    r_Ca2_dyn_smooth = post_states.r_Ca2_dyn_smooth[0] / Hz**2
    r_Ca_var_dyn = post_states.r_Ca_var_dyn[0] / Hz**2
    r_Ca_var_control_dyn = post_states.r_Ca_var_control_dyn[0] / 1

    return time, V_m, spike_times_post, rate_estimate_post, I_syn_e, I_syn_i, g_e, g_i, E_cons_dyn, E_prod_dyn, rho_E, rho_E_smooth, rho_E_P_control_dyn, r_Ca, r_Ca_dyn_smooth, dr_Ca_dyn_smooth, r_PD_control_dyn, w_syn_e_dyn, V_rest_dyn, R_m_dyn, r_Ca2_dyn_smooth, r_Ca_var_dyn, r_Ca_var_control_dyn

################################ variance controlled energy-dependent AdExp ################################


def energy_dependent_AdExp_variance(T, N_e, N_i, spike_times_e, spike_times_i, w_e, w_i, C_m, V_rest, g_L, V_thresh, V_reset, Delta_T_ad, tau_w_ad, a_ad, b_ad, rate_window, tau_Ca, A_Ca, E_e, E_i, tau_e, tau_i, membrane_noise, rho_E_target, s_E, E_target_ATP, r_target, r_Ca_var_target, tau_r_smooth, tau_r_deriv, tau_dr_smooth, tau_E_smooth, dt_sim, dt_plasticity, tau_w_scale, tau_V_rest, tau_R_m, E_HK, V_Na, V_K, V_h, alpha_atp, e_c, A_R_m, A_overlap_AP, A_overlap_syn, S_m, c_m_area, V_peak, N_boutons, E_glu_rec, E_Ca_ex):
    # simulation of energy-dependent adaptive exponential integrate-and-fire neuron with variance control
    
    # input
    # T is the simulation duration in ms
    # N_e is the number of excitatory synapses
    # N_i is the number of inhibitory synapses
    # spike_times_e is a dictionary of excitatory spike times {synapse_index: spike_times_ms}
    # spike_times_i is a dictionary of inhibitory spike times {synapse_index: spike_times_ms}
    # w_e is an array of baseline excitatory synaptic weights in nS
    # w_i is an array of baseline inhibitory synaptic weights in nS
    # C_m is the membrane capacitance in pF
    # V_rest is the initial resting potential in mV
    # g_L is the initial leak conductance in nS
    # V_thresh is the spike threshold in mV
    # V_reset is the reset potential in mV
    # Delta_T_ad is the AdExp exponential slope factor in mV
    # tau_w_ad is the AdExp adaptation-current time constant in ms
    # a_ad is the AdExp subthreshold adaptation conductance in nS
    # b_ad is the AdExp spike-triggered adaptation increment in nA
    # rate_window is the rate-estimation smoothing window in ms
    # tau_Ca is the decay time constant of the calcium/activity trace in ms
    # A_Ca is the spike-triggered increment of the calcium/activity trace in Hz
    # E_e is the excitatory reversal potential in mV
    # E_i is the inhibitory reversal potential in mV
    # tau_e is the excitatory synaptic decay time constant in ms
    # tau_i is the inhibitory synaptic decay time constant in ms
    # membrane_noise is the membrane noise amplitude in mV/sqrt(ms)
    # rho_E_target is the normalized energy-balance target
    # s_E is the energy-supply fraction of rho_E_target
    # E_target_ATP is the ATP normalization factor in ATP/s
    # r_target is the target firing rate in Hz
    # r_Ca_var_target is the target calcium-rate variance in Hz^2
    # tau_r_smooth is the time constant for smoothing the calcium-based firing-rate estimate and second moment in ms
    # tau_r_deriv is the derivative-control time constant in ms
    # tau_dr_smooth is the time constant for low-pass filtering the rate derivative in ms
    # tau_E_smooth is the time constant for smoothing energy availability in ms
    # dt_sim is the numerical integration time step in ms
    # dt_plasticity is the plasticity-controller update interval in ms
    # tau_w_scale is the synaptic scaling time constant in ms
    # tau_V_rest is the resting-potential adaptation time constant in ms
    # tau_R_m is the membrane-resistance adaptation time constant in ms
    # E_HK is the housekeeping energy consumption in ATP/s
    # V_Na is the sodium reversal potential in V
    # V_K is the potassium reversal potential in V
    # V_h is the h-channel reversal potential in V
    # alpha_atp is the relative sodium conductance factor
    # e_c is the elementary charge in Coulomb
    # A_R_m is the effective membrane-resistance scaling factor
    # A_overlap_AP is the sodium-potassium current-overlap factor for spikes
    # A_overlap_syn is the sodium-potassium current-overlap factor for synapses
    # S_m is the total membrane surface area in m^2
    # c_m_area is the membrane capacitance per area in F/m^2
    # V_peak is the spike peak voltage in mV
    # N_boutons is the number of synaptic boutons
    # E_glu_rec is the vesicle-recycling energy cost in ATP
    # E_Ca_ex is the calcium-extrusion energy cost in ATP
    
    # output
    # time is the simulation time in s
    # V_m is the membrane voltage trace in mV
    # w_ad is the adaptation current trace in nA
    # spike_times_post are the postsynaptic spike times in ms
    # rate_estimate_post is the postsynaptic firing-rate estimate in Hz
    # I_syn_e is the excitatory synaptic current trace in nA
    # I_syn_i is the inhibitory synaptic current trace in nA
    # g_e is the excitatory synaptic conductance trace in nS
    # g_i is the inhibitory synaptic conductance trace in nS
    # E_cons_dyn is the normalized energy-consumption trace
    # E_prod_dyn is the normalized energy-production trace
    # rho_E is the instantaneous energy balance
    # rho_E_smooth is the smoothed energy balance
    # rho_E_P_control_dyn is the proportional energy-control signal
    # r_Ca is the calcium-based activity trace in Hz
    # r_Ca_dyn_smooth is the smoothed calcium-based firing-rate estimate in Hz
    # dr_Ca_dyn_smooth is the filtered derivative of the smoothed calcium-based firing-rate estimate in Hz/s
    # r_PD_control_dyn is the proportional-derivative firing-rate control signal
    # w_syn_e_dyn is the dynamic mean excitatory synaptic weight in nS
    # V_rest_dyn is the dynamic resting-potential trace in mV
    # R_m_dyn is the dynamic membrane-resistance trace in MOhm
    # r_Ca2_dyn_smooth is the smoothed calcium-rate second moment in Hz^2
    # r_Ca_var_dyn is the calcium-rate variance estimate in Hz^2
    # r_Ca_var_control_dyn is the dimensionless variance-control signal
    

    start_scope()
    defaultclock.dt = dt_sim

    N = 1

    V_T_ad = V_thresh - 2.5 * Delta_T_ad
    V_thresh_spike_detection = V_thresh + 2.5 * Delta_T_ad
    
    # convert spike_times_e and spike_times_i to SpikeGeneratorGroup format
    indices_e, times_e = zip(*[(i, spike) for i, spikes in spike_times_e.items() for spike in spikes])
    indices_i, times_i = zip(*[(i, spike) for i, spikes in spike_times_i.items() for spike in spikes])
    G_e = SpikeGeneratorGroup(N_e, indices_e, times_e)
    G_i = SpikeGeneratorGroup(N_i, indices_i, times_i)
    
    eqs = '''
    dV_m/dt = (g_L_dyn*(V_rest_dyn - V_m) + g_L_dyn*Delta_T_ad*exp((V_m - V_T_ad)/Delta_T_ad) - w_ad + I_syn_e + I_syn_i)/C_m + membrane_noise*xi : volt
    dw_ad/dt = (a_ad*(V_m - V_rest_dyn) - w_ad)/tau_w_ad : amp

    I_syn_e = g_e*(E_e - V_m) : amp
    I_syn_i = g_i*(E_i - V_m) : amp

    dg_e/dt = -g_e/tau_e : siemens
    dg_i/dt = -g_i/tau_i : siemens

    E_HK_dyn : Hz
    E_RP_dyn : Hz
    E_AP_dyn : Hz
    E_ST_dyn : Hz
    E_glu_dyn : Hz
    E_Ca_dyn : Hz
    E_tot_dyn : Hz

    E_cons_dyn : 1
    E_prod_dyn : 1
    rho_E : 1
    rho_E_smooth : 1
    rho_E_P_control_dyn : 1

    dCa/dt = -Ca/tau_Ca : Hz
    r_Ca_dyn_smooth : Hz
    dr_Ca_dyn_raw : Hz/second
    dr_Ca_dyn_smooth : Hz/second
    r_PD_control_dyn : 1

    r_Ca2_dyn_smooth : Hz**2
    r_Ca_var_dyn : Hz**2
    r_Ca_var_control_dyn : 1

    w_scale_dyn : 1
    V_rest_dyn : volt
    R_m_dyn : ohm
    g_L_dyn : siemens
    '''

    reset = '''
    V_m = V_reset
    w_ad += b_ad
    Ca += A_Ca
    '''

    neuron = NeuronGroup(N, eqs, threshold='V_m > V_thresh_spike_detection', reset=reset, refractory=1*ms, method='euler')

    neuron.V_m = V_rest
    neuron.w_ad = 0*nA

    neuron.rho_E = rho_E_target 
    neuron.rho_E_smooth = rho_E_target 
    neuron.rho_E_P_control_dyn = 0.0

    neuron.Ca = r_target
    neuron.r_Ca_dyn_smooth = r_target
    neuron.dr_Ca_dyn_raw = 0*Hz/second
    neuron.dr_Ca_dyn_smooth = 0*Hz/second
    neuron.r_PD_control_dyn = 0.0

    neuron.r_Ca2_dyn_smooth = (r_target)**2 + r_Ca_var_target
    neuron.r_Ca_var_dyn = r_Ca_var_target
    neuron.r_Ca_var_control_dyn = 0.0

    neuron.R_m_dyn = 1/g_L
    neuron.g_L_dyn = g_L
    neuron.V_rest_dyn = V_rest
    neuron.w_scale_dyn = 1.0

    exc_syn = Synapses(G_e, neuron, 'w_base : siemens', on_pre='g_e += w_base*w_scale_dyn_post')
    exc_syn.connect()
    exc_syn.w_base = w_e

    inh_syn = Synapses(G_i, neuron, 'w : siemens', on_pre='g_i += w')
    inh_syn.connect()
    inh_syn.w = w_i

    neuron.namespace.update(locals())

    neuron.run_regularly('''
    on = int(t > 1*second)
    on_w = int(t > 1*second)

    E_HK_dyn = E_HK

    R_eff = A_R_m*R_m_dyn
    g_Na = (9*alpha_atp/R_eff)*((V_rest_dyn - V_h)/(9*(1 + alpha_atp)*(V_rest_dyn - V_h) + 12*(V_K - V_rest_dyn) + 8*alpha_atp*(V_Na - V_rest_dyn)))
    g_h = (4/R_eff)*((3*(V_K - V_rest_dyn) + 2*alpha_atp*(V_Na - V_rest_dyn))/(9*(1 + alpha_atp)*(V_rest_dyn - V_h) + 12*(V_K - V_rest_dyn) + 8*alpha_atp*(V_Na - V_rest_dyn)))
    E_RP_dyn = ((g_Na*(V_Na - V_rest_dyn)/3) + (g_h*(V_h - V_rest_dyn)/4))/e_c

    Delta_V = V_peak - V_thresh
    E_AP_dyn = Ca*A_overlap_AP*c_m_area*S_m*Delta_V/(3*e_c)

    I_e_pos = clip(I_syn_e, 0*amp, inf*amp)
    E_ST_dyn = I_e_pos*A_overlap_syn/(3*e_c)

    p_ves = 0.6*exp(-Ca/Hz/5)
    E_glu_dyn = Ca*N_boutons*p_ves*E_glu_rec
    E_Ca_dyn = Ca*N_boutons*p_ves*E_Ca_ex

    E_tot_dyn = E_HK_dyn + E_RP_dyn + E_AP_dyn + E_ST_dyn + E_glu_dyn + E_Ca_dyn
    E_cons_dyn = E_tot_dyn/E_target_ATP

    E_prod_dyn = s_E*rho_E_target 

    rho_E = E_prod_dyn/E_cons_dyn
    rho_E_smooth = rho_E_smooth + dt*(rho_E - rho_E_smooth)/tau_E_smooth
    rho_E_P_control_dyn = rho_E_target  - rho_E_smooth

    dr_Ca_dyn_raw = (Ca - r_Ca_dyn_smooth)/tau_r_smooth
    dr_Ca_dyn_smooth = dr_Ca_dyn_smooth + dt*(dr_Ca_dyn_raw - dr_Ca_dyn_smooth)/tau_dr_smooth
    r_PD_control_dyn = (r_target - r_Ca_dyn_smooth - tau_r_deriv*dr_Ca_dyn_smooth)/Hz
    r_Ca_dyn_smooth = r_Ca_dyn_smooth + dt*dr_Ca_dyn_raw

    r_Ca2_dyn_smooth = r_Ca2_dyn_smooth + dt*(Ca**2 - r_Ca2_dyn_smooth)/tau_r_smooth
    r_Ca_var_dyn = r_Ca2_dyn_smooth - r_Ca_dyn_smooth**2
    r_Ca_var_control_dyn = (r_Ca_var_target - r_Ca_var_dyn)/Hz**2

    V_rest_dyn = V_rest_dyn - on*dt*V_rest_dyn*r_PD_control_dyn/tau_V_rest
    R_m_dyn = R_m_dyn + on*dt*R_m_dyn*r_Ca_var_control_dyn/tau_R_m
    g_L_dyn = 1/R_m_dyn

    w_scale_dyn = w_scale_dyn - on_w*dt*w_scale_dyn*rho_E_P_control_dyn/tau_w_scale
    ''', dt=dt_plasticity)

    spikes = SpikeMonitor(neuron)
    rate_monitor = PopulationRateMonitor(neuron)
    post_states = StateMonitor(neuron, ('V_m', 'w_ad', 'I_syn_e', 'I_syn_i', 'g_e', 'g_i', 'Ca', 'rho_E', 'E_prod_dyn', 'E_cons_dyn', 'V_rest_dyn', 'R_m_dyn', 'w_scale_dyn', 'E_tot_dyn', 'E_RP_dyn', 'E_AP_dyn', 'E_ST_dyn', 'E_glu_dyn', 'E_Ca_dyn', 'rho_E_smooth', 'rho_E_P_control_dyn', 'r_Ca_dyn_smooth', 'dr_Ca_dyn_raw', 'dr_Ca_dyn_smooth', 'r_PD_control_dyn', 'r_Ca2_dyn_smooth', 'r_Ca_var_dyn', 'r_Ca_var_control_dyn'), record=True)

    run(T, report='text')

    time = post_states.t / second
    V_m = post_states.V_m[0] / mV
    w_ad = post_states.w_ad[0] / nA
    spike_times_post = spikes.t / ms
    rate_estimate_post = rate_monitor.smooth_rate(window='gaussian', width=rate_window) / Hz
    I_syn_e = post_states.I_syn_e[0] / nA
    I_syn_i = post_states.I_syn_i[0] / nA
    g_e = post_states.g_e[0] / nS
    g_i = post_states.g_i[0] / nS
    E_cons_dyn = post_states.E_cons_dyn[0] / 1
    E_prod_dyn = post_states.E_prod_dyn[0] / 1
    rho_E = post_states.rho_E[0] / 1
    rho_E_smooth = post_states.rho_E_smooth[0] / 1
    rho_E_P_control_dyn = post_states.rho_E_P_control_dyn[0] / 1
    r_Ca = post_states.Ca[0] / Hz
    r_Ca_dyn_smooth = post_states.r_Ca_dyn_smooth[0] / Hz
    dr_Ca_dyn_smooth = post_states.dr_Ca_dyn_smooth[0] / (Hz/second)
    r_PD_control_dyn = post_states.r_PD_control_dyn[0] / 1
    w_syn_e_dyn = post_states.w_scale_dyn[0]*np.mean(w_e)*1e9/siemens
    V_rest_dyn = post_states.V_rest_dyn[0] / mV
    R_m_dyn = post_states.R_m_dyn[0] / Mohm
    r_Ca2_dyn_smooth = post_states.r_Ca2_dyn_smooth[0] / Hz**2
    r_Ca_var_dyn = post_states.r_Ca_var_dyn[0] / Hz**2
    r_Ca_var_control_dyn = post_states.r_Ca_var_control_dyn[0] / 1

    return time, V_m, w_ad, spike_times_post, rate_estimate_post, I_syn_e, I_syn_i, g_e, g_i, E_cons_dyn, E_prod_dyn, rho_E, rho_E_smooth, rho_E_P_control_dyn, r_Ca, r_Ca_dyn_smooth, dr_Ca_dyn_smooth, r_PD_control_dyn, w_syn_e_dyn, V_rest_dyn, R_m_dyn, r_Ca2_dyn_smooth, r_Ca_var_dyn, r_Ca_var_control_dyn

################################ downsampling functions for saving ################################

def downsample_mean(x, downsample_factor, axis=0):
    # downsample an array by averaging over non-overlapping bins of length downsample_factor
    # input
    # x is an array
    # downsample_factor is the number of samples averaged into one output sample
    # axis is the axis along which the array is downsampled
    # output
    # x_downsampled is the mean-downsampled array
    
    if x is None:
        return None
    
    if downsample_factor is None or downsample_factor <= 1:
        return x
    
    x = np.asarray(x)
    downsample_factor = int(downsample_factor)
    
    if x.ndim == 0:
        return x.item()
    
    n_samples = x.shape[axis]
    n_keep = (n_samples // downsample_factor) * downsample_factor
    
    if n_keep == 0:
        return x
    
    slicer = [slice(None)] * x.ndim
    slicer[axis] = slice(0, n_keep)
    x_trimmed = x[tuple(slicer)]
    
    x_moved = np.moveaxis(x_trimmed, axis, 0)
    x_reshaped = x_moved.reshape(n_keep // downsample_factor, downsample_factor, *x_moved.shape[1:])
    x_downsampled = x_reshaped.mean(axis=1)
    x_downsampled = np.moveaxis(x_downsampled, 0, axis)
    
    return x_downsampled

def downsample_energy_data_dict(data, downsample_factor, time_key="time", keep_keys=("spike_times_post",)):
    # downsample all time-dependent arrays in an energy-simulation data dictionary
    # input
    # data is a dictionary containing simulation traces and metadata
    # downsample_factor is the number of samples averaged into one output sample
    # time_key is the dictionary key used to identify the time vector
    # keep_keys are keys that should not be downsampled
    # output
    # data_downsampled is a dictionary with downsampled traces and unchanged metadata
    
    data_downsampled = {}
    time_len = len(np.asarray(data[time_key])) if time_key in data else None
    
    for key, value in data.items():
        if key in keep_keys:
            data_downsampled[key] = value
        elif np.isscalar(value):
            data_downsampled[key] = value
        else:
            value_array = np.asarray(value)
            if time_len is not None and value_array.ndim > 0 and value_array.shape[0] == time_len:
                data_downsampled[key] = downsample_mean(value_array, downsample_factor)
            else:
                data_downsampled[key] = value
    
    if "r_Ca_var_target" in data_downsampled and "r_var_target" not in data_downsampled:
        data_downsampled["r_var_target"] = data_downsampled["r_Ca_var_target"]
    
    return data_downsampled

################################ analysis functions ################################

def calculate_ISI_var_CV(spike_times, T_min=500000):
    # calculates ISIs, their variance and CV_ISI after a minimum spike time
    # input
    # spike_times is an array of spike times
    # T_min is the minimum spike time in ms
    
    # output
    # spike_times_late is an array of spike times after t_min in ms
    # ISIs is an array of inter-spike intervals in ms
    # mean_ISI is the mean of ISIs
    # std_ISI is the standard deviation of ISIs
    # var_ISI is the variance of ISIs
    # CV_ISI is the coefficient of variation of ISIs
    
    spike_times = np.asarray(spike_times)
    spike_times_late = spike_times[spike_times > T_min]
    ISIs = np.diff(spike_times_late)
    
    if len(ISIs) > 1 and np.mean(ISIs) > 0:
        mean_ISI = np.mean(ISIs)
        std_ISI = np.std(ISIs)
        var_ISI = np.var(ISIs)
        CV_ISI = std_ISI / mean_ISI
    else:
        mean_ISI, std_ISI, var_ISI, CV_ISI = np.nan, np.nan, np.nan, np.nan
    
    return spike_times_late, ISIs, mean_ISI, std_ISI, var_ISI, CV_ISI

def calculate_ISI_var_CV_chunks(spike_times, T_min=500000, T_max=None, n_chunks=10):
    # calculates ISI statistics in equally long time chunks and averages them
    # input
    # spike_times is an array of spike times in ms
    # T_min is the minimum analyzed time in ms
    # T_max is the maximum analyzed time in ms; if None, the last spike time is used
    # n_chunks is the number of equally long time chunks
    #
    # output
    # spike_times_late is an array of spike times within the analyzed interval in ms
    # ISIs_chunks is a list containing the ISIs of each chunk in ms
    # mean_ISI is the mean of the chunk-wise mean ISIs in ms
    # std_ISI is the mean of the chunk-wise standard deviations in ms
    # var_ISI is the mean of the chunk-wise variances in ms^2
    # CV_ISI is the mean of the chunk-wise CV_ISI values

    spike_times = np.sort(np.asarray(spike_times, dtype=float))

    if n_chunks < 1:
        raise ValueError("n_chunks must be at least 1.")

    if T_max is None:
        if len(spike_times) == 0:
            return spike_times, [], np.nan, np.nan, np.nan, np.nan
        T_max = spike_times[-1]

    if T_max <= T_min:
        raise ValueError("T_max must be larger than T_min.")

    spike_times_late = spike_times[(spike_times > T_min) & (spike_times <= T_max)]
    chunk_edges = np.linspace(T_min, T_max, n_chunks + 1)

    ISIs_chunks = []
    mean_ISI_chunks = []
    std_ISI_chunks = []
    var_ISI_chunks = []
    CV_ISI_chunks = []

    for i in range(n_chunks):
        chunk_start = chunk_edges[i]
        chunk_end = chunk_edges[i + 1]

        if i == n_chunks - 1:
            mask = (spike_times_late >= chunk_start) & (spike_times_late <= chunk_end)
        else:
            mask = (spike_times_late >= chunk_start) & (spike_times_late < chunk_end)

        spike_times_chunk = spike_times_late[mask]
        ISIs = np.diff(spike_times_chunk)
        ISIs_chunks.append(ISIs)

        if len(ISIs) > 1 and np.mean(ISIs) > 0:
            mean_ISI_chunk = np.mean(ISIs)
            std_ISI_chunk = np.std(ISIs)
            var_ISI_chunk = np.var(ISIs)
            CV_ISI_chunk = std_ISI_chunk / mean_ISI_chunk
        else:
            mean_ISI_chunk = np.nan
            std_ISI_chunk = np.nan
            var_ISI_chunk = np.nan
            CV_ISI_chunk = np.nan

        mean_ISI_chunks.append(mean_ISI_chunk)
        std_ISI_chunks.append(std_ISI_chunk)
        var_ISI_chunks.append(var_ISI_chunk)
        CV_ISI_chunks.append(CV_ISI_chunk)

    mean_ISI = np.nanmean(mean_ISI_chunks) if np.any(np.isfinite(mean_ISI_chunks)) else np.nan
    std_ISI = np.nanmean(std_ISI_chunks) if np.any(np.isfinite(std_ISI_chunks)) else np.nan
    var_ISI = np.nanmean(var_ISI_chunks) if np.any(np.isfinite(var_ISI_chunks)) else np.nan
    CV_ISI = np.nanmean(CV_ISI_chunks) if np.any(np.isfinite(CV_ISI_chunks)) else np.nan

    return spike_times_late, ISIs_chunks, mean_ISI, std_ISI, var_ISI, CV_ISI

################################ trajectory simulation functions ################################

def get_FR_trajectory_values_at_timepoints(time_dyn_FR, w_syn_e_dyn_FR, V_rest_dyn_FR, R_m_dyn_FR, t_end=400, n_timepoints=9, w_scale_target=None):
    # extracts nearest values of FR trajectory at evenly spaced timepoints from 0 to end_s
    # input
    # time_dyn_FR is the time vector in seconds
    # w_syn_e_dyn_FR is the dynamic excitatory synaptic weight array
    # V_rest_dyn_FR is the dynamic resting potential array in mV
    # R_m_dyn_FR_PD is the dynamic membrane resistance array in MOhm
    # t_end is the final time in seconds
    # n_timepoints is the number of timepoints between 0 and t_end
    # w_scale_target is the target value for the first extracted w_syn_e_dyn_FR value
    # output
    # values_every_timepoint is a list of dictionaries with time, w_syn_e, V_rest, and R_m values

    target_times_s = np.linspace(0, t_end, n_timepoints)

    values_every_timepoint = []
    time_s = np.asarray(time_dyn_FR)

    idx_first = np.argmin(np.abs(time_s - target_times_s[0]))
    w_syn_e_first = w_syn_e_dyn_FR[idx_first]

    if w_scale_target is None:
        w_scale_factor = 1.0
    else:
        w_scale_factor = w_scale_target / w_syn_e_first

    for t_target in target_times_s:
        idx = np.argmin(np.abs(time_s - t_target))

        values_every_timepoint.append({
            "timepoint_FR": float(time_dyn_FR[idx]),
            "w_syn_e_dyn_FR": w_syn_e_dyn_FR[idx] * w_scale_factor,
            "V_rest_dyn_FR": float(V_rest_dyn_FR[idx]),
            "R_m_dyn_FR": float(R_m_dyn_FR[idx])})
    
    return values_every_timepoint

def mean_FR_trajectory_results_over_runs(results_single_runs_FR, group_key="trajectory_index"):
    # averages all result quantities across single runs for each FR trajectory timepoint
    # input
    # results_single_runs_FR is a dictionary containing all FR trajectory simulation results from multiple single runs
    # group_key is the key used to group trajectory timepoints, usually "trajectory_index"
    # output
    # results_mean_FR is a dictionary containing mean values for each trajectory timepoint

    results_mean_FR = {}

    trajectory_indices = np.unique(results_single_runs_FR[group_key])

    for trajectory_index in trajectory_indices:

        idx_group = np.where(np.asarray(results_single_runs_FR[group_key]) == trajectory_index)[0]

        for key, values in results_single_runs_FR.items():

            if "spike" in key.lower():
                continue

            values_group = [values[idx] for idx in idx_group]

            if key in ["single_run_index"]:
                continue

            if key in ["trajectory_index", "timepoint_FR", "w_syn_e_dyn_FR", "V_rest_dyn_FR", "R_m_dyn_FR"]:
                results_mean_FR.setdefault(key, []).append(values_group[0])

            else:
                try:
                    results_mean_FR.setdefault(key, []).append(np.mean(values_group, axis=0))
                except:
                    pass

    return results_mean_FR


