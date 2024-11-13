import matplotlib.pyplot as plt
import pandas as pd
from pandas import ExcelWriter
from pandas import ExcelFile
from pydoc import help
from scipy.stats.stats import pearsonr
import pathlib
import scipy.io as sio
from pathlib import Path
import numpy as np
import mne
import os
import os.path as op
 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import date
import time
import os
import mne
import nice 
from nice.markers import (KolmogorovComplexity, PowerSpectralDensityEstimator, 
                          PowerSpectralDensity, SymbolicMutualInformation, PermutationEntropy, PowerSpectralDensitySummary)
 
from sklearn.metrics import roc_auc_score
 
import scipy


mne.viz.set_browser_backend('matplotlib')

column_names = ['komplexity', 'theta_pe', 'alpha_pe', 'beta_pe', 'gamma_pe', 'theta_wSMI', 
                'alpha_wSMI', 'beta_wSMI', 'gamma_wSMI', 'delta', 'delta_n', 'theta',
                'theta_n', 'alpha', 'alpha_n', 'beta', 'beta_n', 'gamma', 'gamma_n', 'se',
                'deltase', 'thetase','alphase', 'betase', 'gammase', 'deltamsf', 'thetamsf',
                'alphamsf', 'betamsf', 'gammamsf', 'deltasef90', 'thetasef90', 'alphasef90',
                 'betasef90', 'gammasef90', 'deltasef95', 'thetasef95', 'alphasef95',
                 'betasef95', 'gammasef95']

df = pd.DataFrame(columns = column_names)

def all_markers(epochs, tmin=None, tmax=None):
    psds_params = dict(n_fft=4096, n_overlap=100, n_jobs='auto', nperseg=128)

    base_psd = PowerSpectralDensityEstimator(

            psd_method='welch', tmin=None, tmax=None, fmin=1., fmax=45.,


            psd_params=psds_params, comment='default')


    base_psd = PowerSpectralDensityEstimator(

            psd_method='welch', tmin=None, tmax=None, fmin=1., fmax=45.,

            psd_params=psds_params, comment='default')

    reduction_func = [

            {'axis': 'epochs', 'function': np.mean},

            {'axis': 'channels', 'function': np.mean},

            {'axis': 'frequency', 'function': np.sum}]
    
    delta = PowerSpectralDensity(estimator=base_psd, fmin=1., fmax=4, normalize=False, comment='delta')
    delta.fit(epochs)
    datadelta = delta._reduce_to(reduction_func, target='channels', picks=None)

    delta_n = PowerSpectralDensity(estimator=base_psd, fmin=1., fmax=4.,normalize=True, comment='deltan')
    delta_n.fit(epochs)
    datadelta_n = delta_n._reduce_to(reduction_func, target='channels', picks=None)

    alpha_n = PowerSpectralDensity(estimator=base_psd, fmin=8., fmax=13.,normalize=True, comment='alphan')
    alpha_n.fit(epochs)
    dataalpha_n = alpha_n._reduce_to(reduction_func, target='channels', picks=None)

    alpha = PowerSpectralDensity(estimator=base_psd, fmin=8., fmax=13.,normalize=False, comment='alpha')
    alpha.fit(epochs)
    dataalpha = alpha._reduce_to(reduction_func, target='channels', picks=None)

    beta = PowerSpectralDensity(estimator=base_psd, fmin=13., fmax=30,normalize=False, comment='beta')
    beta.fit(epochs)
    databeta = beta._reduce_to(reduction_func, target='channels', picks=None)

    beta_n = PowerSpectralDensity(estimator=base_psd, fmin=13., fmax=30,normalize=True, comment='beta_n')
    beta_n.fit(epochs)
    databeta_n = beta_n._reduce_to(reduction_func, target='channels', picks=None)

    theta_n = PowerSpectralDensity(estimator=base_psd, fmin=4., fmax=8., normalize=True, comment='thetan')
    theta_n.fit(epochs)
    datatheta_n = theta_n._reduce_to(reduction_func, target='channels', picks=None)
    
    theta = PowerSpectralDensity(estimator=base_psd, fmin=4., fmax=8,normalize=False, comment='theta')
    theta.fit(epochs)
    datatheta = theta._reduce_to(reduction_func, target='channels', picks=None)

    gamma_n = PowerSpectralDensity(estimator=base_psd, fmin=30., fmax=45.,normalize=True, comment='gamman')
    gamma_n.fit(epochs)
    datagamma_n = gamma_n._reduce_to(reduction_func, target='channels', picks=None)
    
    gamma = PowerSpectralDensity(estimator=base_psd, fmin=30., fmax=45,normalize=False, comment='gamma')
    gamma.fit(epochs)
    datagamma = gamma._reduce_to(reduction_func, target='channels', picks=None)

    se = PowerSpectralDensity(estimator=base_psd, fmin=1., fmax=45.,
                         normalize=False, comment='summary_se')
    se.fit(epochs)
    datase = se._reduce_to(reduction_func, target='channels', picks=None)  

    reduction_func = [

            {'axis': 'epochs', 'function': np.mean},

            {'axis': 'channels', 'function': np.std},

            {'axis': 'frequency', 'function': np.sum}]
    
    deltase = PowerSpectralDensity(estimator=base_psd, fmin=1., fmax=4.,
                         normalize=False, comment='summary_se')
    deltase.fit(epochs)
    datadeltase = deltase._reduce_to(reduction_func, target='channels', picks=None)    

    thetase = PowerSpectralDensity(estimator=base_psd, fmin=4., fmax=8.,
                         normalize=False, comment='summary_se')
    thetase.fit(epochs)
    datathetase = thetase._reduce_to(reduction_func, target='channels', picks=None)    

    alphase = PowerSpectralDensity(estimator=base_psd, fmin=8., fmax=12.,
                         normalize=False, comment='summary_se')
    alphase.fit(epochs)
    dataalphase = alphase._reduce_to(reduction_func, target='channels', picks=None)    

    betase = PowerSpectralDensity(estimator=base_psd, fmin=12., fmax=30.,
                         normalize=False, comment='summary_se')
    betase.fit(epochs)
    databetase = betase._reduce_to(reduction_func, target='channels', picks=None)    

    gammase = PowerSpectralDensity(estimator=base_psd, fmin=30., fmax=45.,
                         normalize=False, comment='summary_se')
    gammase.fit(epochs)
    datagammase = gammase._reduce_to(reduction_func, target='channels', picks=None)    

    reduction_func_summary = [
        {'axis': 'epochs', 'function':np.mean},
        {'axis': 'channels', 'function': np.std}]
    
    deltamsf = PowerSpectralDensitySummary(estimator=base_psd, fmin=1., fmax=4., percentile=.5, comment='summary_msf')
    deltamsf.fit(epochs)
    datadeltamsf = deltamsf._reduce_to(reduction_func_summary, target='channels', picks=None)

    thetamsf = PowerSpectralDensitySummary(estimator=base_psd, fmin=4., fmax=8., percentile=.5, comment='summary_msf')
    thetamsf.fit(epochs)
    datathetamsf = thetamsf._reduce_to(reduction_func_summary, target='channels', picks=None)

    alphamsf = PowerSpectralDensitySummary(estimator=base_psd, fmin=8., fmax=12., percentile=.5, comment='summary_msf')
    alphamsf.fit(epochs)
    dataalphamsf = alphamsf._reduce_to(reduction_func_summary, target='channels', picks=None)

    betamsf = PowerSpectralDensitySummary(estimator=base_psd, fmin=12., fmax=30., percentile=.5, comment='summary_msf')
    betamsf.fit(epochs)
    databetamsf = betamsf._reduce_to(reduction_func_summary, target='channels', picks=None)

    gammamsf = PowerSpectralDensitySummary(estimator=base_psd, fmin=30., fmax=45., percentile=.5, comment='summary_msf')
    gammamsf.fit(epochs)
    datagammamsf = gammamsf._reduce_to(reduction_func_summary, target='channels', picks=None)

    deltasef90 = PowerSpectralDensitySummary(estimator=base_psd, fmin=1., fmax=4., percentile=.9, comment='summary_sef90')
    deltasef90.fit(epochs)
    datadeltasef90 = deltasef90._reduce_to(reduction_func_summary, target='channels', picks=None)

    thetasef90 = PowerSpectralDensitySummary(estimator=base_psd, fmin=4., fmax=8., percentile=.9, comment='summary_sef90')
    thetasef90.fit(epochs)
    datathetasef90 = thetasef90._reduce_to(reduction_func_summary, target='channels', picks=None)

    alphasef90 = PowerSpectralDensitySummary(estimator=base_psd, fmin=8., fmax=12., percentile=.9, comment='summary_sef90')
    alphasef90.fit(epochs)
    dataalphasef90 = alphasef90._reduce_to(reduction_func_summary, target='channels', picks=None)

    betasef90 = PowerSpectralDensitySummary(estimator=base_psd, fmin=12., fmax=30., percentile=.9, comment='summary_sef90')
    betasef90.fit(epochs)
    databetasef90 = betasef90._reduce_to(reduction_func_summary, target='channels', picks=None)

    gammasef90 = PowerSpectralDensitySummary(estimator=base_psd, fmin=30., fmax=45., percentile=.9, comment='summary_sef90')
    gammasef90.fit(epochs)
    datagammasef90 = gammasef90._reduce_to(reduction_func_summary, target='channels', picks=None)

    deltasef95 = PowerSpectralDensitySummary(estimator=base_psd, fmin=1., fmax=4., percentile=.95, comment='summary_sef95')
    deltasef95.fit(epochs)
    datadeltasef95 = deltasef95._reduce_to(reduction_func_summary, target='channels', picks=None)

    thetasef95 = PowerSpectralDensitySummary(estimator=base_psd, fmin=4., fmax=8., percentile=.95, comment='summary_sef95')
    thetasef95.fit(epochs)
    datathetasef95 = thetasef95._reduce_to(reduction_func_summary, target='channels', picks=None)

    alphasef95 = PowerSpectralDensitySummary(estimator=base_psd, fmin=8., fmax=12., percentile=.95, comment='summary_sef95')
    alphasef95.fit(epochs)
    dataalphasef95 = alphasef95._reduce_to(reduction_func_summary, target='channels', picks=None)

    betasef95 = PowerSpectralDensitySummary(estimator=base_psd, fmin=12., fmax=30., percentile=.95, comment='summary_sef95')
    betasef95.fit(epochs)
    databetasef95 = betasef95._reduce_to(reduction_func_summary, target='channels', picks=None)

    gammasef95 = PowerSpectralDensitySummary(estimator=base_psd, fmin=30., fmax=45., percentile=.95, comment='summary_sef95')
    gammasef95.fit(epochs)
    datagammasef95 = gammasef95._reduce_to(reduction_func_summary, target='channels', picks=None)

    komplexity = KolmogorovComplexity(tmin=None, tmax=None, backend='python')
    komplexity.fit(epochs)

    reduction_func_info = [

        {'axis': 'epochs', 'function': np.mean},

        {'axis': 'channels', 'function': np.mean}]

    datakomplexity = komplexity._reduce_to(reduction_func_info, target='channels', picks=None)

    theta_pe = PermutationEntropy(tmin=None, tmax=None, tau = 8)
    theta_pe.fit(epochs)
    data_theta_pe = theta_pe._reduce_to(reduction_func_info, target='channels', picks=None)

    alpha_pe = PermutationEntropy(tmin=None, tmax=None, tau = 4)
    alpha_pe.fit(epochs)
    data_alpha_pe = alpha_pe._reduce_to(reduction_func_info, target='channels', picks=None)

    beta_pe = PermutationEntropy(tmin=None, tmax=None, tau = 2)
    beta_pe.fit(epochs)
    data_beta_pe = beta_pe._reduce_to(reduction_func_info, target='channels', picks=None)

    gamma_pe = PermutationEntropy(tmin=None, tmax=None, tau = 1)
    gamma_pe.fit(epochs)
    data_gamma_pe = gamma_pe._reduce_to(reduction_func_info, target='channels', picks=None)

    reduction_func_mutual = [

        {'axis': 'epochs', 'function': np.mean},

        {'axis': 'channels', 'function': np.mean},

        {'axis': 'channels_y','function':np.mean}]


    theta_wSMI = SymbolicMutualInformation(tmin=None, tmax=None, kernel=3, tau=8, 
                                        backend="python",
                    method_params={'nthreads': 'auto', 'bypass_csd': True}, method='weighted', comment='default')
    theta_wSMI.fit(epochs)
    data_theta_wSMI = theta_wSMI._reduce_to(reduction_func_mutual, target='channels', picks=None)

    alpha_wSMI = SymbolicMutualInformation(tmin=None, tmax=None, kernel=3, tau=4, 
                                     backend="python",
                  method_params={'nthreads': 'auto', 'bypass_csd': True}, method='weighted', comment='default')
    alpha_wSMI.fit(epochs)
    data_alpha_wSMI = alpha_wSMI._reduce_to(reduction_func_mutual, target='channels', picks=None)
    
    beta_wSMI = SymbolicMutualInformation(tmin=None, tmax=None, kernel=3, tau=2, 
                                     backend="python",
                  method_params={'nthreads': 'auto', 'bypass_csd': True}, method='weighted', comment='default')
    beta_wSMI.fit(epochs)
    data_beta_wSMI = beta_wSMI._reduce_to(reduction_func_mutual, target='channels', picks=None)

    gamma_wSMI = SymbolicMutualInformation(tmin=None, tmax=None, kernel=3, tau=1, 
                                     backend="python",
                  method_params={'nthreads': 'auto', 'bypass_csd': True}, method='weighted', comment='default')
    gamma_wSMI.fit(epochs)
    data_gamma_wSMI = gamma_wSMI._reduce_to(reduction_func_mutual, target='channels', picks=None)

    df1 = pd.DataFrame([[np.mean(datakomplexity), np.mean(data_theta_pe), 
                        np.mean(data_alpha_pe), np.mean(data_beta_pe), 
                        np.mean(data_gamma_pe), np.mean(data_theta_wSMI), 
                        np.mean(data_alpha_wSMI), np.mean(data_beta_wSMI),
                        np.mean(data_gamma_wSMI), np.mean(datadelta), 
                        np.mean(datadelta_n), np.mean(datatheta), 
                        np.mean(datatheta_n), np.mean(dataalpha), 
                        np.mean(dataalpha_n), np.mean(databeta), 
                        np.mean(databeta_n), np.mean(datagamma), 
                        np.mean(datagamma_n), np.mean(datase),np.mean(datadeltase), np.mean(datathetase), np.mean(dataalphase),
                         np.mean(databetase), np.mean(datagammase), np.mean(datadeltamsf),
                         np.mean(datathetamsf), np.mean(dataalphamsf), np.mean(databetamsf),
                         np.mean(datagammamsf), np.mean(datadeltasef90), np.mean(datathetasef90),
                         np.mean(dataalphasef90), np.mean(databetasef90), np.mean(datagammasef90),
                         np.mean(datadeltasef95), np.mean(datathetasef95), np.mean(dataalphasef95),
                         np.mean(databetasef95), np.mean(datagammasef95)]], columns = column_names)
    
    return df1

def append_to_excel(file_path, data):
    try:
        df = pd.read_excel(file_path)
    except FileNotFoundError:
        df = pd.DataFrame(columns=column_names)
    df = pd.concat([df, data], ignore_index=True)
    df.to_excel(file_path, index=False)

folder_path = "/Users/samuelborromeo/Downloads/Research/sharma project/EEG/concats"

for fname in os.listdir(folder_path):
    fpath = os.path.join(folder_path, fname)
    if os.path.isfile(fpath):
        epo = Path(fpath)
        epochs = mne.read_epochs(epo)
        results = all_markers(epochs, tmin=None, tmax=None)
        append_to_excel('freq_spec_sums.xlsx', results)

