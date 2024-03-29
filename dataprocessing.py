# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 14:51:31 2015

@author: jchen3
"""
from __future__ import division, print_function
import numpy as np
import pandas as pd
import scipy.fftpack as sfft
from scipy.signal import hanning
from scipy import interpolate
import subprocess,csv,os
import scipy.signal as signal
from matplotlib import pyplot as plt
import gc


def sqrtNewTon(a, iters = 6):
    '''
    This function is only for learning Newton method to solve continuous nonlinear
    curve functions 
    x_next = x - f(x)/f'(x)  Init value of x should never be 0
    Parameters
    ----------
    a : Int
        Non negtive number.
    iters : Int, optional
        Iterations for newton method. The default is 6.

    Returns
    -------
    x : float
        non negtive Square root of a.
    '''
    #Init value never be 0 in case dydz would be 0, which cause divide by 0 error
    x = 1
    if a == 0:        return 0
    elif a < 0:    raise ValueError("a should no less than 0")
    for _ in range(int(iters)):
        y = x**2 - a
        dydx = 2*x
        x_next = x - y/dydx
        x = x_next
    return x
        
def phaseNoise2JitterPs(phaseNoiseDic,f):
    def log2linear_rms(A,f):
        return ((2*(10**(A/10)))**.5)/(2*np.pi*f)
    l = len(phaseNoiseDic)
    phaseNoiseDic = dict(sorted(phaseNoiseDic.items(),key = lambda x:x[0]))
    f_s,pn_s = list(phaseNoiseDic.keys()), [10**(i/10) for i in phaseNoiseDic.values()]
    A = 0
    for i in range(l-1):        A += (pn_s[i]+pn_s[i+1])*.5 * (f_s[i+1]-f_s[i])
    return ((2*A)**.5)/(2*np.pi*f)*1e12
    #A = 10*np.log10(A)
    #return ((2*(10**(A/10)))**.5)/(2*np.pi*f)*1e12

def ber(t,r):
    t = np.array(t).reshape((-1,1)).astype(int)
    r = np.array(r).reshape((-1,1)).astype(int)
    length = min(t.shape[0],r.shape[0])
    err = np.abs(np.sign(t[:length]-r[:length]))
    errcount = np.sum(err)
    return errcount,  errcount /  length
    
def data_normalize(data_ins,bit,signed = 1):

    if signed == 1:
        dataouts = []
        for data in data_ins:
            data = int(data)
            if data & (1<< bit-1 ) :
                data = (~(data & ( (2**(bit-1))-1 ))+1)&((2**(bit-1))-1)
                data = -data
            data/=(2.**(bit-1)-1)
            dataouts.append(data)
    else:
        dataouts = map(lambda x: float(x)/(2.0**bit-1),data_ins)
        
    return dataouts
        
def flatness_compensation(X,y,degree = 5,show = False):
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.metrics import mean_squared_error

    regr = LinearRegression()
    regrs = []
    scores = []
    for i in range(1,degree+1):
        if i == 1:         X_order = X
        else:
            order = PolynomialFeatures(degree = i)
            X_order = order.fit_transform(X)

        #五次多项式拟合
        regrs.append( regr.fit(X_order, y) )
        y_predict = regrs[-1].predict(X_order)
        scores.append( mean_squared_error(y, y_predict) )
        
    score = min(scores)
    regr = regrs[scores.index(score)]

    if show == True:
        print( 'Order = {}, MSE = {}, Coef = {}, Intecept = {}'.format(scores.index(score)+1,scores,regr.coef_, regr.intercept_) )
        plt.plot(X,y,'g')
        plt.plot(X,y_predict,'b')
        plt.show()
    coef = regr.coef_.tolist()
    coef = [regr.intercept_] + coef[1:]
    return coef


def data_split_4channels(data_in):
    ch1i = []
    ch1q = []
    ch2i = []
    ch2q = []
    for i in range(len(data_in)):
        if i%4 == 0:
            ch1i.append(data_in[i])
        elif i%4 == 1:
            ch1q.append(data_in[i])
        elif i%4 == 2:
            ch2i.append(data_in[i])
        else:
            ch2q.append(data_in[i])
    return ch1i,ch1q,ch2i,ch2q

def dBFs_to_amp(x):
    return 10**(x/20.)


def dBFS_to_linearpower(x):
    return 10**(x/10.)

def insert_x(s, ele=0, times=4):
    s = np.insert(s.reshape(-1,1),1,[[ele]]*(times-1),axis=1)
    return s.flatten()

def CWgeneration(N=24576, samplerate=245.76e6, freq_offsets=[1e6], amplitudes=[0], init_phases=[0], save=False, show=False):
    lenf, lenp, lena = len(freq_offsets), len(init_phases), len(amplitudes)
    if lenf > lena:     amplitudes += [amplitudes[-1]]*(lenf - lena)
    elif lenf < lena:       amplitudes = amplitudes[:lenf] 
    
    if lenf > lenp:    init_phases +=[0]*(lenf - lenp)
    elif lenf < lenp:  init_phases = init_phases[:lenf]
    
    N = int(N)
    signal = np.zeros(N,dtype=complex)
    for (f,p,a) in zip(freq_offsets,init_phases,amplitudes):
        #amp = dBFs_to_amp(a) / 0.931 # A Hack cuz the spectrum level is -0.62dBFs lower, need instrument verify
        amp = dBFs_to_amp(a)
        step = (float(f) / float(samplerate)) * 2 * np.pi
        phaseArray = np.arange(N) * step + np.pi/180 * 1j * p
        #欧拉公式，保证正交性，与cos+jsin方法做过对比，生成的信号频谱的均值和方差都更小
        #For a complex sinusoidal theta = 2*pi*f*t where each time step is 1/fs    
        signal += np.exp(1.0j * phaseArray) * amp
    
    #signal /= max(np.abs(signal))
    signal_Is, signal_Qs = signal.real, signal.imag

    if show == True:        plt.plot(signal_Is)
    filename = 0
    if save == True:
        filename = "CW_SAMPR{0}_POINT{1}_OFFSET{2}_CN{3}.txt".format(int(samplerate / 1e6), N, int(freq_offsets[0]), len(freq_offsets))
        df = pd.DataFrame(np.vstack([signal_Is,signal_Qs]).T)
        df.to_csv(filename, sep='\t', header = None, index= False )
        print('Signal source saved to {} succeed'.format(filename))
    elif save == False:
        pass
    else:
        df = pd.DataFrame(np.vstack([signal_Is,signal_Qs]).T)
        df.to_csv(filename, sep='\t', header = None, index= False )
        print('Signal source saved to {} succeed'.format(save))
    return signal_Is, signal_Qs


def CWgeneration2(N=24576, samplerate=245.76e6, freq_offsets=[1e6], save=False, show=False):
    N = int(N)
    ix = np.arange(N)
    signal_complex = np.zeros(N,dtype=complex)
    for freq_offset in freq_offsets:
        signal_complex += np.cos(2 * np.pi * ix * freq_offset / samplerate) + np.sin(2 * np.pi * ix * freq_offset / samplerate)*1j

    max_pwr = max(np.abs(signal_complex))
    signal_complex /= max_pwr 
    print(signal_complex.mean(),np.abs(signal_complex).mean())

    if show == True:        plt.plot(signal_complex.real)

    if save == True:
        f = open("CW_SAMPR{0}_POINT{1}_OFFSET{2}_CN{3}.txt".format(int(samplerate / 1e6), N, int(freq_offsets[0]),
                                                                   len(freq_offsets)), 'wb')
        writer = csv.writer(f, delimiter='\t', )
        for i in range(N):  writer.writerow([signal_complex.real[i], signal_complex.imag[i]])
        f.close()

    return signal_complex.real, signal_complex.imag



#si,sq = CWgeneration2(N=245.76e4,freq_offsets=[118e6],show = True)
#f,psd,dc = fft_spectrum(si,sq,245.76e6,len(si))
#psd,f = ss.my_psd(si+sq*1j,len(si),245.76e6);plt.figure(1);plt.plot(f,psd);plt.show()
#print((si+sq*1j).mean(),(si+sq*1j).std(),psd.mean(),psd.std())
#sii,sqq = CWgeneration(N=245.76e4,freq_offsets=[118e6], amplitudes = [1], show = True) 
#f,psd,dc = fft_spectrum(sii,sqq,245.76e6,len(sii))
#psd,f = ss.my_psd(sii+sqq*1j,len(sii),245.76e6);plt.figure(2);plt.plot(f,psd);plt.show()
#print((sii+sq*1j).mean(),(sii+sqq*1j).std(),psd.mean(),psd.std())

def digital_extend_fallback(si,sq,extendbit = 16,back_off=0,ifsave=False):
    if extendbit == 0:
        return si,sq
    if extendbit > 0:
        gain = 2**round(extendbit-1)-1
        si = map(lambda x: x*gain,si)
        sq = map(lambda x: x*gain,sq)
#        maxi = max(si)
#        maxq = max(sq)
#        si = map(lambda x: x/maxi*gain,si)
#        sq = map(lambda x: x/maxq*gain,sq)

    if back_off != 0:
        back_off = pow(10,back_off/20.0)
        si = map(lambda x: round(x*back_off), si)
        sq = map(lambda x: round(x*back_off), sq)
    else:
        si = map(lambda x: round(float(x)),si)
        sq = map(lambda x: round(float(x)),sq)
    si = map(int,si)
    sq = map(int,sq)
#    meani = round(np.mean(si))
#    meanq = round(np.mean(sq))
#    si = map(lambda x: round(x-meani),si)
#    sq = map(lambda x: round(x-meanq),sq)

    if ifsave:
        f = open("Ex.txt", 'wb')
        writer = csv.writer(f, delimiter='\t' )
        for i in range(len(si)):  writer.writerow([si[i], sq[i]])
        f.close()
    return si,sq

def channelpower(si,sq):
    si = np.array(si)
    sq = np.array(sq)
    linearpwr = np.sum(si**2 + sq**2)
    return 10*np.log10(linearpwr/len(si))

def papr(si,sq):
    ave_pwr = channelpower(si,sq)
    peak_pwr = 10*np.log10( np.max(si**2 + sq**2) )
    return peak_pwr - ave_pwr

def fft_process(sig_i, sig_q, samplingfreq, FFTsize,  ifplot = 1, bit = 0):
    
    if len(sig_i) != len(sig_q):    print ("IQ lenth mismatch")
    else:

        fftsize =np.minimum( FFTsize,len(sig_i) )

        freq = np.linspace(-samplingfreq/2,samplingfreq/2,fftsize,endpoint = False)
        comp = []

        if bit: gain = 2.0**(bit-1)-1
        else:   gain = 1
        for i in range(fftsize):    comp.append( complex(sig_i[i]/gain,sig_q[i]/gain) )
   

#        fft_out = sfft.fft(comp)
#        p_fft_out =  np.abs(sfft.fft(comp)) / fftsize /2
        p_fft_outshift = sfft.fftshift(np.abs(sfft.fft(comp)) / fftsize /2)
        dc_offset = 20*np.log10( sum(p_fft_outshift[FFTsize/2-1:FFTsize/2+2]) )
        fft_out_db = 20*np.log10(np.clip(p_fft_outshift,1e-25,1e100))
        
        if ifplot == 1:
            plt.close('Picture 1')
            plt.figure('Picture 1',figsize = (12,9))
            plt.plot(freq/1e6,fft_out_db,'g')
            plt.xlabel('Frequency (MHz)')
            plt.ylabel('Amplitude (dBFS)')
            plt.grid(True)
            plt.show()
        elif ifplot == 2: 
            plt.close('Picture 2')
            plt.figure('Picture 2',figsize = (12,9))
            plt.plot(freq,fft_out_db,'g')
            plt.xlabel('Frequency (MHz)')
            plt.ylabel('Amplitude (dBFS)')
            plt.grid(True)
            plt.show()        
        return freq, fft_out_db,dc_offset
        

def plotwithreflevel(freqHz,psd,reflevel,span, loMHz=None):
    diff = reflevel - np.max(psd)
    psd += diff
    plt.xlim(span)
    plt.ylim([np.min(psd)-10,reflevel])
    plt.grid(True)
    if loMHz == None:    plt.title('Spectrum @ Ref Level: {}dBm'.format(reflevel))
    else:             plt.title('Spectrum @ Ref Level: {}dBm  LO: {}MHz'.format(reflevel,loMHz))
    plt.plot(freqHz/1e6,psd,'b')
    plt.show()

def fft_transform(sig_i, sig_q, samplingfreq, FFTsize,  show = 1, nolog=False):
    if len(sig_i) != len(sig_q):    print ("IQ lenth mismatch")
    else:

        fftsize =np.minimum( FFTsize,len(sig_i) )

        freq = np.linspace(-samplingfreq/2,samplingfreq/2,fftsize,endpoint = False)
        raw_fft_outshift = sfft.fftshift(sfft.fft(sig_i+1j*sig_q) / fftsize )
        if nolog==False:
            p_fft_outshift = 10*np.log10( np.abs(raw_fft_outshift) / np.sqrt(50) )
            fft_out_db = np.clip(p_fft_outshift,1e-25,1e100)
        else:
            fft_out_db = raw_fft_outshift
        
        if show == 1:
            plt.close('Picture 1')
            plt.figure('Picture 1',figsize = (12,9))
            plt.plot(freq/1e6,fft_out_db,'g')
            plt.xlabel('Frequency (MHz)')
            plt.ylabel('Amplitude (dBFS)')
            plt.grid(True)
            plt.show()
        elif show == 2: 
            plt.close('Picture 2')
            plt.figure('Picture 2',figsize = (12,9))
            plt.plot(freq,fft_out_db,'g')
            plt.xlabel('Frequency (MHz)')
            plt.ylabel('Amplitude (dBFS)')
            plt.grid(True)
            plt.show()        
        return freq, fft_out_db


def fft_spectrum(sig_i, sig_q, samplingfreq, FFTsize,  ifplot = 1, bit = 0):
    
    if len(sig_i) != len(sig_q):    print ("IQ lenth mismatch")
    else:
        fftsize =np.minimum( FFTsize,len(sig_i) )
        #freq = np.linspace(-samplingfreq/2,samplingfreq/2,fftsize,endpoint = False)
        comp = []
        print ('fftsizes = ',fftsize)
        if bit: gain = 2.0**(bit-1)-1
        else:   gain = 1
        for i in range(fftsize):    comp.append( complex(sig_i[i]/gain,sig_q[i]/gain) )
        win = hanning(fftsize)
        #f,psd = signal.welch(comp, samplingfreq, window=win, noverlap=None,scaling = 'spectrum',  nfft=fftsize, return_onesided=False,detrend=False,nperseg= 256)
        f,psd = signal.periodogram(comp, samplingfreq, window=win,scaling = 'spectrum',  nfft=fftsize, return_onesided=False,detrend=False)
        f = np.fft.fftshift(f)
        psd = np.fft.fftshift(psd)
        #dc_offset = 10*np.log10( sum(psd[fftsize//2-1:fftsize//2+2]) )
        dc_offset = 10*np.log10(psd[fftsize//2])
        psd = 10*np.log10(np.clip(psd,1e-25,1e100))
        
        if ifplot == 1:
            plt.close('Picture 1')
            plt.figure('Picture 1',figsize = (12,9))
            plt.plot(f/1e6,psd,'g')
            plt.xlabel('Frequency (MHz)')
            plt.ylabel('Amplitude (dBFS)')
            plt.grid(True)
            plt.show()
        elif ifplot == 2: 
            plt.close('Picture 2')
            plt.figure('Picture 2',figsize = (12,9))
            plt.plot(f,psd,'g')
            plt.xlabel('Frequency (MHz)')
            plt.ylabel('Amplitude (dBFS)')
            plt.grid(True)
            plt.show()        
        gc.collect()
        return f,psd,dc_offset

def fft_spectrum2(sig_i, sig_q, samplingfreq, FFTsize,  show = True, average = 0):
    if len(sig_i) != len(sig_q):    print ("IQ lenth mismatch")
    else:
        comp = sig_i + sig_q*1j
        fftsize = FFTsize
        #win = hanning(len(comp))
        #f,psd = signal.welch(comp, samplingfreq, window=win, noverlap=None,scaling = 'spectrum',  nfft=fftsize, return_onesided=False,detrend=False,nperseg= 256)
        f, psd = 0, 0
        if average < 2:
            fftsize =np.minimum( FFTsize,len(comp)*10 )
            f,psd = signal.periodogram(comp, samplingfreq, window='flattop',scaling = 'spectrum',  nfft=fftsize, return_onesided=False,detrend=False)
        else:
            psd_s = 0
            for i in range(average):
                signals = comp[i].reshape((1,-1)).tolist()[0]
                fftsize =np.minimum( FFTsize,len(signals)*10 )
                f,psd = signal.periodogram(signals, samplingfreq, window='flattop',scaling = 'spectrum',  nfft=fftsize, return_onesided=False,detrend=False)
               # print(psd.shape,fftsize,len(signals))
                psd_s+= psd                
            psd = psd_s / average
        f = np.fft.fftshift(f)
        psd = np.fft.fftshift(psd)
        dc_offset = 10*np.log10(psd[int(fftsize//2)])


        psd = 10*np.log10(np.clip(psd,1e-25,1e100))
        
        if show == True:
            plt.close('Picture 1')
            plt.figure('Picture 1',figsize = (12,9))
            plt.plot(f/1e6,psd,'g')
            plt.xlabel('Frequency (MHz)')
            plt.ylabel('Amplitude (dBFS)')
            plt.grid(True)
            if average < 2:    plt.title('Spectrum PWR')
            else:              plt.title('Spectrum PWR with Average {}'.format(average))
            plt.show()      
        gc.collect()
        return f,psd,dc_offset

def fft_density(sig_i, sig_q, samplingfreq, FFTsize,  show = True):
    if len(sig_i) != len(sig_q):    print ("IQ lenth mismatch")
    else:
        fftsize =np.minimum( FFTsize,len(sig_i)*10 )
        comp = sig_i + sig_q*1j
        f,psd = signal.periodogram(comp, samplingfreq, window='flattop',scaling = 'density',  nfft=fftsize, return_onesided=False,detrend=False)
        f = np.fft.fftshift(f)
        psd = np.fft.fftshift(psd)

        dc_offset = 10*np.log10(psd[fftsize//2])

        #psd = 10*np.log10(np.clip(psd,1e-25,1e100))
        
        if show == True:
            plt.close('Picture 1')
            plt.figure('Picture 1',figsize = (12,9))
            plt.plot(f/1e6,psd,'g')
            plt.xlabel('Frequency (MHz)')
            plt.ylabel('Amplitude (dBFS)')
            plt.grid(True)
            plt.show()      
        gc.collect()
        return f,psd,dc_offset

def rmsNoiseFloor(linearpsd, ave = 1):
    return 10*np.log10(linearpsd / int(ave)).mean()

def peaksearch(signal_in,snr_min = 8):
    vbw = len(signal_in)/1000
    indexs = signal.find_peaks_cwt(signal_in,np.arange(1,vbw),min_snr = snr_min)
    search_range = 3
    index = []
    peaks = []
    for i in indexs:
        if i <= search_range:
            maxindex = np.argmax(signal_in[:2*search_range])
        elif i >= len(signal_in)- search_range:
            maxindex = np.argmax(signal_in[len(signal_in) - 2*search_range:])
            maxindex += len(signal_in)-2*search_range
        else:
            maxindex = np.argmax(signal_in[i-search_range:i+search_range+1])
            maxindex += i-search_range
            
        index.append(maxindex)
     
    
    for i in index:       peaks.append(signal_in[i])
    gc.collect()            
    return index,peaks     
   

def maxcwpwr_V2(sig_in,num):
    cwpwr_index = []
    cwpwr_value = []
    dc_index = len(sig_in)/2
    for j in range(num):
        peak = max(sig_in)
        peak_index = sig_in.tolist().index(peak)
        sum = 0
        #for i in sig_in[peak_index-1:peak_index+2]:     sum+=pow(10,i/10)
        #cwpwr_value.append( 10*np.log10(sum) )
        cwpwr_value.append( sig_in[peak_index] )
        cwpwr_index.append(peak_index)
        
        sum = 0
        img_index = int(2*dc_index - peak_index)
        #for i in sig_in[img_index-1:img_index+2]:       sum+=pow(10,i/10)
        #cwpwr_value.append( 10*np.log10(sum) )
        cwpwr_value.append( sig_in[img_index] )
        cwpwr_index.append(img_index)
        
        for i in range(-3,3):
            if peak_index + i < 0:
                sig_in[0] = -360
            elif peak_index + i>= len(sig_in):
                sig_in[len(sig_in)-1] = -360
            else:
                sig_in[peak_index + i] = -360
                      
            if img_index + i < 0:
                sig_in[0] = -360
            elif img_index + i>= len(sig_in):
                sig_in[len(sig_in)-1] = -360                      
            else:    
                sig_in[ img_index + i] = -360
    gc.collect()
    return cwpwr_index,cwpwr_value


def maxcwpwr(sig_in,num):
    print("[Warning] This function is out of date, use maxcwpwr_V2 instead")
    cwpwr_index = []
    cwpwr_value = []
    dc_index = len(sig_in)/2
    for j in range(num):
        peak = max(sig_in)
        peak_index = sig_in.tolist().index(peak)
        sum = 0
        for i in sig_in[peak_index-20:peak_index+20]:     sum+=pow(10,i/20)
        cwpwr_value.append( 20*np.log10(sum) )
        cwpwr_index.append(peak_index)
        
        sum = 0
        img_index = 2*dc_index - peak_index
        for i in sig_in[img_index-10:img_index+10]:       sum+=pow(10,i/20)
        cwpwr_value.append( 20*np.log10(sum) )
        cwpwr_index.append(img_index)
        
        for i in range(-5000,5000):
            if peak_index + i < 0:
                sig_in[0] = -360
            elif peak_index + i>= len(sig_in):
                sig_in[len(sig_in)-1] = -360
            else:
                sig_in[peak_index + i] = -360
                      
            if img_index + i < 0:
                sig_in[0] = -360
            elif img_index + i>= len(sig_in):
                sig_in[len(sig_in)-1] = -360                      
            else:    
                sig_in[ img_index + i] = -360

    return cwpwr_index,cwpwr_value


def maxcwpwrWithwidthsetting(sig_in,num,sampleRateMHz,widthMHz = 1):
    cwpwr_index = []
    cwpwr_value = []
    dc_index = len(sig_in)/2
    ratio = sampleRateMHz*1e6 / len(sig_in)
    
    for j in range(num):
        peak = max(sig_in)
        peak_index = sig_in.tolist().index(peak)
        sum = 0
        for i in sig_in[peak_index-20:peak_index+20]:     sum+=pow(10,i/20)
        cwpwr_value.append( 20*np.log10(sum) )
        cwpwr_index.append(peak_index)
        
        sum = 0
        img_index = 2*dc_index - peak_index
        for i in sig_in[img_index-10:img_index+10]:       sum+=pow(10,i/20)
        cwpwr_value.append( 20*np.log10(sum) )
        cwpwr_index.append(img_index)
        
        widthPoints = widthMHz*1e6/ratio
        for i in range(-widthPoints,widthPoints):
            if peak_index + i < 0:
                sig_in[0] = -360
            elif peak_index + i>= len(sig_in):
                sig_in[len(sig_in)-1] = -360
            else:
                sig_in[peak_index + i] = -360
                      
            if img_index + i < 0:
                sig_in[0] = -360
            elif img_index + i>= len(sig_in):
                sig_in[len(sig_in)-1] = -360                      
            else:    
                sig_in[ img_index + i] = -360

    return cwpwr_index,cwpwr_value



def imagedet_fast(peak_freq,peak_val,sig_freq):
    sig_index = np.abs(peak_freq - sig_freq).argmin()
    img_index = np.abs(peak_freq + peak_freq[sig_index]).argmin()
    return peak_freq[img_index],peak_val[img_index]
    
def list_sum(a1,a2):
    if len(a1)!=len(a2):        print ('Length mismatch')
    else:
        return list(map(lambda x: x[0]+x[1], zip(a2, a1)))
    
    

def do_ironscript(script_path):
    IRON_PYTHON_CMD = 'cd C:\\Program Files (x86)\\IronPython 2.7'    
    try:
        f = open(script_path,'r')
        f.close()
    except:
        print ("No Init Script named " + script_path)
        return -1
        
    try:
        print (IRON_PYTHON_CMD+'&&ipy '+'"'+script_path+'"')
        res = subprocess.check_output(IRON_PYTHON_CMD+'&&ipy '+'"'+script_path+'"' , shell = True)
        print (res)
    except:
        print ("Init Script running failure")
        return -2
        
    return 0


def get_LTEdatasource(path):
    files = os.listdir(path)
    result = []
    import re
    regex = re.compile(r'(.*)_Setup\.txt')     
    for f in files:
        r = re.findall(regex,f)
        if len(r) < 1:
            pass
        else:
            if r[0]+r'.txt' in files:
                result.append(r[0])
    return result
                

def analyse_LTEsource(setup_file_name):
    carrier_num = 0
    carrier1=88
    carrier2=88
    carrier3=88
    carrier4=88
    sig_band=0
    car_band=0
    speed = 0
    ff = open(setup_file_name,'r')
    f = ff.read()
    index = f.find('Setup:')+5
    import re
    reg = re.compile(r'Carrier Bandwidth = (\d+) MHz')
    result = re.findall(reg,f[index:])
    
    if len(result) > 0:
        car_band = result[0]
    
    reg = re.compile(r'Signal Bandwidth = (\d+) MHz')
    result = re.findall(reg,f[index:])
    if len(result) > 0:
        sig_band = result[0]   
        
      
    reg = re.compile(r'Carrier 1 Center Freq\. = ([-]?[0-9]+\.[0-9]+) MHz')
    result = re.findall(reg,f[index:])
    if len(result) > 0:
        carrier1 = result[0]
        carrier_num+=1        
        
    reg = re.compile(r'Carrier 2 Center Freq\. = ([-]?[0-9]+\.[0-9]+) MHz')
    result = re.findall(reg,f[index:])
    if len(result) > 0:
        carrier2 = result[0]
        carrier_num+=1          

    reg = re.compile(r'Carrier 3 Center Freq\. = ([-]?[0-9]+\.[0-9]+) MHz')
    result = re.findall(reg,f[index:])
    if len(result) > 0:
        carrier3 = result[0] 
        carrier_num+=1

    reg = re.compile(r'Carrier 4 Center Freq\. = ([-]?[0-9]+\.[0-9]+) MHz')
    result = re.findall(reg,f[index:])
    if len(result) > 0:
        carrier4 = result[0] 
        carrier_num+=1
        
    reg = re.compile(r'Output Sample Rate = ([0-9]+\.[0-9]+) MHz')
    result = re.findall(reg,f[index:])
    if len(result) > 0:
        speed = result[0] 
        
        
    return [int(carrier_num),float(carrier1),float(carrier2),float(carrier3),float(carrier4),float(car_band),float(sig_band),float(speed)]


def get_LTEcarrierPositions_bitmode(wave_info):
    if len(wave_info) <3 or wave_info[0] == 0:
        print ('Invalid wave info provided')
        return 0
    pos = 0
    pos_all = int(wave_info[-2]/wave_info[-3])
    cnt = 0    
    for i in range(int(wave_info[-2]/-2) ,int(wave_info[-2]/2),int(wave_info[-3])):
        for j in range(wave_info[0]):
            if wave_info[j+1] >= i and wave_info[j+1] < i+wave_info[-3]:
                pos |= 1<<(pos_all-cnt-1)
                break
        cnt+=1
    return pos
    
def integrated_dBpower(pwr_list):
    totalpwr = 0
    for i in pwr_list:       totalpwr += dBFS_to_linearpower(i)
    return 10*np.log10(totalpwr)


def detect_peaks(x, mph=None, mpd=1, threshold=0, edge='rising',
                 kpsh=False, valley=False, show=False, ax=None):

    """Detect peaks in data based on their amplitude and other features.
    Parameters
    ----------
    x : 1D array_like
        data.
    mph : {None, number}, optional (default = None)
        detect peaks that are greater than minimum peak height (if parameter
        `valley` is False) or peaks that are smaller than maximum peak height
         (if parameter `valley` is True).
    mpd : positive integer, optional (default = 1)
        detect peaks that are at least separated by minimum peak distance (in
        number of data).
    threshold : positive number, optional (default = 0)
        detect peaks (valleys) that are greater (smaller) than `threshold`
        in relation to their immediate neighbors.
    edge : {None, 'rising', 'falling', 'both'}, optional (default = 'rising')
        for a flat peak, keep only the rising edge ('rising'), only the
        falling edge ('falling'), both edges ('both'), or don't detect a
        flat peak (None).
    kpsh : bool, optional (default = False)
        keep peaks with same height even if they are closer than `mpd`.
    valley : bool, optional (default = False)
        if True (1), detect valleys (local minima) instead of peaks.
    show : bool, optional (default = False)
        if True (1), plot data in matplotlib figure.
    ax : a matplotlib.axes.Axes instance, optional (default = None).
    Returns
    -------
    ind : 1D array_like
        indeces of the peaks in `x`.
    Notes
    -----
    The detection of valleys instead of peaks is performed internally by simply
    negating the data: `ind_valleys = detect_peaks(-x)`
    
    The function can handle NaN's 
    See this IPython Notebook [1]_.
    References
    ----------
    .. [1] http://nbviewer.ipython.org/github/demotu/BMC/blob/master/notebooks/DetectPeaks.ipynb
    Examples
    --------
    >>> from detect_peaks import detect_peaks
    >>> x = np.random.randn(100)
    >>> x[60:81] = np.nan
    >>> # detect all peaks and plot data
    >>> ind = detect_peaks(x, show=True)
    >>> print(ind)
    >>> x = np.sin(2*np.pi*5*np.linspace(0, 1, 200)) + np.random.randn(200)/5
    >>> # set minimum peak height = 0 and minimum peak distance = 20
    >>> detect_peaks(x, mph=0, mpd=20, show=True)
    >>> x = [0, 1, 0, 2, 0, 3, 0, 2, 0, 1, 0]
    >>> # set minimum peak distance = 2
    >>> detect_peaks(x, mpd=2, show=True)
    >>> x = np.sin(2*np.pi*5*np.linspace(0, 1, 200)) + np.random.randn(200)/5
    >>> # detection of valleys instead of peaks
    >>> detect_peaks(x, mph=-1.2, mpd=20, valley=True, show=True)
    >>> x = [0, 1, 1, 0, 1, 1, 0]
    >>> # detect both edges
    >>> detect_peaks(x, edge='both', show=True)
    >>> x = [-2, 1, -2, 2, 1, 1, 3, 0]
    >>> # set threshold = 2
    >>> detect_peaks(x, threshold = 2, show=True)
    Version history
    ---------------
    '1.0.5':
        The sign of `mph` is inverted if parameter `valley` is True
    
    """
    x = np.atleast_1d(x).astype('float64')
    if x.size < 3:
        return np.array([], dtype=int)
    if valley:
        x = -x
        if mph is not None:
            mph = -mph
    # find indices of all peaks
    dx = x[1:] - x[:-1]
    # handle NaN's
    indnan = np.where(np.isnan(x))[0]
    if indnan.size:
        x[indnan] = np.inf
        dx[np.where(np.isnan(dx))[0]] = np.inf
    ine, ire, ife = np.array([[], [], []], dtype=int)
    if not edge:
        ine = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) > 0))[0]
    else:
        if edge.lower() in ['rising', 'both']:
            ire = np.where((np.hstack((dx, 0)) <= 0) & (np.hstack((0, dx)) > 0))[0]
        if edge.lower() in ['falling', 'both']:
            ife = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) >= 0))[0]
    ind = np.unique(np.hstack((ine, ire, ife)))
    # handle NaN's
    if ind.size and indnan.size:
        # NaN's and values close to NaN's cannot be peaks
        ind = ind[np.in1d(ind, np.unique(np.hstack((indnan, indnan-1, indnan+1))), invert=True)]
    # first and last values of x cannot be peaks
    if ind.size and ind[0] == 0:
        ind = ind[1:]
    if ind.size and ind[-1] == x.size-1:
        ind = ind[:-1]
    # remove peaks < minimum peak height
    if ind.size and mph is not None:
        ind = ind[x[ind] >= mph]
    # remove peaks - neighbors < threshold
    if ind.size and threshold > 0:
        dx = np.min(np.vstack([x[ind]-x[ind-1], x[ind]-x[ind+1]]), axis=0)
        ind = np.delete(ind, np.where(dx < threshold)[0])
    # detect small peaks closer than minimum peak distance
    if ind.size and mpd > 1:
        ind = ind[np.argsort(x[ind])][::-1]  # sort ind by peak height
        idel = np.zeros(ind.size, dtype=bool)
        for i in range(ind.size):
            if not idel[i]:
                # keep peaks with the same height if kpsh is True
                idel = idel | (ind >= ind[i] - mpd) & (ind <= ind[i] + mpd) \
                    & (x[ind[i]] > x[ind] if kpsh else True)
                idel[i] = 0  # Keep current peak
        # remove the small peaks and sort back the indices by their occurrence
        ind = np.sort(ind[~idel])

    if show:
        if indnan.size:
            x[indnan] = np.nan
        if valley:
            x = -x
            if mph is not None:
                mph = -mph
        _plot(x, mph, mpd, threshold, edge, valley, ax, ind)

    return ind


def _plot(x, mph, mpd, threshold, edge, valley, ax, ind):
    """Plot results of the detect_peaks function, see its help."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print('matplotlib is not available.')
    else:
        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=(8, 4))

        ax.plot(x, 'b', lw=1)
        ax.grid()
        if ind.size:
            label = 'valley' if valley else 'peak'
            label = label + 's' if ind.size > 1 else label
            ax.plot(ind, x[ind], '*', mfc=None, mec='r', mew=2, ms=8,
                    label='%d %s' % (ind.size, label))
            ax.legend(loc='best', framealpha=.5, numpoints=1)
        ax.set_xlim(-.02*x.size, x.size*1.02-1)
        ymin, ymax = x[np.isfinite(x)].min(), x[np.isfinite(x)].max()
        yrange = ymax - ymin if ymax > ymin else 1
        ax.set_ylim(ymin - 0.1*yrange, ymax + 0.1*yrange)
        ax.set_xlabel('Data #', fontsize=14)
        ax.set_ylabel('Amplitude', fontsize=14)
        mode = 'Valley detection' if valley else 'Peak detection'
        ax.set_title("%s (mph=%s, mpd=%d, threshold=%s, edge='%s')" % (mode, str(mph), mpd, str(threshold), edge))
        ax.set_title("Peak Detection")
        # plt.grid()
        plt.show()   
        
def peaksearch_v2(sigin,threshold = None,show=False):
    if threshold == None:
        indexs = detect_peaks(sigin, mph=sigin.mean()+sigin.std(), mpd=2, show=show)
    else:
        indexs = detect_peaks(sigin, mph=threshold, mpd=2, show=show)
    return indexs, [sigin[i] for i in indexs ]

def getCorrPeak(tx=[1,2,3], rx=[3,2,1,2,3]):
    r = np.correlate(rx,tx,'full')
    return np.argmax(r) - len(tx) + 1, r

class OFDM:
    def __init__(self, num_subcarriers = 64, num_pilots = 8, pilot_val = 3+3j, qam_order = 16):
        self.K = num_subcarriers # number of OFDM subcarriers
        offset = int(0.1*num_subcarriers)
        self.Kon = int(self.K-2*offset)
        self.CP = self.K//4  # length of the cyclic prefix: 25% of the block
        self.P = num_pilots # number of pilot carriers per OFDM block
        self.pilotValue = pilot_val # The known value each pilot transmits
        self.allCarriers_index = np.arange(self.K)  # indices of all subcarriers ([0, 1, ... K-1])
        self.allActiveCarriers_index = self.allCarriers_index[offset:-offset] # indices of active subcarriers ([offset, 1, ... -offset])
        self.pilotCarriers_index = self.allActiveCarriers_index[::self.Kon//self.P] # Pilots is every (K/P)th carrier.
        # For convenience of channel estimation, let's make the last carriers also be a pilot
        self.pilotCarriers_index = np.hstack([self.pilotCarriers_index, np.array([self.allActiveCarriers_index[-1]])])
        self.P += 1
        # data carriers are all remaining carriers
        self.dataCarriers_index = [i for i in self.allActiveCarriers_index if i not in self.pilotCarriers_index]
        self.mu = int(np.log2(qam_order))
        print ("allCarriers:       %s" % self.allCarriers_index)
        print ("allActiveCarriers: %s" % self.allActiveCarriers_index)
        print ("pilotCarriers:     %s" % self.pilotCarriers_index)
        print ("dataCarriers:      %s" % self.dataCarriers_index)
        print ("Bits per sample  %s" % self.mu)
        
        self.mapping_table = {(0,0,0,0) : -3-3j,    (0,0,0,1) : -3-1j,    (0,0,1,0) : -3+3j,    (0,0,1,1) : -3+1j,
                              (0,1,0,0) : -1-3j,    (0,1,0,1) : -1-1j,    (0,1,1,0) : -1+3j,    (0,1,1,1) : -1+1j,
                              (1,0,0,0) :  3-3j,    (1,0,0,1) :  3-1j,    (1,0,1,0) :  3+3j,    (1,0,1,1) :  3+1j,
                              (1,1,0,0) :  1-3j,    (1,1,0,1) :  1-1j,    (1,1,1,0) :  1+3j,    (1,1,1,1) :  1+1j
                             }   
        self.demapping_table = {v : k for k, v in self.mapping_table.items()}
        self.payloadBits_per_OFDM = len(self.dataCarriers_index)*self.mu  # number of payload bits per OFDM symbol
        self.channelResponse = np.array([1+1j])#np.array([1, 0, 0.3+0.3j])
        
    def SP(self,bits):
        remain_samples = (len(self.dataCarriers_index) * self.mu) - int( len(bits) % (len(self.dataCarriers_index) * self.mu) )
        print(remain_samples, bits.shape)
        bits = np.hstack( [bits,np.zeros(remain_samples,dtype=complex)] )
        print(bits.shape)
        return bits.reshape((len(self.dataCarriers_index), self.mu))


    def Mapping(self,bits):
        return np.array([self.mapping_table[tuple(b)] for b in bits])


    def OFDM_symbol(self,QAM_payload):
        symbol = np.zeros(self.K, dtype=complex) # the overall K subcarriers
        symbol[self.pilotCarriers_index] = self.pilotValue  # allocate the pilot subcarriers 
        symbol[self.dataCarriers_index] = QAM_payload  # allocate the pilot subcarriers
        return symbol

    def addCP(self,OFDM_time,show = True):
        cp = OFDM_time[-self.CP:]               # take the last CP samples ...
        wave =  np.hstack([cp, OFDM_time])  # ... and add them to the beginning
        if show == True:
            plt.figure()
            f = np.linspace(-self.K/2, self.K/2, len(wave)*8, endpoint=False)
            #plt.plot(f, 20*np.log10(abs(np.fft.fftshift(np.fft.fft(wave, 8*len(wave))/np.sqrt(len(wave))))))
            plt.plot(f, 20*np.log10(abs(np.fft.fft(wave, 8*len(wave))/np.sqrt(len(wave)))))
            plt.show()
        return wave
    
    def channel(self,signal,SNRdb = 25):
        convolved = np.convolve(signal, self.channelResponse)
        signal_power = np.mean(abs(convolved**2))
        sigma2 = signal_power * 10**(-SNRdb/10)  # calculate noise power based on signal power and SNR
        print ("RX Signal power: %.4f. Noise power: %.4f" % (signal_power, sigma2))
        # Generate complex noise with given variance
        noise = np.sqrt(sigma2/2) * (np.random.randn(*convolved.shape)+1j*np.random.randn(*convolved.shape))
        return convolved + noise
    
    def removeCP(self,signal):
        return signal[self.CP:(self.CP+self.K)]

    def DFT(self,OFDM_RX):
        return np.fft.fft(OFDM_RX)

    def IDFT(self,OFDM_data):
        return np.fft.ifft(OFDM_data)
    
    def channelEstimate(self,OFDM_demod = [], show = True):
        pilots = OFDM_demod[self.pilotCarriers_index]  # extract the pilot values from the RX signal
        Hest_at_pilots = pilots / self.pilotValue # divide by the transmitted pilot values
    
        # Perform interpolation between the pilot carriers to get an estimate
        # of the channel in the data carriers. Here, we interpolate absolute value and phase 
        # separately
        Hest_abs = interpolate.interp1d(self.pilotCarriers_index, abs(Hest_at_pilots), kind='linear')(self.allActiveCarriers_index)
        Hest_phase = interpolate.interp1d(self.pilotCarriers_index, np.angle(Hest_at_pilots), kind='linear')(self.allActiveCarriers_index)
        Hest = Hest_abs * np.exp(1j*Hest_phase)
        if show == True:
            plt.figure()
            #plt.plot(self.allCarriers_index, abs(H_exact), label='Correct Channel')
            plt.stem(self.pilotCarriers_index, abs(Hest_at_pilots), label='Pilot estimates')
            plt.plot(self.allActiveCarriers_index, abs(Hest), label='Estimated channel via interpolation')
            plt.grid(True); plt.xlabel('Carrier index'); plt.ylabel('$|H(f)|$'); plt.legend(fontsize=10)
            plt.show()
        return Hest

    def equalize(self,OFDM_demod, Hest):
        OFDM_demod[self.allActiveCarriers_index] /= Hest    
        return OFDM_demod
    
    def get_payload(self,equalized,show=True):
        QAM_est = equalized[self.dataCarriers_index]
        if show == True:
            plt.figure()
            plt.plot(QAM_est.real, QAM_est.imag, 'bo')
            plt.grid(True, linestyle = "-.")
            plt.show()
        return QAM_est

    def Demapping(self,QAM, show = True):
        # array of possible constellation points
        constellation = np.array([x for x in self.demapping_table.keys()])
    
        # calculate distance of each RX point to each possible point
        dists = abs(QAM.reshape((-1,1)) - constellation.reshape((1,-1)))
    
        # for each element in QAM, choose the index in constellation 
        # that belongs to the nearest constellation point
        const_index = dists.argmin(axis=1)
    
        # get back the real constellation point
        hardDecision = constellation[const_index]
    
        if show == True:
            plt.figure()
            for qam, hard in zip(QAM, hardDecision):
                plt.plot([qam.real, hard.real], [qam.imag, hard.imag], 'b-o');
                plt.plot(hardDecision.real, hardDecision.imag, 'ro')
            plt.grid(True, linestyle = "-.")
            plt.show()

        # transform the constellation point into the bit groups
        return np.vstack([self.demapping_table[C] for C in hardDecision]), hardDecision

try:    
    from sk_dsp_comm import digitalcom as dc
    class SIMPLEOFDM:
        def __init__(self,fsMHz = 245.76, total_carriers = 64, used_carriers = 32, num_pilots = 8, pilot_val = 3+3j, qam_order = 16):
            if qam_order in (2, 4, 16, 64, 256):    self.QAM_MODE = qam_order  
            else:  self.QAM_MODE = 16
            self.FSMHZ = fsMHz
            self.TOTAL_SUBCARRIERS = total_carriers
            self.USED_CARRIERS = used_carriers  
            self.NUM_PIOLOTS = num_pilots
            self.PILOT_PATTEN = pilot_val
            self.H = 0
            self.initScrambleStatus = 0
            self.CPLEN = 0
            
        def qam_mapping(self,N_symb=10000,Ns=1, data = None, plot=False):
            iq_uni,b,iq = 0,0,0
            if isinstance(None, type(data)) == False:
                iq_uni,b,iq = dc.QAM_gray_encode_bb(None,Ns,self.QAM_MODE,'src', ext_data = data)
            else:    iq_uni,b,iq = dc.QAM_gray_encode_bb(N_symb,Ns,self.QAM_MODE,'src', ext_data = data)
            if plot == True:
                plt.plot(iq_uni.real,iq_uni.imag,'.')
                plt.title('Constellation')
                plt.xlabel('In-Phase')
                plt.ylabel('Quadrature')
                plt.axis('equal')
                plt.grid()
                plt.show() 
            return iq_uni,b,iq
        
        def OFDM_tx(self, data, dataCarrierNums, totalCarrierNums, pilotNums, cplen):
            data = np.array(data)[:len(data) // dataCarrierNums * dataCarrierNums]
            data = data.reshape(-1,dataCarrierNums)
            numOfRows = data.shape[0]
            zeroCarriers = np.zeros((numOfRows, totalCarrierNums - dataCarrierNums))
            finalData = np.concatenate([data[:,:dataCarrierNums//2],zeroCarriers,data[:,dataCarrierNums//2:]],axis = 1)
            rawSymble = np.fft.ifft(finalData)
            if cplen > 0:
                rawSymble = np.concatenate([rawSymble[:,rawSymble.shape[1]-1-cplen:],rawSymble], axis = 1)
                
            rawSymble = rawSymble.reshape(rawSymble.size)
            self.CPLEN = cplen
            return rawSymble
        
        def OFDM_rx(self, data, dataCarrierNums, totalCarrierNums, pilotNums, cplen):
            data = np.array(data).reshape(-1, totalCarrierNums + cplen)
            #Remove CP
            data = data[:,cplen:]
            symble = np.fft.fft(data)
            symble = np.concatenate([symble[:,:dataCarrierNums//2], symble[:,totalCarrierNums - dataCarrierNums//2:]],axis = 1)
            return symble.reshape(symble.size)
        
        def ofdm_symble(self,qam_data,cplen = 0, plot = False):
            if cplen == 0:
                r = dc.OFDM_tx(qam_data,self.USED_CARRIERS,self.TOTAL_SUBCARRIERS,self.NUM_PIOLOTS,False,0) * np.sqrt(self.TOTAL_SUBCARRIERS)
            else:
                r = dc.OFDM_tx(qam_data,self.USED_CARRIERS,self.TOTAL_SUBCARRIERS,self.NUM_PIOLOTS,True,int(cplen)) * np.sqrt(self.TOTAL_SUBCARRIERS)
            if plot == True:
                plt.psd(r, self.TOTAL_SUBCARRIERS + cplen,self.FSMHZ);
                plt.xlabel(r'Frequency (MHz)')
                #plt.ylim(-240)
                plt.show()
            self.CPLEN = cplen
            return r
    
        def ofdm_symbleV2(self,qam_data,cplen = 0, plot = False):
            if cplen == 0:
                #r = dc.OFDM_tx(qam_data,self.USED_CARRIERS,self.TOTAL_SUBCARRIERS,self.NUM_PIOLOTS,False,0) * np.sqrt(self.TOTAL_SUBCARRIERS)
                r = self.OFDM_tx(qam_data,self.USED_CARRIERS,self.TOTAL_SUBCARRIERS,self.NUM_PIOLOTS,0) * np.sqrt(self.TOTAL_SUBCARRIERS)
            else:
                #r = dc.OFDM_tx(qam_data,self.USED_CARRIERS,self.TOTAL_SUBCARRIERS,self.NUM_PIOLOTS,True,int(cplen)) * np.sqrt(self.TOTAL_SUBCARRIERS)
                r = self.OFDM_tx(qam_data,self.USED_CARRIERS,self.TOTAL_SUBCARRIERS,self.NUM_PIOLOTS,cplen) * np.sqrt(self.TOTAL_SUBCARRIERS)
            if plot == True:
                plt.psd(r, self.TOTAL_SUBCARRIERS + cplen,self.FSMHZ);
                plt.xlabel(r'Frequency (MHz)')
                #plt.ylim(-240)
                plt.show()
            return r
        
        def ofdm_demode(self, data):
            return self.OFDM_rx(data, self.USED_CARRIERS, self.TOTAL_SUBCARRIERS, 0, self.CPLEN) / np.sqrt(self.TOTAL_SUBCARRIERS)
        
        def channel_equal_estimate(self, patten, preamble):
            data_preamble, channel = dc.OFDM_rx(preamble, self.USED_CARRIERS, self.TOTAL_SUBCARRIERS, 0, False, 0, alpha=0.95, ht= None)
            self.H = np.true_divide(patten, data_preamble)
            return self.H
        
        def channel_equal_apply(self, data_frames, H = None):
            if isinstance(None, type(H)) == True:
                data_after_equal = (data_frames.reshape(-1, self.USED_CARRIERS)*self.H).reshape(-1,1).flatten()
            else:
                data_after_equal = (data_frames.reshape(-1, self.USED_CARRIERS)*H).reshape(-1,1).flatten()
            return data_after_equal
        
        def scramble(self, data, patten = [7,4],initStatus = '1111111'):
            self.initScrambleStatus = initStatus
            status = initStatus
            out = []   
            for i in data:
                temp = 0
                for j in range(len(patten)):    temp ^= int(status[patten[j]-1])
                out.append(i^temp)
                status = str(temp) + status[:-1]
            return np.array(out)
        
        def unscramble(self, data, patten = [7,4]):
            return self.scramble(data, patten, self.initScrambleStatus)
except:
    pass

class TxDigital:
    def __init__(self, fs_inMHz, fs_outMHz):
        self.FS_INMHZ = fs_inMHz
        self.FSOUTMHZ = fs_outMHz
        
    def nco_mixer(self, freqMHz, data_in,fs_inMHz = None):
        length = np.arange(len(data_in))
        if np.isclose(freqMHz,0) == True:   return data_in
        if fs_inMHz != None:     return data_in*np.exp(2*np.pi*1j*freqMHz / fs_inMHz * length)
        else:     return data_in*np.exp(2*np.pi*1j*freqMHz / self.FSOUTMHZ * length)
    
    def upsampling(self, data_in, times, w = 0.6, plotfilter = False):
        times = int(times)
        sos = signal.butter(87, w/times, 'lp',output='sos')
        #b,a = signal.butter(97, w/times, 'lp')
        if plotfilter == True:
            w, h = signal.sosfreqz(sos)
            plt.semilogx(w, 20 * np.log10(abs(h)))
            plt.title('Butterworth filter frequency response')
            plt.xlabel('Frequency [radians / second]')
            plt.ylabel('Amplitude [dB]')
            plt.margins(0, 0.1)
            plt.grid(which='both', axis='both')
            plt.axvline(0.25, color='green') # cutoff frequency
            plt.show()
        updata = insert_x(data_in,times=times)
        filtered = signal.sosfiltfilt(sos, updata)
        return filtered   
    
    def upsampling2(self,data_in, times, w = 0.6, plotfilter = False):
        times = int(times)
        updata = signal.resample(data_in, times*len(data_in), window = None)
        sos = signal.butter(87, w/times, 'lp',output='sos')
        filtered = signal.sosfiltfilt(sos, updata)
        return filtered
    
    def addwindow(self, data, window = 'hanning'):
        WINDOWLEN = 2048
        shape = data.shape
        if window == 'hanning':    window = signal.hanning(WINDOWLEN).reshape([-1,1])
        else:    return data
        
        window_seq = np.concatenate((window[:WINDOWLEN//2],np.ones(len(data) - WINDOWLEN).reshape([-1,1]),window[WINDOWLEN//2:]))
        return (data.reshape([-1,1])*window_seq).reshape(shape)

class RxDigital:
    def __init__(self, fs_inMHz, fs_outMHz):
        self.FS_INMHZ = fs_inMHz
        self.FSOUTMHZ = fs_outMHz
    
    def decimate(self,data_in,deci_rate, start = 0):
        return np.array([data_in[i] for i in range(int(start), len(data_in), int(deci_rate))])
        
    def downsampling(self, data_in, times, wcutoff = 0.6, start_position = 0):
        sos = signal.butter(87, wcutoff/times*2, 'lp',output='sos')
        filtered = signal.sosfiltfilt(sos, data_in)
        y = self.decimate(filtered, times, start_position)       
        return y
    
    def nco_mixer(self, freqMHz, data_in, fs_inMHz = None):
        length = np.arange(len(data_in))
        if np.isclose(freqMHz,0) == True:   return data_in
        if fs_inMHz != None:     return data_in*np.exp(2*np.pi*1j*freqMHz / fs_inMHz * length)
        else:     return data_in*np.exp(2*np.pi*1j*freqMHz / self.FSOUTMHZ * length)

class ADC_Eval:
    def __init__(self, fs = 1000):
        self.fs = fs
        self.rawFFTData = 0
        self.SpectrumDataDB = 0
        self.DcPwrDB = 0
        self.freqHz = 0
        self.binHarmonic = 3
        self.binFund = 10
        
        self.totalNoisePwr = 0
        self.noisePwrPerBin = 0
        
        self.__fundamentalTonePwrDB = 0
        self.__fundamentalToneFreqHz = 0

        self.__indexFund = 0
        self.__indexHarm = 0
        self.__indexDC = 0
        self.__indexDCWithBin = list(range(7))
        self.__indexFundWithBin = []
        self.__indexHarmWithBin = []
        
    def realFFTTransform(self, data, fftsize = 0, window = 'blackmanharris', plot = False):
        '''
        Do raw FFT transfrom to real data, return frequency and raw fft complex values 
        Parameters
        ----------
        data : list
            Numbers to be FFT transformed
        plot : BOOL, optional
            If plot the FFT result. The default is False.
        format : 'linear' or 'db', optional
            Decide the FFT return the value in linear unit or dB. The default is 'db'.

        Returns
        -------
        f_Hz : list
            frequency in Hz.
        s : list
            The raw FFT result.
        '''
        datalen = len(data)
        n = datalen if fftsize == 0 else int(fftsize)
        #f = np.linspace(0, int(self.fs // 2), int(n // 2) + 1)
        f = np.fft.rfftfreq(n, d=1 / self.fs) # 生成频率轴
        dcindex = np.argmin(f)
        try:
            WINLEN = 128
            windows = signal.get_window(window, n)
            data[:WINLEN//2]  *= windows[:WINLEN//2]
            data[-WINLEN//2:] *= windows[WINLEN//2:]
        except:
            print('No valid window type found: {}'.format(window))
        
        s = np.fft.rfft(data, n) / n * 2
        s[dcindex] *= .5
        print("FFTSIZE = {}, Freq_LEN = {}, DATA_LEN = {}, FFT_RESULT_LEN = {}".format(n,len(f),datalen, len(s)))
        if plot == True:
            plt.figure(figsize = (12,9))
            plt.plot(f/1000, np.abs(s))
            plt.xlabel('Frequency (kHz)')
            plt.ylabel('Amplitude')
            plt.grid()
            plt.show()
        
        self.freqHz, self.rawFFTData = f,s
        return f,s
    
    def realFFTSpectrum(self, data, fftsize = 0, window = 'blackmanharris', plot = False, format = 'db'):
        f,s = self.realFFTTransform(data, fftsize, window, False)
        s = np.abs(s)
        self.SpectrumDataDB = 20*np.log10(np.clip(s,1e-32,1e100))
        self.DcPwrDB = self.SpectrumDataDB[0]
        if format == 'db':    s = self.SpectrumDataDB
        
        if plot == True:
            plt.figure(figsize = (12,9))
            plt.plot(f/1000, s)
            plt.xlabel('Frequency (kHz)')
            if format == 'db':    plt.ylabel('Amplitude (dBFs)')
            else:                 plt.ylabel('Amplitude')
            plt.grid()
            plt.show()   
    
        self.genFFTAnalysis()
        return f, s
    
    def binPwr(self, singleSideWidth = 3, centralIndex = 0):
        if self.noisePwrPerBin == 0:    self.averageBinNoise()
        peakindex_range = [i for i in np.arange(-singleSideWidth,singleSideWidth+1)+centralIndex] if centralIndex >= singleSideWidth else list(range(centralIndex + singleSideWidth)) 
        peak = 0
        for i in peakindex_range: peak += np.abs(self.rawFFTData[i])**2 
        return 10*np.log10(np.clip(peak - self.noisePwrPerBin*(2*singleSideWidth+1),1e-32,1e100))  
        
    def loadData(self,filename):
        dataarray = pd.read_csv(filename, header=None).to_numpy()
        return dataarray.T.tolist()[0]
        
    def exportToVisualAnalog(self, data, filename_txt):
        np.savetxt(filename_txt, data, fmt='%.10f') 
    
    def dcPowerMeasure(self):
        return np.argmin(self.freqHz), self.SpectrumDataDB[np.argmin(self.freqHz)]
    
    def genFFTAnalysis(self, binFund = -1, binHarm = -1):
        if binFund >=0:    self.binFund     = binFund
        if binHarm >=0:    self.binHarmonic = binHarm
        peak = max(self.SpectrumDataDB[2:])
        self.__indexFund = int(self.SpectrumDataDB.index(peak)) if type(self.SpectrumDataDB) == list  else int(np.argmax(self.SpectrumDataDB[2:])) + 2 
        self.__indexFundWithBin = [int(i) for i in np.arange(-self.binFund,self.binFund+1)+self.__indexFund] if self.__indexFund >= self.binFund else list(range(self.__indexFund + self.binFund))
        freqHz = self.freqHz.tolist()
        fFoundHz = freqHz[self.__indexFund]
        freqs = [i*fFoundHz for i in range(1,12+1)]   #12nd harmonic
        NYQZONE = self.fs / 2
        for i in range(1,len(freqs)):
            n = freqs[i] // NYQZONE
            if n % 2 == 0:    freqs[i] %= NYQZONE
            else:             freqs[i] = NYQZONE - (freqs[i] % NYQZONE)
        self.__indexHarm = [freqHz.index(freqs[i]) for i in range(1,len(freqs))]
        for index in self.__indexHarm:
            temp = [int(i) for i in np.arange(-self.binHarmonic,self.binHarmonic+1)+index] if index >= self.binHarmonic else list(range(index + self.binHarmonic)) 
            self.__indexHarmWithBin.append(temp)
    
    def fundPowerSingleTone(self):
        self.__fundamentalTonePwrDB = self.binPwr(singleSideWidth = self.binFund, centralIndex = self.__indexFund)
        return self.freqHz[self.__indexFund], self.__fundamentalTonePwrDB
    
    def harmonicMeasure(self,highestOrder = 6):
        fFundHz, peakDB = self.fundPowerSingleTone() 
        freqHz = self.freqHz.tolist()
        harmonics = [peakDB] + [self.binPwr(singleSideWidth = self.binHarmonic, centralIndex = i) for i in self.__indexHarm[:highestOrder]]
        return [fFundHz] + [freqHz[i] for i in self.__indexHarm[:highestOrder]], harmonics
   
    def totalHarmonicPwr(self, highestOrder = 6):
        f, harmonics = self.harmonicMeasure(highestOrder)
        return 10*np.log10( np.sum(10**(np.array(harmonics[1:]) / 10)) )
        
    def sfdr(self, withHarmoics = True, isdBFS = False):
        '''
        Spurious free dynamic range (SFDR) is the ratio of the rms value of the signal to the rms value
        of the worst spurious signal regardless of where it falls in the frequency spectrum. The worst
        spur may or may not be a harmonic of the original signal. SFDR is an important specification in
        communications systems because it represents the smallest value of signal that can be distinguished 
        from a large interfering signal (blocker). SFDR can be specified with respect to full-scale (dBFS)
        or with respect to the actual signal amplitude (dBc). 
        
        reference：https://www.analog.com/media/cn/training-seminars/tutorials/MT-003_cn.pdf
        '''
        f, harmonics = self.harmonicMeasure(6)
        if withHarmoics is True:    
            return harmonics[0] - max(harmonics[1:]) if isdBFS == True else -max(harmonics[1:])
        freqListHz = self.freqHz.tolist()
        harmoicIndex_s = [freqListHz.index(i) for i in f]
        
        SpectrumDataDB_s = self.SpectrumDataDB.tolist()
        for i in harmoicIndex_s:    SpectrumDataDB_s[i] = min(self.SpectrumDataDB)
        sortedSpectrumDataDB_s = SpectrumDataDB_s[1:].sort(reverse = True)
        return sortedSpectrumDataDB_s[0] - sortedSpectrumDataDB_s[1]
            
    def thd(self, highestOrder = 6):
        '''
        Total harmonic distortion (THD) is the ratio of the rms value of the fundamental signal to the
        mean value of the root-sum-square of its harmonics (generally, only the first 5 harmonics are
        significant). THD of an ADC is also generally specified with the input signal close to full-scale,
        although it can be specified at any level
        
        reference：https://www.analog.com/media/cn/training-seminars/tutorials/MT-003_cn.pdf
        '''
        totalDistortion = self.totalHarmonicPwr(highestOrder)
        return totalDistortion - self.__fundamentalTonePwrDB  
    
    def thd_add_N(self):
        '''
        Total harmonic distortion plus noise (THD + N) is the ratio of the rms value of the fundamental
        signal to the mean value of the root-sum-square of its harmonics plus all noise components
        (excluding dc). The bandwidth over which the noise is measured must be specified. In the case of
        an FFT, the bandwidth is dc to fs/2. (If the bandwidth of the measurement is dc to fs/2 (the
        Nyquist bandwidth), THD + N is equal to SINAD). Be warned, however, that in audio applications 
        the measurement bandwidth may not necessarily be the Nyquist bandwidth.
        
        reference：https://www.analog.com/media/cn/training-seminars/tutorials/MT-003_cn.pdf
        '''
        pwr_total_s = np.abs(self.rawFFTData)**2
        pwr_total_withoutDC = np.sum( pwr_total_s[len(self.__indexDCWithBin):] )  #remove DC power
        pwr_tone = self.fundPowerSingleTone()[1]
        pwr_tone = 10**(pwr_tone/10)
        return 10*np.log10( pwr_total_withoutDC - pwr_tone )
    
    def sinad(self):
        #pwr_time = np.sum(np.array(data)**2) / len(data)
        pwr_total_s = np.abs(self.rawFFTData)**2
        pwr_total_withoutDC = np.sum( pwr_total_s[len(self.__indexDCWithBin):] )   #remove DC power
        pwr_tone = self.fundPowerSingleTone()[1]
        pwr_tone = 10**(pwr_tone/10)
        #print('Totoal Freq Domain Power without DC = {}, Signal Power = {}'.format(pwr_total_withoutDC,pwr_tone))
        sinad = 10*np.log10(pwr_tone / (pwr_total_withoutDC - pwr_tone))
        #print(pwr_time,pwr_freq,pwr_signal,sinad)
        return sinad
        
    def enob(self):
        return np.round((self.sinad() - 1.76) / 6.02, 2)
    
    def snr(self, isDC = False, highestOrder = 9):
        '''
        Signal-to-noise ratio (SNR, or sometimes called SNR-without-harmonics) is calculated from the
        FFT data the same as SINAD, except that the signal harmonics are excluded from the
        calculation, leaving only the noise terms. In practice, it is only necessary to exclude the first 5
        harmonics, since they dominate. The SNR plot will degrade at high input frequencies, but
        generally not as rapidly as SINAD because of the exclusion of the harmonic terms.
        
        reference：https://www.analog.com/media/cn/training-seminars/tutorials/MT-003_cn.pdf / Eq. 12
        '''
        if isDC == False:    return -10*np.log10( 10**(-self.sinad() / 10) - 10**( self.thd(highestOrder) / 10) )
        pwr_total_s = np.abs(self.rawFFTData)**2
        pwr_total_withoutDC = np.sum( pwr_total_s[len(self.__indexDCWithBin):] )      #remove DC power 
        return -10*np.log10(np.clip(pwr_total_withoutDC,1e-32,1e100))
    
    def snrFS(self, isDC = False, highestOrder = 9):
        if isDC == False:     return self.snr(isDC, highestOrder) - self.__fundamentalTonePwrDB
        else:    return self.snr(isDC, highestOrder)
    
    def noisefloor(self, isDC = False):
        '''
        reference: https://www.analog.com/cn/technical-articles/noise-spectral-density.html
        '''
        if isDC == False:    return 10*np.log10(self.totalNoisePwr / (self.fs / 2))
        return -self.snr(isDC) - 10*np.log10(self.fs / 2)
    
    def averageBinNoise(self):
        self.totalNoisePwr = 0
        index_totalHarmWithBin = []
        for i in self.__indexHarmWithBin:    index_totalHarmWithBin.extend(i)
        #print(self.__indexDCWithBin, self.__indexFundWithBin, index_totalHarmWithBin)
        index_DC_Fund_Harm = list(set(self.__indexDCWithBin + self.__indexFundWithBin + index_totalHarmWithBin))
        for i in range(len(self.rawFFTData)):
            if i not in index_DC_Fund_Harm:    self.totalNoisePwr += np.abs(self.rawFFTData[i])**2
        self.noisePwrPerBin = self.totalNoisePwr / (len(self.rawFFTData) - len(index_DC_Fund_Harm))
        return 10*np.log10(self.noisePwrPerBin)    
    
        
    def noiseFreeAndEffectiveResolution(self, rawSamplesUnderDC, adcWidth):
        datas = np.array(rawSamplesUnderDC)
        mean, var, std, vpp = datas.mean(), datas.var(), datas.std(), datas.max()-datas.min()
        nf_resolution, eff_resolustion = np.log2(2**adcWidth / vpp), np.log2(2**adcWidth / std)
        print('Average = {}, Var = {}, LSBPP = {}'.format(mean,var,vpp))
        print('ER = {}, NFR = {}'.format(eff_resolustion, nf_resolution))
        return [nf_resolution, eff_resolustion]
        
'''
if __name__ == '__main__':
    ofdm = OFDM(num_subcarriers = 1024, num_pilots = 8, pilot_val = 3+3j, qam_order = 16)
    bits = np.random.binomial(n=1, p=0.5, size=(ofdm.payloadBits_per_OFDM-5, ))
    bits_SP = ofdm.SP(bits)

    QAM = ofdm.Mapping(bits_SP)

    OFDM_data = ofdm.OFDM_symbol(QAM)
    print ("Number of OFDM carriers in frequency domain: ", len(OFDM_data))
    OFDM_time = ofdm.IDFT(OFDM_data)
    print ("Number of OFDM samples in time-domain before CP: ", len(OFDM_time))
    OFDM_withCP = ofdm.addCP(OFDM_time, True)
    print ("Number of OFDM samples in time domain with CP: ", len(OFDM_withCP))
    OFDM_TX = OFDM_withCP
    OFDM_RX = ofdm.channel(OFDM_TX,45)
    #OFDM_RX = OFDM_TX
    plt.figure(figsize=(8,2))
    plt.plot(abs(OFDM_TX), label='TX signal')
    plt.plot(abs(OFDM_RX), label='RX signal')
    plt.legend(fontsize=10)
    plt.xlabel('Time'); plt.ylabel('$|x(t)|$');
    plt.grid(True)
    OFDM_RX_noCP = ofdm.removeCP(OFDM_RX)
    print(OFDM_RX_noCP.shape)
    OFDM_demod = ofdm.DFT(OFDM_RX_noCP)
    print(OFDM_demod.shape)
    Hest = ofdm.channelEstimate(OFDM_demod)
    equalized_Hest = ofdm.equalize(OFDM_demod, Hest)
    QAM_est = ofdm.get_payload(equalized_Hest)
    PS_est, hardDecision = ofdm.Demapping(QAM_est)
'''