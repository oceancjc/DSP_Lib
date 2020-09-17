# -*- coding: utf-8 -*-
"""
Created on Thu Feb 04 09:58:53 2016

@author: jchen3

Last edit:   20170331    ignore_warning max_cnt for screen capture
"""
import visa,pyvisa
import time,math,sys,ctypes,os,csv

def frange(a,b,s=1.0):
    ret = [a]
    n = 1
    if s > 0:
        while a+n*s < b:
            ret.append(a+n*s)
            n+=1
    elif s < 0:
        while a + n*s > b:
            ret.append(a+n*s)
            n+=1
    return map(lambda x: round(x*1e15)/1.0e15,ret)

      
        
def get_input(string):
    while True:    
        try:
            return input(string) 
        except:
            print('Please Enter Correct number!')

rm = visa.ResourceManager()
rm.ignore_warning(pyvisa.constants.VI_SUCCESS_MAX_CNT)
rm.timeout = 2
print(rm.list_resources())
vpp43 = rm.visalib
session,state=vpp43.open_default_resource_manager()

def close_resourcemanager():
    global rm
    rm.close()

class E5052B_device(object):
    def __init__(self,ip='None'):
        global rm,session        
        print('Step 1\t Connecting E5052B ...\n')
        try:  
            if ip == 'None':
                find_list,return_counter,instrument_description_E5052B,state=vpp43.find_resources(session,r'TCPIP?*::?*5052?*::inst0::INSTR')
                self.E5052B = rm.open_resource(instrument_description_E5052B)
            else:
                self.E5052B = rm.open_resource(ip)

            print('[Info]\t E5052B Connected')
            self.device_found = 0
            self.reset()
        except:
            self.device_found = -1
            print('[ERR]\t E5052B not found \n')     
            
    def reset(self):
        self.E5052B.write(r'*RST')
        self.E5052B.write(r'DISPlay:WINDow:ACTive PN1')
        self.E5052B.write(r'DISPlay:PN1:STATe ON')
        self.E5052B.write(r'DISPlay:MAXimize ON')  
        
    def save_screen(self, in_f, re_f):
        pic = self.E5052B.query_binary_values(r'MMEM:TRAN? "%s"' %in_f, datatype=u'b')
        f = open(re_f,'ab')
        for i in pic:
            #f.write(chr(ctypes.c_ubyte(i).value))
            f.write(ctypes.c_ubyte(i).value)
        f.close()
        



class SMA100_device(object):
    def __init__(self,ip):
        global rm,session    
        print('Step 1\t Connecting SMA100 ...\n')
        ip = 'TCPIP0::'+ip+'::inst0::INSTR'
        try:  

            self.SMA = rm.open_resource(ip)
            print('[Info]\t SMA100 Connect Success ...')
            self.device_found = 0
            self.reset()
            self.SMA.timeout = 10000
        except:
            self.device_found = -1
            print('[ERR]\t SMA100 not found \n')
            
    def reset(self):
        self.SMA.write(r'SYST:PRES')
        self.SMA.write(r'OUTP OFF')
        
        
    def set_single_tone_MHz(self,freqMHz,pwrdBm= 88):
        if pwrdBm < 88:
            self.SMA.write(r'SOUR:POW:POW %f' %pwrdBm)
        self.SMA.write(r'FREQ %fMHz' %freqMHz)
        self.SMA.write(r'FREQ:MODE FIXed')
        self.SMA.write(r'OUTP ON')
        time.sleep(0.3)
        
    def output_on_off(self,on = 1):
        if on == 1:
            self.SMA.write(r'OUTP ON')
        else:
            self.SMA.write(r'OUTP OFF')
            
    def swap_freq_amp(self, freqrangeMHz, amprangedBm):
        for f in freqrangeMHz:
            for a in amprangedBm:
                yield self.set_single_tone_MHz(f,a)
            
            
            
        
class N9030A_device(object):
    def __init__(self,ip):
        global rm,session        
        print('Step 1\t Connecting N9030A ...\n')
        self.DEVIP = ip
        try:  
            self.N9030A = rm.open_resource('TCPIP0::%s::inst0::INSTR' %ip)
            print('[Info]\t N9030A Connect Success ...')
            self.device_found = 0
            self.N9030A.timeout = 8000
            #self.preset()
        except:
            self.device_found = -1
            print('[ERR]\t N9030A not found \n')   
        
    def open_device(self):
        try:  
            self.N9030A = rm.open_resource('TCPIP0::%s::inst0::INSTR' %self.DEVIP)
            print('[Info]\t N9030A Connect Success ...')
            self.device_found = 0
#            self.reset()
        except:
            self.device_found = -1
            print('[ERR]\t N9030A not found \n')   
        
    def preset(self):
        self.N9030A.write(r'INST:NSEL 1')
        self.N9030A.write(r'SYST:PRES')
        self.N9030A.write(r'INIT:SAN')
        
        
    def enable_overloadwarning(self,stat = 1):
        if stat > 0:     self.N9030A.write(r'SYST:ERR:OVER 1')
        else:            self.N9030A.write(r'SYST:ERR:OVER 0')
        
        
    def set_sweptfreqMode(self):
        self.N9030A.write(r'INIT:SAN')
        
    def set_chpowerMode(self,averagenum=0 ):
        self.N9030A.write(r'CONFigure:CHPower')
        time.sleep(1)
        if averagenum <= 0:
            self.N9030A.write('CHP:AVER OFF')
        else:
            self.N9030A.write('CHP:AVER:COUN %d' %averagenum)
            self.N9030A.write('CHP:AVER ON')
        self.N9030A.write('CHP:FREQ:SPAN:AUTO ON')
    
    
    def set_chpowerspan_MHz(self,span):
        if span > 0:     self.N9030A.write('CHP:FREQ:SPAN %fMHz' %span)
        elif span <0:    self.N9030A.write('CHP:FREQ:SPAN FULL')
        else:            self.N9030A.write('CHP:FREQ:SPAN:AUTO ON')
        self.N9030A.query_ascii_values(r'*OPC?')

        
        
        
    def set_chpowerAverage(self,num=0):
        if num <= 0:   self.N9030A.write('CHP:AVER OFF')
        else:
            self.N9030A.write('CHP:AVER:COUN %d' %num)
            self.N9030A.write('CHP:AVER ON')      
            
         
    def set_chpowerInteBand_MHz(self,band):
        self.N9030A.write('CHP:BAND:INT %dMHz' %band)
        
        
    def set_chpowerRBW_KHz(self,rbw = 0):
        if rbw == 0:
            self.N9030A.write('CHP:BAND:AUTO ON')
        else:
            self.N9030A.write('CHP:BAND: %fKHz' %rbw) 


    def set_chpowerVBW_KHz(self,vbw = 0):
        if vbw == 0:
            self.N9030A.write('CHP:BAND:VID:AUTO  ON')
        else:
            self.N9030A.write('CHP:BAND:VID: %fKHz' %vbw) 
            
    def get_chpower(self):
        return self.N9030A.query_ascii_values(r'READ:CHP:CHP?')[0]
        
    def set_cent_freq_Hz(self,cent = 1000 ):
        cmd = r'FREQ:CENT %fHz'
        self.N9030A.write(cmd %cent)
        self.N9030A.query_ascii_values(r'*OPC?')
       
    def set_span_MHz(self,band = 50, unit = 'MHz'):
        cmd = r'FREQ:SPAN %f'+unit
        self.N9030A.write(cmd %band)
        self.N9030A.write('CALC:MARK1:FUNC:BAND:SPAN %f %s' %(band,unit))
        self.N9030A.query_ascii_values(r'*OPC?')
        
    def set_freq_span_MHz(self,freq = 1000, band = 50):
        self.set_cent_freq_Hz(freq*1e6)
        self.set_span_MHz(band)
        self.N9030A.query_ascii_values(r'*OPC?')
     
    def set_vbw_kHz(self,vbw):
        cmd = r'BAND %fkHz' %vbw
        self.N9030A.write(cmd)    

    def set_rbw_kHz(self,rbw):
        if rbw == 0:
            self.N9030A.write('BWID:AUTO ON')
        else:
            self.N9030A.write(r'BAND %fkHz' %rbw) 

    def set_marktable(self,state = 1):
        if state == 1:  self.N9030A.write('CALC:MARK:TABL 1')
        else:           self.N9030A.write('CALC:MARK:TABL 0')
        
    def set_ref_level(self, reflvldBm):
        self.N9030A.write(r'CORR:NOIS:FLO OFF')
        self.N9030A.write('DISP:WIND1:TRAC:Y:RLEV {}DBM'.format(reflvldBm))
        self.N9030A.query_ascii_values(r'*OPC?')

    def set_mechAtt(self,att):
        self.N9030A.write('POW:ATT %d' %att)
        self.N9030A.query_ascii_values(r'*OPC?')
    
    def set_averagetime(self,time):
        self.N9030A.write('AVER:COUN %d' %time)
        
        
    def trace_average(self):
        self.N9030A.write(r'TRACe:TYPE AVERage')
        self.N9030A.write(r'INIT:IMM')
            
    def trace_immediate(self):
        self.N9030A.write(r'TRACe:TYPE WRITe')   
        
    def noise_reduction(self,state=1):
        if state > 0:     self.N9030A.write(r'CORR:NOIS:FLO ON')
        else:             self.N9030A.write(r'CORR:NOIS:FLO OFF')
        

    def peakSearch(self,marknum):
        self.N9030A.write(r'CALC:MARK%d:MAX' %marknum)
    
    def get_markPower_dBm(self,mark):
        return self.N9030A.query_ascii_values(r'CALC:MARK%d:Y?' %mark)[0]

    def get_freq_value_MHz_dBm(self,trace_average_need = 0,show=False):
        if trace_average_need !=0:
            self.trace_average()
            time.sleep(3)
        
        self.N9030A.write(r'CALC:MARK1:MAX')
        self.N9030A.query_ascii_values(r'*OPC?')
        y = self.N9030A.query_ascii_values(r'CALC:MARK1:Y?')
        x = self.N9030A.query_ascii_values(r'CALC:MARK1:X?')
        x = x[0]/1e6
        if show == True:    print('Freq = %fMHz , Amp = %fdBm' %(x,y[0]))

        if trace_average_need !=0:
            self.trace_immediate()
        return x,y[0]
    
    def get_EVM(self):
        s = self.N9030A.query(r'FETC:EVM?')
        return s.split(',')
        
    def get_TOI_value(self,freq1,freq2,unit='MHz'):
        '''
        Returns 12 scalar results, in the following order.

        1. The worst case Output Intermod Point value in Hz.

        2. The worst case Output Intermod Power value in dBm

        3. The worst case Output Intercept Power value in dBm

        4. The lower base frequency value in Hz

        5. The lower base power value in dBm
        
        6. The upper base frequency value in Hz

        7. The upper base power value in dBm

        8. The lower Output Intermod Point in Hz

        9. The lower Output Intermod Power value in dBm

        10. The lower Output Intercept Power value in dBm

        11. The upper Output Intermod Point in Hz

        12. The upper Output Intermod Power value in dBm

        13. The upper Output Intercept Power value in dBm

        '''
        
        self.N9030A.write('CONFigure:TOI')
        self.set_cent_freq_Hz( (freq1+freq2)/2.*1e6 )
        band = 5*math.fabs(freq2-freq1)
        self.N9030A.write(r'SENSe:TOI:FREQuency:SPAN %fMHz' %band)
        self.N9030A.query_ascii_values(r'*OPC?')
        self.N9030A.write(r'TOI:BWID %f kHz' %band)
        self.N9030A.query_ascii_values(r'*OPC?')
        self.N9030A.write(r'SENSe:TOI:AVERage:STATe ON')
        self.N9030A.query_ascii_values(r'*OPC?')
        time.sleep(5)
        #self.N9030A.write(r'INIT:CONT 0')
        results = self.N9030A.query_ascii_values(r'FETCh:TOI2?')
        #self.N9030A.write(r'INIT:CONT 1')
        return results
    
    def get_ip3(self,freq1,freq2,unit='MHz'):
        self.set_cent_freq_Hz( (freq1+freq2)/2.*1e6 )
        #self.N9030A.write('SENSe:POWer:RF:ATTenuation 0') 
        self.N9030A.write('CONFigure:TOI')
        time.sleep(1)
        band = 6*math.fabs(freq2-freq1)
        self.N9030A.write(r'SENSe:TOI:FREQuency:SPAN %fMHz' %band)
        time.sleep(0.5)
        self.N9030A.write(r'TOI:BWID 15 kHz')
        print('bandwidth =',band)
        ip3 = self.N9030A.query_ascii_values(r'FETCh:TOI:IP3?')
        return ip3
        
    def get_noise_floor(self):
        self.N9030A.write(r'CONFigure:CHPower')
        time.sleep(1)    
        self.N9030A.write(r'SENSe:POWer:RF:ATTenuation 0')    
        self.N9030A.write(r'SENSe:POWer:RF:GAIN:STATe ON')
        self.N9030A.write(r'SENSe:POWer:RF:GAIN:BAND LOW')
        self.N9030A.query_ascii_values(r'*OPC?')
        time.sleep(0.5)
        noise_floor = self.N9030A.query_ascii_values(r'MEASure:CHPower:DENSity?')
        print('noise_floor =',noise_floor[0])
        time.sleep(0.5)
        noise_floor = min(noise_floor[0], self.N9030A.query_ascii_values(r'MEASure:CHPower:DENSity?')[0])
        return noise_floor

    
    def get_startfreq(self):
        return self.N9030A.query_ascii_values(r'FREQ:STAR?')[0]
    
    
    def get_peaks(self,freq_startMHz = 0, freq_stopMHz = 0,spanMHz = 50, capPath = ''):
        self.trace_average()
        self.N9030A.write('CALC:MARK:TABL 0')
        self.N9030A.write('CALC:MARK:TABL 1')
        xs = []
        ys = []
        if freq_startMHz == 0 or freq_stopMHz == 0:
            time.sleep(3)
            num = int(self.N9030A.query('TRACe:MATH:PEAK:POINts?'))
            print('current window num of peak =',num)
            if num != 0:
                time.sleep(3)
                self.N9030A.write('CALC:MARK:AOFF')
                for i in range(1,num+1):
                    for j in range(i):         self.N9030A.write('CALC:MARK%d:MAX:NEXT' %i)
                    self.N9030A.query_ascii_values(r'*OPC?')
                    xs.append( self.N9030A.query_ascii_values(r'CALC:MARK%d:X?' %i)[0] / 1e6 )
                    ys.append( self.N9030A.query_ascii_values(r'CALC:MARK%d:Y?' %i)[0] )
            
            if capPath != '':
                time.sleep(1)
                freq_startMHz = self.get_startfreq()/1e6
                self.save_screen_png(capPath+'Cap'+str(freq_startMHz)+'MHz.png')
            
        else:
            for centf in frange(freq_startMHz+spanMHz/2,freq_stopMHz,spanMHz):
                self.set_freq_span_MHz(centf,spanMHz)
                time.sleep(5)
                num = int(self.N9030A.query('TRACe:MATH:PEAK:POINts?'))
                print(num)
                self.N9030A.write('CALC:MARK:AOFF')
                if num == 0:        continue
                for i in range(1,num+1):
                    for j in range(i):         self.N9030A.write('CALC:MARK%d:MAX:NEXT' %i)
                    self.N9030A.query_ascii_values(r'*OPC?')
                    xs.append( self.N9030A.query_ascii_values(r'CALC:MARK%d:X?' %i)[0] / 1e6 )
                    ys.append( self.N9030A.query_ascii_values(r'CALC:MARK%d:Y?' %i)[0] )
                if capPath != '':
                    time.sleep(1)
                    freq_startMHz = int(self.get_startfreq()/1e6)
                    self.save_screen_png(capPath+'\Cap'+str(freq_startMHz)+'MHz.png')            
        return xs, ys
    
        
    def save_screen_png(self,file_name):
        self.N9030A.write(r'MMEM:STOR:SCR "D:\\Temp.png"')
        f = open(file_name,'wb')
        b = self.N9030A.query_binary_values(r'MMEM:DATA? "D:\\Temp.png"',datatype=u'b')
        for i in b:
            f.write(chr(ctypes.c_ubyte(i).value))
        f.close()

		
    def recalltrace(self,traceID):
        self.N9030A.write(r'*RCL %d' %traceID)	
        
		
    def setEVMformat(self,DB_ON = False):
        if DB_ON == False:    self.N9030A.write(r'EVM:REP:DB OFF')	
        else:                 self.N9030A.write(r'EVM:REP:DB ON')	
        self.N9030A.write(r'FORM:TRAC:DATA ASCii,32')

		
    def fetchEVM(self):
        return self.N9030A.query(r'FETC:EVM?')
    
    def initACPR(self,carrier_cnt = 1, loMHz = None, spanMHz = None):
        self.N9030A.write('INIT:ACP')
        if int(carrier_cnt) < 1:    carrier_cnt = 1
        self.N9030A.query_ascii_values(r'*OPC?')
        self.N9030A.write('ACP:CARR:COUN {}'.format(int(carrier_cnt)))
        if loMHz != None:    self.N9030A.write(r'FREQ:CENT {}Hz'.format(loMHz*1e6))
        if spanMHz != None:  self.N9030A.write(r'ACP:FREQ:SPAN {}Hz'.format(spanMHz*1e6))
        

    def setACPRCarrier(self, carrier_number, freqFromLOMHz, bandwidthMHz, spacingMHz):
        #self.N9030A.write()
        self.N9030A.write('ACP:CARR{}:LIST:BAND {}MHz'.format(int(carrier_number),float(bandwidthMHz)))
        self.N9030A.write('ACP:CARR{}:LIST:WIDT {}MHz'.format(int(carrier_number),float(spacingMHz)))
        
    def setACPROffsets(self, offsetFreqMHz = [], offsetBandwidthMHz = [], offsetSpacingMHz = []):
        if len(offsetFreqMHz) != len(offsetBandwidthMHz) or  len(offsetFreqMHz) != len(offsetSpacingMHz):
            print('Input length mismatch')
        else:
            cmd = [str(float(i))+'MHz' for i in offsetFreqMHz]
            cmd = ', '.join(cmd)
            self.N9030A.write('ACP:OFFS1:LIST '+cmd)
            cmd = [str(float(i))+'MHz' for i in offsetBandwidthMHz]
            cmd = ', '.join(cmd)
            self.N9030A.write('ACP:OFFS1:LIST:BAND '+cmd) 
            cmd = ['0']*6
            for i in range(len(offsetFreqMHz)):    cmd[i] = '1'
            cmd = ', '.join(cmd)            
            self.N9030A.write('ACP:OFFS1:LIST:STAT '+cmd)
            '''
            cmd = [str(float(i))+'MHz' for i in offsetSpacingMHz]
            cmd = ', '.join(cmd)
            self.N9030A.write('ACP:OFFS1:LIST:BAND '+cmd)  
            '''
            
    def getACPR(self):
        offsets = self.N9030A.query('ACP:OFFS1:LIST:STAT?')[:-1].split(',')
        r = self.N9030A.query('READ:ACP?')[:-1].split(',')
        dicr = {}
        dicr['TotalCarrierPWR(dBm)'] = float(r[1])
        for i in range(len(offsets)):
            if offsets[i] == '1':    
                dicr['LowerOffset{}RelPWR(dBc)'.format(i+1)] = float(r[4+i*4])
                dicr['LowerOffset{}AbsPWR(dBm)'.format(i+1)] = float(r[5+i*4])
                dicr['UpperOffset{}RelPWR(dBc)'.format(i+1)] = float(r[6+i*4])
                dicr['UpperOffset{}AbsPWR(dBm)'.format(i+1)] = float(r[7+i*4])
        return dicr
            
    def sendCmd(self,string):
        self.N9030A.write(string)
    
    def retrieve(self,string):
        return self.N9030A.query(string)
        
    def disconnect(self):
        self.N9030A.clear(session)
        

class E4438_device(object):
    def __init__(self,ip):
        global rm,session        
        print('Step 1\t Connecting E4438 ...\n')
        self.DEVIP = ip
        try:  
            self.E4438 = rm.open_resource('TCPIP0::%s::INSTR' %ip)
            print('[Info]\t E4438 Connect Success ...')
            self.device_found = 0
            self.E4438.timeout = 8000
        except:
            self.device_found = -1
            print('[ERR]\t E4438 not found \n')   

    def set_single_tone_MHz(self,freqMHz,pwrdBm= 88):
        if pwrdBm < 88:        self.E4438.write(r'SOUR:POW %f' %pwrdBm)
        self.E4438.write(r'FREQ %fMhz' %freqMHz)
        self.E4438.write(r'FREQ:MODE FIXed')
        self.E4438.write(r'OUTP ON')
        
    def output_on_off(self,enable = 1):
        if enable == 1:
            self.E4438.write(r'OUTP ON')
        else:
            self.E4438.write(r'OUTP OFF')     
            
    def set_multitone_MHz(self,num,loMHz,spacingMHz,ampdBm):
        self.E4438.write(r'FREQ %fMhz' %loMHz)
        self.E4438.write(r'SOUR:POW %f' %ampdBm)
        self.E4438.write(r'RADio:MTONe:ARB:SETup:TABLe:FSPacing %fMhz' %spacingMHz)
        self.E4438.write(r'RADio:MTONe:ARB:SETup:TABLe:NTONes %d' %num)
        self.E4438.write(r':RADio:MTONe:ARB:SETup:TABLe:PHASe:INITialize FIXed')
        self.E4438.write(r'OUTPut:MODulation ON')
            
    def disconnect(self):
        self.E4438.close()


class PWR_device(object):
    def __init__(self):        
        global rm,session       
        print('Step 1\t Connecting PWR ...\n')
        try:  
            find_list,return_counter,instrument_description,state=vpp43.find_resources(session,r'GPIB0::?*::INSTR')
            self.pwr = rm.open_resource(instrument_description)
#            self.pwr = rm.open_resource('GPIB0::5::INSTR')
            self.reset()
            print('[Info]\t POWER SOURCE Connected')
            self.device_found = 0
        except:
            self.device_found = -1
            print('[ERR]\t POWER SOURCE not found \n')  
            
    def reset(self):
        self.pwr.write('*RST')
        
    def output(self,stat):
        '''
        stat == 1 if on, 0 if off
        '''
        self.pwr.write('OUTP %d' %stat)
        
    def config_channel(self,channel,vol,veri=0):
        import math
        self.pwr.write('INST:NSEL %d' %channel)
        if vol < 8:
            self.pwr.write('VOLT:RANG LOW')
        else:
            self.pwr.write('VOLT:RANG HIGH')
        time.sleep(2)
        self.pwr.write(':SOUR:VOLT:LEV:IMM:AMPL %f' %vol) 
        time.sleep(2)
        self.pwr.write('OUTP ON')
        if veri:
            val = self.pwr.query_ascii_values('MEAS:VOLT?')[0]
            if(math.fabs(val-vol)<= 0.01):
                return 0
            else:
                print('Output is %f while the setting voltage is %f, pls check' %(val,vol))
                return 1
        return 0
        
    def set_current_limit(self,channel,limit='MAX'):
        self.pwr.write('INST:NSEL %d' %channel)
        self.pwr.write('CURR %s' %limit)
        
    def set_voltage(self,channel,vol):
        self.pwr.write('INST:NSEL %d' %channel)
        self.pwr.write(':SOUR:VOLT:LEV:IMM:AMPL %f' %vol)
        
    def vol_increase(self,step):
        self.pwr.write(':SOURce:VOLTage:LEVel:IMMediate:STEP:INCRement %f' %step)
        
        
class M3458A_device(object):
    def __init__(self,ip=''):
        global rm,session    
        print('Step 1\t Connecting 3458A ...\n')
        if ip != '':    self.__addr = ip
        self.__addr = 'GPIB0::22::INSTR'
        try:  

            self.Meter = rm.open_resource(self.__addr)
            print('[Info]\t 3458A Connect Success ...')
            self.device_found = 0
            self.reset()
            self.Meter.timeout = 10000
        except:
            self.device_found = -1
            print('[ERR]\t 3458A not found \n')
            
    def reset(self):
        self.Meter.write(r'RESET')
        self.Meter.write(r'PRESET DIG;MFORMAT DINT;OFORMAT ASCII;MEM FIFO')   
        self.Meter.write('APER 2E-6;TRIG AUTO')
        self.Meter.write('DCV AUTO')
    
    def preset(self):
        self.Meter.write(r'PRESET;TRIG AUTO')
        
    def IDGet(self):
        answer = self.Meter.query('ID?')
        return answer.split('\r\n')[0]
    
    def DCvoltageMeasureEnable(self, range_V = 0, resolution = 0.00001):
        if range_V == 0:        self.Meter.write('DCV AUTO')
        else:   self.Meter.write('DCV {} {}'.format(range_V,resolution))
        
    def nplcSet(self, nplc = 1):
        self.Meter.write('NPLC {}'.format(nplc))
        
    def nplcGet(self):
        return float(self.Meter.query('NPLC?').split(',')[0])
        
    def ohmMeasureEnable(self):
        self.Meter.write('OHM')
    
    def ACvoltageMeasureEnable(self):
        self.Meter.write('ACV')
        
    def DCcurrentMeasureEnable(self):
        self.Meter.write('DCI')
        
    def ACcurrentMeasureEnable(self):
        self.Meter.write('ACI')
        
    def freqMeasureEnable(self):
        self.Meter.write("FSOURCE ACDCV")
        self.Meter.write('FREQ AUTO .0001')
    
    def autoZeroEnableSet(self, enable = False):
        if enable == False:    self.Meter.write("AZERO OFF")
        else:    self.Meter.write("AZERO ON")
    def errClear(self):
        cnt = 10
        while True:
            err = int(self.Meter.query('ERRSTR?').split(',')[0])
            if err == 0:    return err
            cnt-=1
        return err
    
    def errCheck(self):
        return int(self.Meter.query('ERRSTR?').split(',')[0])
    
    def valueRead(self):
        return float(self.Meter.read())
    
        
#import smtplib  
#from email.mime.multipart import MIMEMultipart  
#from email.mime.text import MIMEText  
#from email.mime.image import MIMEImage  
#from email.header import Header 
#
#class Email(object):
#    def __init__(self,sender = 'yhmcjc@sina.com'):
#        self.sender = sender
#        self.receiver = []        
#        self.cc_list = []
#        self.passwd = 'oceancjc19891016'
#        index = sender.find('@')
#        self.smtpserver = 'smtp.'+sender[index+1:]
#        self.username = sender
#        self.msg = MIMEMultipart()
#    
#    def edit_message(self,string):
#        self.msg.attach(MIMEText(string+"\n", 'plain', 'utf-8')) 
#        self.msg['From'] = self.sender
#        self.msg['To'] = str(self.receiver_list)
#        #self.msg['Cc'] = str(self.cc_list)
#
#    def add_subject(self,subject):
#        self.msg['Subject'] = Header(subject, 'utf-8').encode()
#        
#    def add_attach(self,file_path,file_name = None):
#        att = MIMEText(open(file_path, 'rb').read(), 'base64', 'gb2312')
#        att["Content-Type"] = 'application/octet-stream'
#        if file_name == None:
#            name = file_path.split('\\')
#            if len(name) >1:
#                file_name = name[-1]
#            else:
#                name = file_path.split('/')
#                file_name = name[-1]
#        
#        att["Content-Disposition"] = 'attachment; filename="%s"' %file_name#这里的filename可以任意写，写什么名字，邮件中显示什么名字
#        self.msg.attach(att)
#    
#    
#    def send_email(self):
#        if self.smtpserver.find('analog.com') > 0:
#            smtp = smtplib.SMTP(self.smtpserver)
##            smtp.set_debuglevel(1)
#            connect_to_exchange_as_current_user(smtp)
##            smtp.ehlo()
##            ntlm_authenticate(smtp, self.username, self.passwd) 
#        else:
#            smtp = smtplib.SMTP() 
#            smtp.connect(self.smtpserver)
#            try:
#                smtp.login(self.username, self.passwd)  
#            except:
#                print 'Login Failed, Pls check username and password'
#                return 
#        if len(self.receiver_list) !=0:
#            try:
#                smtp.sendmail(self.sender, self.receiver_list, self.msg.as_string())  
#                smtp.quit()
#            except Exception, e:
#                print e
#        else:
#            print 'Receiver List is Empty!'
#            
#from smtplib import SMTPException, SMTPAuthenticationError
#import string
#import base64
#import sspi
#
## NTLM Guide -- http://curl.haxx.se/rfc/ntlm.html
#
#SMTP_EHLO_OKAY = 250
#SMTP_AUTH_CHALLENGE = 334
#SMTP_AUTH_OKAY = 235
#
#def asbase64(msg):
#    return string.replace(base64.encodestring(msg), '\n', '')
#
#def connect_to_exchange_as_current_user(smtp):
#    """Example:
#    >>> import smtplib
#    >>> smtp = smtplib.SMTP("my.smtp.server")
#    >>> connect_to_exchange_as_current_user(smtp)
#    """
#
#    # Send the SMTP EHLO command
#    code, response = smtp.ehlo()
#    if code != SMTP_EHLO_OKAY:
#        raise SMTPException("Server did not respond as expected to EHLO command")
#
#    sspiclient = sspi.ClientAuth('NTLM')
#
#    # Generate the NTLM Type 1 message
#    sec_buffer=None
#    err, sec_buffer = sspiclient.authorize(sec_buffer)
#    ntlm_message = asbase64(sec_buffer[0].Buffer)
#
#    # Send the NTLM Type 1 message -- Authentication Request
#    code, response = smtp.docmd("AUTH", "NTLM " + ntlm_message)
#
#    # Verify the NTLM Type 2 response -- Challenge Message
#    if code != SMTP_AUTH_CHALLENGE:
#        raise SMTPException("Server did not respond as expected to NTLM negotiate message")
#
#    # Generate the NTLM Type 3 message
#    err, sec_buffer = sspiclient.authorize(base64.decodestring(response))
#
#    ntlm_message = asbase64(sec_buffer[0].Buffer)
#
#    # Send the NTLM Type 3 message -- Response Message
#    code, response = smtp.docmd("", ntlm_message)
#    if code != SMTP_AUTH_OKAY:
#        raise SMTPAuthenticationError(code, response)            
#
#
#
#from base64 import decodestring 
#from ntlm import ntlm
# 
#def ntlm_authenticate(smtp, username, password): 
#    """Example: 
#    >>> import smtplib 
#    >>> smtp = smtplib.SMTP("my.smtp.server") 
#    >>> smtp.ehlo() 
#    >>> ntlm_authenticate(smtp, r"DOMAIN\username", "password") 
#    """ 
#    code, response = smtp.docmd("AUTH", "NTLM " + asbase64(ntlm.create_NTLM_NEGOTIATE_MESSAGE(username))) 
#    if code != 334: 
#        raise SMTPException("Server did not respond as expected to NTLM negotiate message") 
#    challenge, flags = ntlm.parse_NTLM_CHALLENGE_MESSAGE(decodestring(response)) 
#    user_parts = username.split("\\", 1) 
#    code, response = smtp.docmd("", asbase64(ntlm.create_NTLM_AUTHENTICATE_MESSAGE(challenge, user_parts[1], user_parts[0], password, flags))) 
#    if code != 235: 
#        raise SMTPAuthenticationError(code, response) 
if __name__ == '__main__':
    n = M3458A_device()
    n.preset()
    # for i in range(10):
    #     print(n.valueRead())

    
    #n.set_single_tone_MHz(3570,-18.41+6.8)
    