

from __future__ import print_function
import numpy as np
#import healpy as hp 

from pixell import enmap,curvedsky,reproject

import numpy as np
import os,sys

from scipy.interpolate import interp1d
import scipy.constants as constants





def _deltaTOverTcmbToJyPerSr(freq_GHz,T0 = 2.7255):
    """
    @brief the function name is self-eplanatory
    @return the converstion factor
    """
    kB = 1.380658e-16
    h = 6.6260755e-27
    c = 29979245800.
    nu = freq_GHz*1.e9
    x = h*nu/(kB*T0)
    cNu = 2*(kB*T0)**3/(h**2*c**2)*x**4/(4*(np.sinh(x/2.))**2)
    cNu *= 1e23
    return cNu

def greyBody(freq_GHz,beta=1.4,T_d=13.6):# Gispert:1.4, 13.6  https://arxiv.org/abs/astro-ph/0005554
        freq=freq_GHz*1e9
        expT=np.e**(constants.h*freq/(constants.k*T_d))
        mu_0=freq**beta*2*constants.h*freq**3/(constants.c**2)/(expT-1)
        return mu_0




def galaticDust_SED(freq_GHz,beta_d=1.48,t_d=19.6,freq_GHz_0=150,**kwargs):
    if freq_GHz is None:
        freq_GHz = freq_GHz_0
    if "in_uk" in kwargs and kwargs['in_uk']==True:
        return greyBody(freq_GHz, beta=beta_d,T_d=t_d)/_deltaTOverTcmbToJyPerSr(freq_GHz)*2.7255e6

    else:
        return greyBody(freq_GHz, beta=beta_d,T_d=t_d)*2.7255e6


def galaticDust_Cl(mapType1,mapType2,a_d_t=3e3,a_d_e=205.,a_d_b=120.,n_d_t=-2.7,n_d_e=-2.43,n_d_b=-2.48,scalePol=1.e-2,chi_d=.3,lmax=8000,**kwargs):
    ls = np.arange(lmax)+1e-3
    if mapType1=='E' and mapType2=='E':
        cls_tmp = 1*np.abs(a_d_e)*scalePol*(ls/100.)**n_d_e/galaticDust_SED(None,in_uk=True,**kwargs)**2/100**2*2*np.pi
    elif mapType1=='B' and mapType2=='B':
        cls_tmp = 1*np.abs(a_d_b)*scalePol*(ls/100.)**n_d_b/galaticDust_SED(None,in_uk=True,**kwargs)**2/100**2*2*np.pi
    elif mapType1=='T'  and mapType2=='T':
        cls_tmp =  1*np.abs(a_d_t)*(ls/100.)**n_d_t/galaticDust_SED(None,in_uk=True,**kwargs)**2/100**2*2*np.pi
    elif mapType1 in ['E','T'] and mapType2 in ['E','T']:
        c_tt = galaticDust_Cl('T','T',a_d_t=a_d_t,a_d_e=a_d_e,a_d_b=a_d_b,n_d_t=n_d_t,n_d_e=n_d_e,lmax=lmax,scalePol=scalePol,**kwargs)
        c_ee = galaticDust_Cl('E','E',a_d_t=a_d_t,a_d_e=a_d_e,a_d_b=a_d_b,n_d_t=n_d_t,n_d_e=n_d_e,lmax=lmax,scalePol=scalePol,**kwargs)
        return np.sqrt(c_tt*c_ee)*chi_d
    else:
        cls_tmp= ls*0
    # cls_tmp[:5] = cls_tmp[5]
    cls_tmp[:2]=0
    return cls_tmp



def synchrotron_SED(freq_GHz,beta_sync=-3.1,freq_GHz_0=30,**kwargs):
    if freq_GHz is None:
        freq_GHz = freq_GHz_0
    if "in_uk" in kwargs and kwargs['in_uk']==True:
        return (freq_GHz)**(beta_sync)/_deltaTOverTcmbToJyPerSr(freq_GHz)*2.7255e6
    else:
        return (freq_GHz)**(beta_sync)*2.7255e6

def synchrotron_Cl(mapType1,mapType2,a_s_t=3.e5,a_s_e=1000.,a_s_b=500.,n_s_t=-2.7,n_s_e=-2.7,n_s_b=-2.7,scalePol=1.e-2,chi_s=.3,lmax=8000,**kwargs):
    ls = np.arange(lmax)+1e-3
    if mapType1=='E' and mapType2=='E':
        cls_tmp = 1*np.abs(a_s_e)*scalePol*(ls/100.)**n_s_e/synchrotron_SED(None,in_uk=True,**kwargs)**2/100**2*2*np.pi#/(2*np.pi)
    elif mapType1=='B' and mapType2=='B':
        cls_tmp = 1*np.abs(a_s_b)*scalePol*(ls/100.)**n_s_b/synchrotron_SED(None,in_uk=True,**kwargs)**2/100**2*2*np.pi#/(2*np.pi)
    elif mapType1=='T'  and mapType2=='T':
        cls_tmp =  1*np.abs(a_s_t)*(ls/100.)**n_s_t/synchrotron_SED(None,in_uk=True,**kwargs)**2/100**2*2*np.pi
    elif mapType1 in ['E','T'] and mapType2 in ['E','T']:
        c_tt = synchrotron_Cl('T','T',a_s_t=a_s_t,a_s_e=a_s_e,a_s_b=a_s_b,n_s_t=n_s_t,n_s_e=n_s_e,lmax=lmax,scalePol=scalePol)
        c_ee = synchrotron_Cl('E','E',a_s_t=a_s_t,a_s_e=a_s_e,a_s_b=a_s_b,n_s_t=n_s_t,n_s_b=n_s_b,lmax=lmax,scalePol=scalePol)
        return np.sqrt(c_tt*c_ee)*chi_s
    else:
        cls_tmp= ls*0
    cls_tmp[:2]=0
    return cls_tmp


def syncxdust_Cls(mapType1,mapType2,correlationCoef=-.1,synchrotron_fnc=synchrotron_Cl,galaticDust_fnc=galaticDust_Cl,lmax=8000,**kwargs):
    cl11 = np.abs(synchrotron_fnc(mapType1,mapType1,lmax,**kwargs))**.5
    cl22 = np.abs(galaticDust_fnc(mapType2,mapType2,lmax,**kwargs))**.5
    return correlationCoef*cl11*cl22


def gauss_beam(ell,fwhm_arcmin):
    tht_fwhm = np.deg2rad(fwhm_arcmin / 60.)
    return np.exp(-(tht_fwhm**2.)*(ell**2.) / (16.*np.log(2.)))




class simple_sky_model:
    def __init__(self,camb_file='./CMB_fiducial_totalCls.dat',seed=1,pixRes_arcmin=4,lmax_sim=500):



        cls_camb = np.loadtxt(camb_file,unpack=True)
        cl_tt = cls_camb[1]/(cls_camb[0]*(cls_camb[0]+1)/2/np.pi)
        cl_tt = np.append([0,0],cl_tt)
        cl_ee = cls_camb[2]/(cls_camb[0]*(cls_camb[0]+1)/2/np.pi)
        cl_ee = np.append([0,0],cl_ee)
        cl_bb = cls_camb[3]/(cls_camb[0]*(cls_camb[0]+1)/2/np.pi)*0.05
        cl_bb = np.append([0,0],cl_bb)
        cl_te = cls_camb[4]/(cls_camb[0]*(cls_camb[0]+1)/2/np.pi)
        cl_te = np.append([0,0],cl_te)
        # ells = np.append([0,1],cls_camb[0])

        self.pixRes_arcmin=pixRes_arcmin/180./60*np.pi
        self.lmax_sim = lmax_sim

        shape,wcs = enmap.fullsky_geometry(self.pixRes_arcmin)

        #pmap = enmap.enmap(shape,wcs).pixsizemap()
        opos = enmap.posmap(shape, wcs)
        galShape = 1/(np.abs(opos[0])+1e-1)
        galShape/=np.mean(galShape**2)**.5



        np.random.seed(seed)
        self.shape = shape
        self.wcs = wcs

        self.alm_cmb_T = curvedsky.rand_alm(cl_tt,lmax=lmax_sim)#hp.synalm(cl_ee,lmax=3*nside-1,new=True)
        
        tmp_empty = enmap.empty(shape,wcs)
        self.T_cmb = curvedsky.alm2map(self.alm_cmb_T,tmp_empty,spin=[0])

        ps = np.zeros([2,2,lmax_sim])
        ps[0,0] = synchrotron_Cl('T','T')[:lmax_sim]
        ps[1,1] = galaticDust_Cl('T','T')[:lmax_sim]
        ps[1,0] = syncxdust_Cls('T','T')[:lmax_sim]

        self.alm_sync_T,self.alm_dust_T = curvedsky.rand_alm(ps,lmax=lmax_sim)

    
        tmp_empty = enmap.empty(shape,wcs)
        self.T_dust= curvedsky.alm2map(self.alm_dust_T,tmp_empty,spin=[0])

        self.T_dust *=galShape


        self.alm_dust_T = curvedsky.map2alm(self.T_dust,lmax=lmax_sim)

        tmp_empty = enmap.empty(shape,wcs)
        self.T_sync= curvedsky.alm2map(self.alm_sync_T,tmp_empty,spin=[0])

        self.T_sync *=galShape

        self.alm_sync_T = curvedsky.map2alm(self.T_sync,lmax=lmax_sim)


    def observe(self,freq_GHz,noise_ukarcmin=3.,beam_fwhm_arcmin=8.):


        #np.random.seed(213114124+int(freq_GHz))


        beam = gauss_beam(np.arange(self.lmax_sim+10),beam_fwhm_arcmin)#hp.gauss_beam(beam_fwhm_arcmin*(np.pi/60./180),lmax=3*self.nside)
        beam[beam==0] = np.inf
        beam = 1/beam

        shape,wcs = enmap.fullsky_geometry(self.pixRes_arcmin)

        T_noise = noise_ukarcmin*(np.pi/180/60)*curvedsky.rand_map(shape, wcs, beam**2)
        T_map =  self.T_cmb.copy()
        T_map += T_noise
        T_map += self.T_dust*galaticDust_SED(freq_GHz,in_uk=True)
        T_map += self.T_sync*synchrotron_SED(freq_GHz,in_uk=True)

        return T_map

    def get_input_cmb_alms(self):
        """
        Get the exact realization of the CMB EE and BB present in the sky (sadly not directly accessible in reality)!
        """
        return self.alm_cmb_T#


    def get_input_dust_alms(self):
        """
        Get the exact realization of the CMB EE and BB present in the sky (sadly not directly accessible in reality)!
        """
        return self.alm_dust_T

    def get_input_sync_alms(self):
        """
        Get the exact realization of the CMB EE and BB present in the sky (sadly not directly accessible in reality)!
        """
        return self.alm_sync_T


class simple_sky_model_pol:
    def __init__(self,camb_file='./CMB_fiducial_totalCls.dat',seed=1,pixRes_arcmin=4,lmax_sim=500):



        cls_camb = np.loadtxt(camb_file,unpack=True)
        cl_tt = cls_camb[1]/(cls_camb[0]*(cls_camb[0]+1)/2/np.pi)
        cl_tt = np.append([0,0],cl_tt)
        cl_ee = cls_camb[2]/(cls_camb[0]*(cls_camb[0]+1)/2/np.pi)
        cl_ee = np.append([0,0],cl_ee)
        cl_bb = cls_camb[3]/(cls_camb[0]*(cls_camb[0]+1)/2/np.pi)*0.05
        cl_bb = np.append([0,0],cl_bb)
        cl_te = cls_camb[4]/(cls_camb[0]*(cls_camb[0]+1)/2/np.pi)
        cl_te = np.append([0,0],cl_te)
        ells = np.append([0,1],cls_camb[0])

        self.pixRes_arcmin=pixRes_arcmin/180./60*np.pi
        self.lmax_sim = lmax_sim

        shape,wcs = enmap.fullsky_geometry(self.pixRes_arcmin,dims=(2,))


        opos = enmap.posmap(shape, wcs)
        galShape = 1/(np.abs(opos[0])+1e-1)
        galShape/=np.mean(galShape**2)**.5


        np.random.seed(seed)
        self.shape = shape
        self.wcs = wcs

        self.alm_cmb_E = curvedsky.rand_alm(cl_ee,lmax=lmax_sim)#hp.synalm(cl_ee,lmax=3*nside-1,new=True)
        self.alm_cmb_B = curvedsky.rand_alm(cl_bb,lmax=lmax_sim)#hp.synalm(cl_bb,lmax=3*nside-1,new=True)
        tmp_empty = enmap.empty(shape,wcs)
        self.Q_cmb,self.U_cmb = curvedsky.alm2map(np.array([self.alm_cmb_E,self.alm_cmb_B]),tmp_empty,spin=[2])

        ps = np.zeros([2,2,lmax_sim])
        ps[0,0] = synchrotron_Cl('E','E')[:lmax_sim]
        ps[1,1] = galaticDust_Cl('E','E')[:lmax_sim]
        ps[1,0] = syncxdust_Cls('E','E')[:lmax_sim]

        self.alm_sync_E,self.alm_dust_E = curvedsky.rand_alm(ps,lmax=lmax_sim)

        ps = np.zeros([2,2,lmax_sim])
        ps[0,0] = synchrotron_Cl('B','B')[:lmax_sim]
        ps[1,1] = galaticDust_Cl('B','B')[:lmax_sim]
        ps[1,0] = syncxdust_Cls('B','B')[:lmax_sim]

        self.alm_sync_B,self.alm_dust_B = curvedsky.rand_alm(ps,lmax=lmax_sim)


        tmp_empty = enmap.empty(shape,wcs)
        self.Q_dust,self.U_dust = curvedsky.alm2map(np.array([self.alm_dust_E,self.alm_dust_B]),tmp_empty,spin=[2])

        self.Q_dust*=galShape
        self.U_dust*=galShape
        
        self.alm_dust_E,self.alm_dust_B = curvedsky.map2alm([self.Q_dust,self.U_dust],spin=[2],lmax=lmax_sim)


        tmp_empty = enmap.empty(shape,wcs)
        self.Q_sync,self.U_sync = curvedsky.alm2map(np.array([self.alm_sync_E,self.alm_sync_B]),tmp_empty,spin=[2])


        self.Q_sync*=galShape
        self.U_sync*=galShape

        self.alm_sync_E,self.alm_sync_B = curvedsky.map2alm([self.Q_sync,self.U_sync],spin=[2],lmax=lmax_sim)

    def observe(self,freq_GHz,noise_ukarcmin=3.,beam_fwhm_arcmin=8.):


        #np.random.seed(213114124+int(freq_GHz))


        beam = gauss_beam(np.arange(self.lmax_sim+10),beam_fwhm_arcmin)#hp.gauss_beam(beam_fwhm_arcmin*(np.pi/60./180),lmax=3*self.nside)
        beam[beam==0] = np.inf
        beam = 1/beam

        shape,wcs = enmap.fullsky_geometry(self.pixRes_arcmin)

        Q_noise = np.sqrt(2)*noise_ukarcmin*(np.pi/180/60)*curvedsky.rand_map(shape, wcs, beam**2)
        U_noise = np.sqrt(2)*noise_ukarcmin*(np.pi/180/60)*curvedsky.rand_map(shape, wcs, beam**2)
        Q_map =  self.Q_cmb.copy()
        Q_map += Q_noise
        Q_map += self.Q_dust*galaticDust_SED(freq_GHz,in_uk=True)
        Q_map += self.Q_sync*synchrotron_SED(freq_GHz,in_uk=True)


        U_map =  self.U_cmb.copy()
        U_map += U_noise
        U_map += self.U_dust*galaticDust_SED(freq_GHz,in_uk=True)
        U_map += self.U_sync*synchrotron_SED(freq_GHz,in_uk=True)

        return Q_map,U_map

    def get_input_cmb_alms(self):
        """
        Get the exact realization of the CMB EE and BB present in the sky (sadly not directly accessible in reality)!
        """
        return self.alm_cmb_E,self.alm_cmb_B


    def get_input_dust_alms(self):
        """
        Get the exact realization of the CMB EE and BB present in the sky (sadly not directly accessible in reality)!
        """
        return self.alm_dust_E,self.alm_dust_B


    def get_input_sync_alms(self):
        """
        Get the exact realization of the CMB EE and BB present in the sky (sadly not directly accessible in reality)!
        """
        return self.alm_sync_E,self.alm_sync_B

class pysm_sky_model:
    def __init__(self,camb_file='./CMB_fiducial_totalCls.dat',seed=1,pixRes_arcmin=2.,lmax_sim=500,nside_pysm=512):

        cls_camb = np.loadtxt(camb_file,unpack=True)
        cl_tt = cls_camb[1]/(cls_camb[0]*(cls_camb[0]+1)/2/np.pi)
        cl_tt = np.append([0,0],cl_tt)
        cl_ee = cls_camb[2]/(cls_camb[0]*(cls_camb[0]+1)/2/np.pi)
        cl_ee = np.append([0,0],cl_ee)
        cl_bb = cls_camb[3]/(cls_camb[0]*(cls_camb[0]+1)/2/np.pi)
        cl_bb = np.append([0,0],cl_bb)
        cl_te = cls_camb[4]/(cls_camb[0]*(cls_camb[0]+1)/2/np.pi)
        cl_te = np.append([0,0],cl_te)
        ells = np.append([0,1],cls_camb[0])

        self.pixRes_arcmin=pixRes_arcmin/180./60*np.pi
        self.lmax_sim = lmax_sim

        shape,wcs = enmap.fullsky_geometry(self.pixRes_arcmin,dims=(2,))
        self.shape = shape
        self.wcs = wcs
        np.random.seed(seed)

        self.nside_pysm =nside_pysm

        self.alm_cmb_E = curvedsky.rand_alm(cl_ee,lmax=lmax_sim)#hp.synalm(cl_ee,lmax=3*nside-1,new=True)
        self.alm_cmb_B = curvedsky.rand_alm(cl_bb,lmax=lmax_sim)#hp.synalm(cl_bb,lmax=3*nside-1,new=True)
        tmp_empty = enmap.empty(shape,wcs)
        self.Q_cmb,self.U_cmb = curvedsky.alm2map(np.array([self.alm_cmb_E,self.alm_cmb_B]),tmp_empty,spin=[2])

    def observe(self,freq_GHz,noise_ukarcmin=3.,beam_fwhm_arcmin=8.):

        import pysm3
        import pysm3.units as u
        #np.random.seed(213114124+int(freq_GHz))

        beam = gauss_beam(np.arange(self.lmax_sim+10),beam_fwhm_arcmin)#hp.gauss_beam(beam_fwhm_arcmin*(np.pi/60./180),lmax=3*self.nside)
        beam[beam==0] = np.inf
        beam = 1/beam

        shape,wcs = enmap.fullsky_geometry(self.pixRes_arcmin)

        Q_noise = np.sqrt(2)*noise_ukarcmin*(np.pi/180/60)*curvedsky.rand_map(shape, wcs, beam**2)
        U_noise = np.sqrt(2)*noise_ukarcmin*(np.pi/180/60)*curvedsky.rand_map(shape, wcs, beam**2)

        sky = pysm3.Sky(nside=self.nside_pysm,preset_strings=["d1","s1"],output_unit="K_CMB")
        # Get the map at the desired frequency:
        I,Q_foreground,U_foreground = sky.get_emission(freq_GHz*u.GHz)*1e6

        I,Q_foreground,U_foreground = reproject.enmap_from_healpix([I,Q_foreground,U_foreground], shape, wcs,
                                  ncomp=3, unit=1, lmax=self.lmax_sim,rot=None)

        Q_map =  self.Q_cmb.copy()
        Q_map += Q_noise
        Q_map += Q_foreground


        U_map =  self.U_cmb.copy()
        U_map += U_noise
        U_map += U_foreground

        return Q_map,U_map

    def get_true_alms(self):
        """
        Get the exact realization of the CMB EE and BB present in the sky (sadly not directly accessible in reality)!
        """
        return self.alm_cmb_E,self.alm_cmb_B

