import numpy as np
import matplotlib.pyplot as plt
# We use some stuff we learned before
from cmb_modules import calculate_2d_spectrum,make_CMB_T_map

# We need to load the theory spectra
def get_theory():
    ells,tt,_,_,pp,_ = np.loadtxt("CAMB_fiducial_cosmo_scalCls.dat",unpack=True)
    TCMB2 = 7.4311e12
    ckk = pp/4./TCMB2
    ucltt = tt / ells/(ells+1.)*2.*np.pi
    ells2,lcltt = np.loadtxt("CMB_fiducial_totalCls.dat",unpack=True,usecols=[0,1])
    lcltt = lcltt / ells2/(ells2+1.)*2.*np.pi
    lcltt = lcltt[:len(ells)]
    return ells,ucltt,lcltt,ckk

def get_lensed(patch_deg_width,pix_size,ells,ucltt,clkk):
    
    # Number of pixels in each direction
    N = int(patch_deg_width*60./pix_size)
    # We next generate an unlensed CMB map as a Gaussian random field as we learned before
    DlTT = ucltt*ells*(ells+1.)/2./np.pi
    unlensed = make_CMB_T_map(N,pix_size,ells,DlTT)
    # We also need a lensing convergence (kappa) map
    DlKK = clkk*ells*(ells+1.)/2./np.pi
    kappa = make_CMB_T_map(N,pix_size,ells,DlKK)
    # We get the Fourier coordinates
    ly,lx,modlmap = get_ells(N,pix_size)

    # Now we can lens our input unlensed map
    lensed = lens_map(unlensed,kappa,modlmap,ly,lx,N,pix_size)


    return N,lensed,kappa,ly,lx,modlmap


def lens_map(imap,kappa,modlmap,ly,lx,N,pix_size):
    # First we convert lensing convergence to lensing potential
    phi = kappa_to_phi(kappa,modlmap,return_fphi=True)
    # Then we take its gradient to get the deflection field
    grad_phi = gradient(phi,ly,lx)
    # Then we calculate the displaced positions by shifting the physical positions by the deflections
    pos = posmap(N,pix_size) + grad_phi
    # We convert the displaced positions into fractional displaced pixel numbers
    # because scipy doesn't know about physical distances
    pix = sky2pix(pos, N,pix_size)
    # We prepare an empty output lensed map array
    omap = np.empty(imap.shape, dtype= imap.dtype)
    # We then tell scipy to calculate the values of the input lensed map
    # at the displaced fractional positions by interpolation and grid that onto the final lensed map
    from scipy.ndimage import map_coordinates
    map_coordinates(imap, pix, omap, order=5, mode='wrap')
    return omap

# This function needs to know about the Fourier coordinates of the map
def get_ells(N,pix_size):
    # This function returns Fourier wavenumbers for a Cartesian square grid
    N=int(N)
    ones = np.ones(N)
    inds  = (np.arange(N)+.5 - N/2.) /(N-1.)
    ell_scale_factor = 2. * np.pi 
    lx = np.outer(ones,inds) / (pix_size/60. * np.pi/180.) * ell_scale_factor
    ly = np.transpose(lx)
    modlmap = np.sqrt(lx**2. + ly**2.)
    return ly,lx,modlmap

# We need to convert kappa to phi
def kappa_to_phi(kappa,modlmap,return_fphi=False):
    return filter_map(kappa,kmask(2./modlmap/(modlmap+1.),modlmap,ellmin=2))

# where we used a Fourier space masking function which will come in handy
def kmask(filter2d,modlmap,ellmin=None,ellmax=None):
    # Apply a minimum and maximum multipole mask to a filter
    if ellmin is not None: filter2d[modlmap<ellmin] = 0
    if ellmax is not None: filter2d[modlmap>ellmax] = 0
    return filter2d

# To do that we also need to know generally how to filter a map
def filter_map(Map,filter2d):
    FMap = np.fft.fftshift(np.fft.fft2(Map))
    FMap_filtered = FMap * filter2d
    Map_filtered = np.real(np.fft.ifft2(np.fft.ifftshift(FMap_filtered)))
    return Map_filtered


# We also need to calculate a gradient
# We do this in Fourier space
def gradient(imap,ly,lx):
    # Filter the map by (i ly, i lx) to get gradient
    return np.stack([filter_map(imap,ly*1j),filter_map(imap,lx*1j)])

# We also needed the map of physical positions
def posmap(N,pix_size):
    pix    = np.mgrid[:N,:N]
    return pix2sky(pix,N,pix_size)

# For that we need to be able to convert pixel indices to sky positions
def pix2sky(pix,N,pix_size):
    py,px = pix
    dec = np.deg2rad((py - N//2 - 0.5)*pix_size/60.)
    ra = np.deg2rad((px - N//2 - 0.5)*pix_size/60.)
    return np.stack([dec,ra])

# Finally, for the lensing operation, we also needed to convert physical sky positions to pixel indices
# which is just the inverse of the above
def sky2pix(pos,N,pix_size):
    dec,ra = np.rad2deg(pos)*60.
    py = dec/pix_size + N//2 + 0.5
    px = ra/pix_size + N//2 + 0.5
    return np.stack([py,px])
