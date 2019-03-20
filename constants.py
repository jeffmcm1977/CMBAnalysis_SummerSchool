## variables to set up the size of the map
N = 2**10  # this is the number of pixels in a linear dimension
            ## since we are using lots of FFTs this should be a factor of 2^N
pix_size  = 0.5 # size of a pixel in arcminutes
N_iterations = 16
## variables to set up the map plots
c_min = -400  # minimum for color bar
c_max = 400   # maximum for color bar
X_width = N*pix_size/60.  # horizontal map width in degrees
Y_width = N*pix_size/60.  # vertical map width in degrees

beam_size_fwhp = 1.25

Number_of_Sources  = 5000
Amplitude_of_Sources = 200
Number_of_Sources_EX = 50
Amplitude_of_Sources_EX = 1000
Number_of_SZ_Clusters  = 500
Mean_Amplitude_of_SZ_Clusters = 50
SZ_beta = 0.86
SZ_Theta_core = 1.0


white_noise_level = 10.
atmospheric_noise_level = 0.5*0.
one_over_f_noise_level = 0.

#### parameters for setting up the spectrum
delta_ell = 50.
ell_max = 5000.
