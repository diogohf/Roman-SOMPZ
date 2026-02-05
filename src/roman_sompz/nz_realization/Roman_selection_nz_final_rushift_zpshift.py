import numpy as np
import pickle
import pandas as pd
colors = ['k', 'teal', 'orange', 'powderblue','tomato'] 
#from astropy.coordinates import SkyCoord
#from astropy import units as u
import healpy as hp
from astropy.io import fits
import time
import h5py
from scipy.interpolate import interp1d
import multiprocessing as mp
import os 
import yaml
from functions_nzrealizations_Roman_pointz import*
import sys
sys.path.append('/global/cfs/cdirs/des/boyan/sompz_y6/sompz/')
from functions_sompz import *
from functions_WL import *
from datetime import datetime
today = datetime.today()
today = today.strftime('%B%d')
import gc
    
Nsamples = 10000

# Whether you want ZPU (photometric redshift zero point uncertainty) or RU (redshift sample uncertainty) inside analysis
ZPU = False
RU = False
#Bin conditionalization uncertainty is negelected in Y6 analysis 
#people have found in Y3 this is very small, about 0.003 for all bins while the combined uncertainty is always dominated by other factors, like photometric calibration uncertainty on 0th bin, and SN+SV for all 4 bins
#Y6 uncertainty should be similar as it share simliar redshift sample and deep sample. Thus, this uncertainty can be safely neglected
#RUBC = False

ZPU_label = "_ZPU" if ZPU else ""
RU_label = "_RU" if RU else ""


### dir
data_dir = '/global/cfs/cdirs/des/boyan/sompz_output_Roman/sc3b_d5/'
outpath = '/global/cfs/cdirs/des/boyan/sompz_output_Roman/sc3b_d5/SVSN/'
#save_h5 = f'{outpath}/SOMPZ_newcode_4_fullZ_{today}.h5'

bands = ['U','G','R','I','Z','J','H','K']

#Here We are making the files required
#I am making the assumption that Chun-Hao generates deep galaxies with both deep and wide flux
#And these deep galaxies don't have injection counts and shear weights like DES
#Also, I assume Chun-Hao randomly assign spec to deep galaxy and make the spec catalog
#So there is no need to match spec to deep by ra/dec

deep_balrog_file = data_dir + 'deep_0.hdf5_10000000'
redshift_deep_balrog_file = data_dir + 'deep_0.hdf5_spec_10000000'
deep_cells_assignment_balrog_file = data_dir + 'balrog_data_deep_assignment.hdf5'
wide_cells_assignment_balrog_file = data_dir + 'balrog_data_wide_assignment.hdf5'
deep_cells_assignment_spec_file = data_dir + 'spec_data_deep_assignment.hdf5'
wide_cells_assignment_spec_file = data_dir + 'spec_data_assignment.hdf5'
wide_cells_assignment_wide_file = data_dir + 'wide_data_assignment.hdf5'
tomo_bins_dict_file = data_dir + 'tomo_bins_wide_dict.hdf5'

#Make Balrog File
balrog_data_raw = pd.read_hdf(deep_balrog_file)
balrog_data = balrog_data_raw.reset_index(drop=True, inplace=False)
balrog_data = balrog_data[['ra','dec']]
balrog_data['cell_deep'] = h5py.File(deep_cells_assignment_balrog_file, 'r')['cells'][:]
balrog_data['cell_wide_unsheared'] = h5py.File(wide_cells_assignment_balrog_file, 'r')['cells'][:]
balrog_data['overlap_weight'] = np.ones(np.shape(h5py.File(wide_cells_assignment_balrog_file, 'r')['cells'][:])[0])
balrog_data_noshift = balrog_data

#Make spec file
spec_data_raw = pd.read_hdf(redshift_deep_balrog_file)
spec_data = spec_data_raw.reset_index(drop=True, inplace=False)
spec_data = spec_data[['ra','dec','z']]
spec_data['cell_deep'] = h5py.File(deep_cells_assignment_spec_file, 'r')['cells'][:]
spec_data['cell_wide_unsheared'] = h5py.File(wide_cells_assignment_spec_file, 'r')['cells'][:]
spec_data['overlap_weight'] = np.ones(np.shape(h5py.File(wide_cells_assignment_spec_file, 'r')['cells'][:])[0])
spec_noshift = spec_data

#Wide assignment
wide_assignment = h5py.File(wide_cells_assignment_wide_file, 'r')['cells'][:]

#Here I am making a dictionary: {bin_num: array(wide cells in bin)} 
'''
tomo_dict = h5py.File(tomo_bins_dict_file, 'r')
wide_assignment = h5py.File(wide_cells_assignment_wide_file, 'r')['cells'][:]

#number of tomographic bins
num_bins = 9
for i in range(num_bins):
    bin_msk = np.where(tomo_dict['bin'][:] == i )
    tomo_bins_wide[i] = np.unique(np.array(wide_assignment[bin_msk]))
'''

with open(tomo_bins_dict_file, 'rb') as f:
   tomo_dict = pickle.load(f)

tomo_bins_wide = {}
num_bins = 9
for i in range(num_bins):
    tomo_bins_wide[i] = (np.array(tomo_dict[i])[:,0]).astype(int)




print('finish data prep')  
######################################### Finished all data prep #############################################
############################################################################################################## 

ncells_deep = 64*64
ncells_wide = 32*32

## Tomographic bin selection
zpdfcols = ['z']

LHC_samples = np.load("./LHC_samples.npy")
print("Starting loop of LHC shifts")

# prepare data for ZPU and RU
if ZPU:
    # Load deep galaxy ID, this is the galaxy ID for the flux-perturbed galaxy cell assignments
    deep_data_ID = np.load(outpath + 'deep_data_ID.npy')
    
if RU:
    #SC is the median redshift bias of COSMOS galaxies wrt spectroscopic galaxies
    #SC[:,0] is mag_i, SC[:,1] is the bias(z)/(1+z) for that mag_i
    #The function below interpolates this bias for all mag_i values
    #SP is the median redshift bias of PAUS galaxies wrt spectroscopic galaxies
    median_bias_SC_file = np.load('/global/cfs/cdirs/des/boyan/sompz_data_y6/Redshift_Catalog/redshift_bias/median_bias_SC.npz')
    median_bias_SC = median_bias_SC_file['bias']
    std_bias_SC = median_bias_SC_file['std']
    ibin_edges_SC = median_bias_SC_file['ibin_edges']
    zbin_edges_SC = median_bias_SC_file['zbin_edges']
    
    
    median_bias_SP_file = np.load('/global/cfs/cdirs/des/boyan/sompz_data_y6/Redshift_Catalog/redshift_bias/median_bias_SP.npz')
    median_bias_SP = median_bias_SP_file['bias']
    std_bias_SP = median_bias_SP_file['std']
    ibin_edges_SP = median_bias_SP_file['ibin_edges']
    zbin_edges_SP = median_bias_SP_file['zbin_edges']
    
   


#Start running
for LHC_id, LHC_sample in enumerate(LHC_samples[0:25]):
    
    LHC_id += 0
    
    print('id', LHC_id)
    filename = 'nz_samples_%s%s_LHC%d_pointZ_1e6_Roman_sc3b_d5.h5'%(RU_label, ZPU_label, LHC_id)
    if os.path.exists(outpath + filename):
        print("%d exists, continuing..."%LHC_id)
        continue
    
    T0 = time.time()
    
    # zero point uncertainty
    if ZPU:
        # If ZPU, load new cell deep assignment of galaxies based on purtubed deep fluxes
        
        # Load deep cell assignment with zpu
        cells_deep = np.load("%s/som_deep_64x64_assign_LHC%d.npz"%(outpath, LHC_id))['cells']
    
        cells_deep_df = pd.DataFrame({'ID': deep_data_ID, 'cell_deep': cells_deep})
        
        # use this cell_deep column for balrog_data and cosmos
        balrog_data = balrog_data_noshift.copy().drop('cell_deep', axis=1)
        cosmos = spec_noshift.copy().drop('cell_deep', axis=1)

        balrog_data = balrog_data.merge(cells_deep_df, on='ID', how='left')
        cosmos = cosmos.merge(cells_deep_df, on='ID', how='left')
       
    else:
        
        balrog_data = balrog_data_noshift.copy()
        cosmos = spec_noshift.copy()

        


    ## Redshift sample uncertainty shift
    if RU:
        print('running RU')
        #New way magi-z bins of estimating RU
        
        #only perturb unique galaxy and match later
        unique_zpdf = cosmos[['ID','SOURCE','BDF_MAG_DERED_CALIB_I'] + zpdfcols].drop_duplicates()
        zpdf = unique_zpdf[zpdfcols].values
        origins = unique_zpdf['SOURCE'].values
        mags = unique_zpdf['BDF_MAG_DERED_CALIB_I'].values
        z_bins = np.arange(0, 4, 0.01)
        zmeans = np.dot(zpdf, z_bins) / zpdf.sum(axis=1)
        print('zmeans shape', np.shape(zmeans))
        
        #add 0s for zpdf 
        zero_columns_beginning = np.zeros((zpdf.shape[0], 100))
        zero_columns_end = np.zeros((zpdf.shape[0], 100))
        zpdf = np.hstack((zero_columns_beginning, zpdf, zero_columns_end))
        zbinsc_laigle_stacked = np.arange(-1.0,5.0,0.01)
       
        
        #Get the gaussian draws from pre-generated LHC
        #sigma_C, sigma_P, sigma_bincond = LHC_sample[-4:-1]
        #print(sigma_C, sigma_P)
        
        new_zpdf = np.zeros_like(zpdf)
        shift_SC = np.random.normal(loc=-median_bias_SC, scale=std_bias_SC)
        shift_SP = np.random.normal(loc=-median_bias_SP, scale=std_bias_SP)

        #add a 0 column before and after zpdf, and do pileup on 0 and 3.99 after interpolation
        #This ensures in final pdf, the tail information is not mitigated
        for i in range(len(zpdf)):
            if origins[i] == 'COSMOS2020_FARMER_LEPHARE':

                mag_index =  np.digitize(mags[i], ibin_edges_SC) -1
                z_index = np.digitize(zmeans[i], zbin_edges_SC) -1
                mag_index = np.clip(mag_index, 0, len(ibin_edges_SC) - 2)
                z_index = np.clip(z_index, 0, len(zbin_edges_SC) - 2)
                '''
                bias = median_bias_SC_draw[mag_index, z_index]
                std = std_bias_SC[mag_index, z_index]
                # bias is photo_z - spec_z, so want to shift COSMOS to -1 * bias
                shift = np.random.normal(-bias, std)
                '''
                shift = shift_SC[mag_index, z_index]
                f = interp1d(zbinsc_laigle_stacked, zpdf[i], bounds_error=False, fill_value=(0,0))
                #shift f(x) to the right is the same as shifting x axis to the left
                _npz = f(zbinsc_laigle_stacked-shift)
                _npz /= _npz.sum()
                new_zpdf[i] = _npz
                del f

    
                
                
            elif origins[i] == 'PAUSCOSMOS':

                mag_index =  np.digitize(mags[i], ibin_edges_SP) -1
                z_index = np.digitize(zmeans[i], zbin_edges_SP) -1
                mag_index = np.clip(mag_index, 0, len(ibin_edges_SP) - 2)
                z_index = np.clip(z_index, 0, len(zbin_edges_SP) - 2)
                '''
                bias = median_bias_SP_draw[mag_index, z_index]
                std = std_bias_SP[mag_index, z_index]
                # bias is photo_z - spec_z, so want to shift COSMOS to -1 * bias
                shift = np.random.normal(-bias, std)
                '''
                shift = shift_SP[mag_index, z_index]
                f = interp1d(zbinsc_laigle_stacked, zpdf[i], bounds_error=False, fill_value=(0,0))
                #shift f(x) to the right is the same as shifting x axis to the left
                _npz = f(zbinsc_laigle_stacked-shift)
                _npz /= _npz.sum()
                new_zpdf[i] = _npz
                del f
              
                
            else:
                new_zpdf[i] = zpdf[i]
               
            
            

        #pileup added columns
        new_zpdf[:, 100] += np.sum(new_zpdf[:, :100], axis=1) 
        new_zpdf[:, -101] += np.sum(new_zpdf[:, -100:], axis=1) 
        new_zpdf = new_zpdf[:, 100:-100]

        all_sums_to_1 = np.allclose(new_zpdf.sum(axis=1), 1)
        print(new_zpdf.sum(axis=1))
        print('Does all galaxy sum to 1: ',all_sums_to_1)
        
        unique_zpdf.loc[:,zpdfcols] = new_zpdf
        unique_zpdf.drop(columns = ['SOURCE','BDF_MAG_DERED_CALIB_I'])
        
        cosmos = cosmos.drop(columns=zpdfcols).merge(unique_zpdf, on='ID', how='left')
                
    

        print('finish RU')
        
        


        
    #cosmos is the redshift galaxies matched to deep and detected by Balrog. (not unique)


    ###########################################################################################
    ### All code below is for calaulating Sample Variance and Shot Noise
    ###########################################################################################

    ### Load shear-response weightd p(chat) and p(c|chat)
    # pcchat needs to be calculated in everey cell assignment
    pchat = np.load(data_dir+'pchat.npy')
    pcchat = np.array(h5py.File(data_dir +'pc_chat.hdf5', 'r')['pc_chat'][:]) # (4096,1024)
    pcchat = np.zeros_like(pcchat)
    np.add.at(pcchat, 
          (balrog_data.cell_deep.values,balrog_data.cell_wide_unsheared.values),
          balrog_data.overlap_weight.values)
    
    ###########################################################################################
    ### Define new redshift bins
    ###########################################################################################

    # Define the redshift binning. This is currently set by the sample variance.
    # This binning (dz = 0.05) ensures minimal correlation between bins, which is crucial since the Dir sampling is assuming each redshfit bin to be independent
    min_z   = 0.01
    max_z   = 2.32 #2.01
    delta_z = 0.05
    zbins   = np.arange(min_z,max_z,delta_z)
    zbinsc  = zbins[:-1]+(zbins[1]-zbins[0])/2.
    
    
    

    #####################################################################################
    ### Calculating bincond and shear-response weight factors (with interpolation)
    #####################################################################################
    

    Nzc_bins = []
    Nc_bins = []
    Rzc_bins = []
    Rc_redshift_bins = []
    Rc_deep_bins = []
    
    # Loop over the tomographic bins
    for i in range(num_bins):
        # Compute Nzc for cosmos
        Nzc_bins.append(return_Nzc(cosmos[cosmos.cell_wide_unsheared.isin(tomo_bins_wide[i])]))
        # Compute Nc for balrog_data
        Nc_bins.append(return_Nc(balrog_data[balrog_data.cell_wide_unsheared.isin(tomo_bins_wide[i])]))
        # Compute Rzc for cosmos. This is the response weighted Nzc.
        Rzc_bins.append(return_Rzc(cosmos[cosmos.cell_wide_unsheared.isin(tomo_bins_wide[i])]))
        # Compute Rc_redshift for cosmos. 
        Rc_redshift_bins.append(return_Rc(cosmos[cosmos.cell_wide_unsheared.isin(tomo_bins_wide[i])]))
        # Compute Rc_deep for balrog_data
        Rc_deep_bins.append(return_Rc(balrog_data[balrog_data.cell_wide_unsheared.isin(tomo_bins_wide[i])]))

    # Stack Nzc and Nc for each bin. The first is no tomo-binning 
    Nzc_bins = np.array([return_Nzc(cosmos)] + Nzc_bins)
    Nc_bins = np.array([return_Nc(balrog_data)] + Nc_bins)

    # Handle the case where no redshift counts in a deep cell after the bin condition
    for i in range(num_bins):
        sel = (np.sum(Nzc_bins[0], axis=0) > 0) & (np.sum(Nzc_bins[i + 1], axis=0) == 0)
        Nzc_bins[i + 1][:, sel] = Nzc_bins[0][:, sel].copy()

    # Compute the responseXlensing weighted averages for redshift and deep samples
    Rzc_bins = np.array([return_Rzc(cosmos)] + Rzc_bins)
    Rc_redshift_bins = np.array([return_Rc(cosmos)] + Rc_redshift_bins)
    Rc_deep_bins = np.array([return_Rc(balrog_data)] + Rc_deep_bins)

    # Perform the bin condition fractions for redshift and deep samples
    fraction_Nzt = return_bincondition_fraction_Nzt_redshiftsample(Nzc_bins)
    fraction_Nt_D = return_bincondition_fraction_Nt_deepsample(Nc_bins)

    # Compute bin condition weights gzc(R,Bin)/gc(R,Bin) * gc(D,Bin)
    bincond_combined = fraction_Nzt * fraction_Nt_D[:, None, :]

    # Compute the response weights final Rzt = <Rzt>r * <Rt>D / <Rt>r
    Rt_combined = return_bincondition_weight_Rzt_combined(Rzc_bins, Rc_redshift_bins, Rc_deep_bins)

    del cosmos
    del balrog_data

    
    #############################################################
    ### Compute superphenotypes
    #############################################################

    # We need superphenotypes because this method require each sample variance to be disjoint. phenotype t (or deep cell c) are highly correlated. So we are going to use fzc = f(c|zT) * f(z|T) * f(T) etc. to aviod correlation in the algorithm. also redshift is rebinnind to dz = 0.05 for same reason.
    
    
    # The small t(phenotype) here represnts a deep cell c
    nts = Nc_bins[0].copy()
    nzt = Nzc_bins[0].copy()  

    # Removing pheno-type t that don't have galaxies
    # Note all Dir-related properties are masked - we don't want a deep cell with 0 galaxy inside to be sampled with >0 galaxies
    maskt = (np.sum(nzt,axis=0)>0.)
    nts = nts[maskt]
    nzt = nzt[:,maskt]
    
    # Computing the mean redshift per phenotype - prepare combining similar redshift phenotype to super-phenotype T
    # here zmeant is the mean redshift per phenotype t * 100
    # for example a mean redshift = 1.017, here zmeant = 102
    # zmeant - (4096) with maskt
    zmeant = np.zeros(nzt.shape[1])
    for i in range(nzt.shape[1]):
        zmeant[i] = np.average(np.arange(len(zbinsc)),weights=nzt.T[i])
    zmeant = np.rint(zmeant)


    
    # Decide which phenotypes go to which superphenotype
    # Choose number of superphenotypes nT
    nT = 6
    #bins stores the zmeant list for each T
    bins = {str(b):[] for b in range(nT)}
    j = 0 
    sumbin = 0
    #nTs stores the number of galaxies in t wrt cell meanz 
    nTs = np.zeros(len(zbinsc))
    for i in range(len(zbinsc)):
        #increase mean redshift by 0.05(i increase 1 here) 
        #add all galaxies in t with this mean redshift into sumbin
        sumbin += np.sum(nzt[:,((zmeant==i))],axis=1).sum()
        nTs[i] = np.sum(nzt[:,((zmeant==i))],axis=1).sum()
        #if not enough galaxy in this T still add, otherwide T num j+=1. append i to T dict
        if (sumbin <= np.sum(nzt)/(nT-1))|(j==nT-1):
            bins[str(j)].append(i)
        else:
            j += 1
            bins[str(j)].append(i)
            sumbin = np.sum(nzt[:,((zmeant==i))],axis=1).sum()

    # tot number of redshift galaxies
    print('total galaxy num', np.sum(nzt), np.sum(nzt)/6) 
    # number of redshift galaxies in each phenotype
    print('nTs', np.sum(nTs[np.array(bins['0'])]), np.sum(nTs[np.array(bins['1'])]),np.sum(nTs[np.array(bins['2'])]),np.sum(nTs[np.array(bins['3'])]),np.sum(nTs[np.array(bins['4'])]),np.sum(nTs[np.array(bins['5'])]))
    # what is the t zmeans for each T
    print('bins', bins)

    # Make a map that converts z to T
    z2Tmap = np.zeros((len(zmeant))).astype(int)
    for i in range(nT):
        z2Tmap[np.isin(zmeant.astype(int),bins[str(i)])] = i
        

    # Compute p(T), p(z,T) for the superphenotypes
    nzTi = np.zeros((len(zbinsc),nT))
    nTi = np.zeros((nT))
    for i in range(nT):
        nzTi[:,i] = np.sum(make_nzT(nzt,1,False)[:,bins[str(i)]],axis=1)
        nTi[i] = np.sum(make_nT(nzt,nts,1)[bins[str(i)]])

    # Calculate correlation metirc to see the overlap of nz between superphenotypes
    # Should be ~1 because we made assumption that T are independent, so det(corr metric) ->1
    print ('Correlation metric = %.3f'%corr_metric(nzTi))

    
    #############################################################
    ### Load Sample Variance from theory. 
    #############################################################

    ### Load the sample variance theory ingredient. This estimates the ratio between Shot noise and sample variance.
    
    #load redshift sample variance
    sv_dir = '/global/cfs/cdirs/des/boyan/sompz_y6/sv_sn_theory/code/results_Roman/'
    sv_th = np.load(sv_dir + 'sample_variance_spec_20deep_20spec_full.npy')[:len(zbinsc)]
    assert sv_th.shape[0]==len(zbinsc)
    
    #load deep sample variance
    sv_th_new_final_diag = np.load(sv_dir + 'sample_variance_deep_20deep_20spec_full.npy')[:len(zbinsc)]
    assert sv_th_new_final_diag.shape[0]==len(zbinsc)
    
    
    
    #sv_th = np.zeros(len(zbinsc))
    #sv_th_new_final_diag = np.zeros(len(zbinsc))
    # var = 1 + N* sv_th
    varn_th = 1 + np.sum(nzt,axis=1)*sv_th
    varn_th_deep_v2 = 1 + np.sum(nzt/np.sum(nzt,axis=0) * nts,axis=1)*sv_th_new_final_diag
    


    
    #######################################################################################
    ### Compute matrices and lambdas in prep for Dirichlet sampling
    ### Redshift: N_Tcz_Rsampe, N_Tz_Rsample (lambda_T), N_T_Rsample (lambda_mean_R)
    ### Deep: N_c_Dsample , N_T_Dsample(lambda_mean)
    #######################################################################################
    
    nt = sum(maskt)
    nz=len(zbinsc)
    N_Tcz_Rsample = np.zeros((nT,nt,nz))  #(nT,maskt,80) #maskt masks t that don't have galaxies
    for i in range(nT):
        sel = z2Tmap==i
        N_Tcz_Rsample[i, sel] = nzt.T[sel]

    N_Tc_Dsample = np.zeros((nT,nt))    #(nT,maskt) #maskt masks t that don't have galaxies
    for i in range(nT):
        sel = z2Tmap==i
        N_Tc_Dsample[i, sel] = nts[sel]

    
    
    N_T_Rsample = np.sum(N_Tcz_Rsample, axis=(1,2))
    N_z_Rsample = np.sum(N_Tcz_Rsample, axis=(0,1))
    N_Tz_Rsample = np.sum(N_Tcz_Rsample, axis=(1))
    N_cz_Rsample = np.sum(N_Tcz_Rsample, axis=(0))

    N_T_Dsample = np.sum(N_Tc_Dsample, axis=(1))
    N_c_Dsample = np.sum(N_Tc_Dsample, axis=(0))

    lambda_z_step1 = varn_th_deep_v2.copy()
    lambda_z_step2 = varn_th.copy()
    lambda_mean = np.sum(lambda_z_step1*N_z_Rsample/N_z_Rsample.sum())
    lambda_mean_R = np.sum(lambda_z_step2*N_z_Rsample/N_z_Rsample.sum())
    lambda_T = np.array([np.sum(lambda_z_step2 * x/x.sum()) for x in N_Tz_Rsample])

    # onecell -  cells c that have only one redshift galaxy incide 
    # N_cz_Rsample_onecell - normalized N_cz_Rsample for onecell
    # why are we doing this?
    onecell = np.sum(N_cz_Rsample>0,axis=1) == 1
    N_cz_Rsample_onecell = (N_cz_Rsample/np.sum(N_cz_Rsample,axis=1)[:,None])[onecell]
    
    ###########################################################################################
    ### Compute f(c, chat|bin) / (f(c|bin)*f(chat|bin)) - Balrog portion for nz realization 
    ### Compute f(chat|bin) - Wide portion for nz realization 
    ### Compute Fcchat_i - Balrog * Wide portion for nz realization
    ###########################################################################################
    
    fcchat = pcchat.T / pcchat.sum()

    fcchat_bins = []
    fchat_bins = []

    for i in range(num_bins):
        fcchat_i = fcchat[tomo_bins_wide[i]]
        normfactor_i = np.multiply.outer(np.sum(fcchat_i, axis=1), np.sum(fcchat_i, axis=0))
        fcchat_i = np.divide(fcchat_i, normfactor_i, np.zeros(np.shape(fcchat_i)), where=normfactor_i != 0)
        fcchat_i[~np.isfinite(fcchat_i)] = 0
        fcchat_bins.append(fcchat_i)

        fchat_i = pchat[tomo_bins_wide[i]]
        fchat_bins.append(fchat_i)

    Fcchat_bins = [fcchat_bins[i] * fchat_bins[i][:, None] for i in range(num_bins)]

    try:
        print(save_h5)
        store = pd.HDFStore(save_h5)
        store['nzt'] = pd.DataFrame(nzt)
        store['nzTi'] = pd.DataFrame(nzTi)
        store['nTi'] = pd.Series(nTi)
        store['nts'] = pd.Series(nts)

        for i in range(num_bins):
            store[f'bincond_combined_{i}'] = pd.DataFrame(bincond_combined[:, :, maskt][i])
            store[f'R_combined_{i}'] = pd.DataFrame(Rt_combined[:, :, maskt][i])
            store[f'fcchat_{i}'] = pd.DataFrame(fcchat_bins[i][:, maskt])
            store[f'fchat_{i}'] = pd.Series(fchat_bins[i])

        store['sv_th'] = pd.Series(sv_th)
        store['sv_th_deep'] = pd.Series(sv_th_new_final_diag)
        store['varn_th'] = pd.Series(varn_th)
        store['varn_th_deep'] = pd.Series(varn_th_deep_v2)
        store['z2Tmap'] = pd.Series(z2Tmap)
        store['maskt'] = pd.Series(maskt)
        store.close()
    except:
        pass

    
    
    ################################################################################################
    ### Until now, we have finished preparing all data that don't need sampling
    ### 1) The Balrog and Wide probabilities: Fcchat_0, Fcchat_1, Fcchat_2, Fcchat_3 
    ### 2) NzcT etc. matrices for Dirchlet sampling
    ### 3) bin-cond fraction: bincond_combined
    ### 4) shear response weight fracton: Rt_combined
    ### We now go inside the loop and use Dirichlet sampling to sample Nsample nzs
    ################################################################################################

    # epsilon of dirichlet equation, to ensure that dirchlet don't give error on Dir(0)
    alpha = 1e-300
    
    # The functions for producing Nsamples nz. These functions use previous (global) data 
    # and can't be in another file
    ########################################################################################
    def draw_3sdir_onlyR():
    
        #3sDir Redshift sample fzc(Redshift) = sum_T: fc|zT * fz|T * fT
    
        ### step1
        f_T = np.random.dirichlet(N_T_Rsample/lambda_mean_R+alpha)

        ### step2
        f_z_T = np.array([np.random.dirichlet(x/lambda_T[i]+alpha) for i,x in enumerate(N_Tz_Rsample)])

        ### step3
        f_cz_Rsample = np.random.dirichlet(N_cz_Rsample.reshape(np.prod(N_cz_Rsample.shape))+alpha).reshape(N_cz_Rsample.shape)
        f_cz = np.zeros((nt,nz))
        for k in range(N_Tcz_Rsample.shape[0]):  # For each T
            #z2Tmap - (4096) with maskt. Each value is what T c belongs to 
            #sel is the c indices that belong to kth T
            sel = z2Tmap==k                   
            dummy = f_cz_Rsample[sel] 
            dummy = np.divide(dummy, np.sum(dummy,axis=0), np.zeros(np.shape(dummy)), where = np.sum(dummy,axis=0)!=0)
            dummy[np.isnan(dummy)] = 0
            f_cz[sel] += np.einsum('cz,z->cz', dummy, f_z_T[k])* f_T[k]

        return f_cz


    def draw_3sdir_newmethod():
        
        # All Ns (both Redshift and deep) are masked from phenotype t that don't have redshift galaxies
        
        #3s Dir deep sample fc(Deep) = sum_T: fc|T * fT
        f_T = np.random.dirichlet(N_T_Dsample/lambda_mean+alpha)

        f_cT = np.zeros(nt)
        for k in range(nT):
            #z2Tmap - (4096) with maskt. Each value is what T c belongs to 
            #sel is the c indices that belong to kth T
            sel = z2Tmap==k
            #f_cT - (4096) with maskt, is actually fc(Deep)
            f_cT[sel] = np.random.dirichlet(N_Tc_Dsample[k,sel]+alpha) * f_T[k]

        #3sDir Redshift sample fzc(Redshift) = sum_T: fc|zT * fz|T * fT
        f_cz = draw_3sdir_onlyR()

        # fzc = fzc(Redshift)/sum_z:fzc(Redshift) * fc(Deep)
        sums = np.sum(f_cz, axis=1)[:, None]  
        f_z_c = np.divide(f_cz , sums, np.zeros(np.shape(f_cz)), where = sums!=0)
        
        
        # if onecell (only one redshift galaxy in cell c), then put pre-calculated values inside without Dirichlet sampling and phenotype T
        # This look like a 'hard code' to put inside correct value. Maybe there is something going one in this case: like superphenotypy T won't be a good representation for a one galaxy cell phenotype
        f_z_c[onecell] = N_cz_Rsample_onecell   

        ### compute f_{zc}
        f_cz = f_z_c * f_cT[:,None]
        return f_cz



    def aux_fun(i):
        np.random.seed()
        #want to generate Nsamples nz realization
        nz_samples_newmethod = np.zeros((Nsamples,num_bins, len(zbinsc)))

        for i_sample in range(Nsamples):
            # f_zt = fzc(redshift)/fc(Redshfit) * fc(Deep)
            # fc(Redshfit) = sum_z (fzc(redshift))
            f_zt = draw_3sdir_newmethod()
            # generate nz: multiply by bincond fraction, weight, and Balrog + Wide portions
            nz_samples_newmethod[i_sample] = return_nzsamples_fromfzt(f_zt, i_sample)
            if i_sample%(1e3) == 0 :
                print(i_sample)

        return nz_samples_newmethod  
    
    def return_nzsamples_fromfzt(fzt_dummy, i):
        fzt = np.zeros((4096, len(zbinsc))).T  # (80, 4096)
        fzt[:, maskt] = fzt_dummy.T

        nz_samples = []

        for j in range(num_bins):
            # fzt = fzc(Redshift)/fc(Redshift) * fc(Deep)
            fzt_j = fzt * bincond_combined[j] * Rt_combined[j]
            fzt_j /= np.sum(fzt_j)
            fzt_j[~np.isfinite(fzt_j)] = 0

            # add Balrog and Wide fraction and do summation over c and chat
            nz_j = np.einsum('zt,dt->z', fzt_j, Fcchat_bins[j])
            nz_j /= nz_j.sum()

            nz_samples.append(nz_j)

        return np.array(nz_samples)

  
    ########################################################################################
    # for Nsamples = 10000, using pool uses ~ same time as direct run
    '''
    indices = np.array_split(np.arange(Nsamples), 64)
    
    p = mp.Pool(64)
    nz_samples_newmethod = np.concatenate(p.map(aux_fun, indices), axis=0)
    p.terminate()
    '''
    p = mp.Pool(100)
    nz_samples_newmethod = np.concatenate(p.map(aux_fun, range(100)), axis=0)
    p.terminate()
    


    ##########################################################################################################
    ### Input: all the pre-prepared data and probabilities
    ### 1) Draw 3sDir sampling for Redshift and Deep using precomputed NzcT etc. matrics
    ### 2) Multiply pre-computed bincond and shear-response weight to Redshift and Deep probabilities
    ### 3) combine with pre-computed Balrog and Wide probabilities -> nz
    ### Output: nz (Nsamples, 4, 80)
    ##########################################################################################################
    #nz_samples_newmethod = aux_fun()

    # save the nz realiztions to h5 file
    print("Saving " + filename)
    with h5py.File(outpath + filename, 'w') as h5:
        h5.create_dataset("zbins", data=zbins)
        h5.create_dataset("zbinsc", data=zbinsc)

        for i in range(num_bins):
            h5.create_dataset(f"bin{i}", data=nz_samples_newmethod[:, i])



    T1 = time.time()
    print("Total time for %d: %.2f"%(LHC_id, T1-T0))
    
    # only SV + SN 
    if ((RU==False) & (ZPU==False)):
        break
        
