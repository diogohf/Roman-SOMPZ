import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import pickle

###################################################
### Functions for generating nz realizations
###################################################
min_z   = 0.01
max_z   = 2.32
delta_z = 0.05
zbins   = np.arange(min_z,max_z,delta_z)
zbinsc  = zbins[:-1]+(zbins[1]-zbins[0])/2.
zpdfcols = ["Z{:.3f}".format(s).replace(".","_") for s in zbinsc]
num_bins = 9

def return_Nzc(df):
    """
    - This function returns the counts Nzc=N(z,c) in each bin z and cell c.
    - The input is a pandas Dataframe containing a redshift sample. 
    - The redshift sample must have redshift and deep cell assignment.
    - It computes the balrog probability defined as #detections/#injections 
    to weight the counts of each galaxy in N(z,c).
    """
    
    num_galaxy = np.shape(np.array(df['z']))[0]
    w = np.ones(num_galaxy) 
    
    Ncz = np.zeros((64**2,len(zbinsc)))
    for ti in range(64**2):
        index = df['cell_deep'].values==ti
        sub = df[index]
        Ncz[ti], _ = np.histogram(sub['z'], bins=zbins, weights=w[index])
    
    Nzc = Ncz.T
    
    return Nzc


def return_Nc(df):
    """
    - This function returns the counts Nc=N(c) in each cell c.
    - The input is a pandas Dataframe containing a deep sample. 
    - The deep sample must have a deep cell assignment.
    - It computes the balrog probability defined as #detections/#injections 
    to weight the counts of each galaxy in N(c).
    """

    Nc = np.zeros((64*64))
    for ti in range(64**2):
        index = df['cell_deep'].values==ti
        Nc[ti] = np.sum(index)

    return Nc

def return_Rzc(df):
    """
    - This function returns the average lensingXshear weight in each bin z and cell c, Rzc= <ResponseXshear>(z,c)
    - The average is weighted by the balrog probability of each galaxy, defined as #detections/#injections.
    """
    # No response-weight for sim Roman
    Rzc = np.ones((len(zbinsc), 64**2))
    return Rzc

def return_Rc(df):
    """
    - This function returns the average lensingXshear weight in each cell c, Rc= <ResponseXshear>(c)
    - The average is weighted by the balrog probability of each galaxy, defined as #detections/#injections.
    """
    # No response-weight for sim Roman
    Rc = np.ones(64**2)

    return Rc

def return_bincondition_fraction_Nzt_redshiftsample(redshift_sample_Nzt):
    """This function returns the fraction of counts in Nzt with over 
    without bin condition, for each tomographic bin.
    """
    
    pz_c_list = []
    gzt_list = []

    # Compute pz_c for each tomographic bin & no tomo
    for i in range(num_bins+1):
        pz_c = np.divide(redshift_sample_Nzt[i], np.sum(redshift_sample_Nzt[i], axis=0), 
                         np.zeros(np.shape(redshift_sample_Nzt[i])), where=np.sum(redshift_sample_Nzt[i], axis=0) != 0)
        pz_c[~np.isfinite(pz_c)] = 0
        pz_c_list.append(pz_c)

    # Compute gzt for each tomographic bin, relative to no tomo
    for i in range(1, num_bins+1):
        gzt = np.divide(pz_c_list[i], pz_c_list[0], np.zeros(np.shape(pz_c_list[i])), where=pz_c_list[0] != 0)
        gzt[~np.isfinite(gzt)] = 0
        gzt_list.append(gzt)

    return np.array(gzt_list)



def return_bincondition_fraction_Nt_deepsample(deep_sample_Nt):
    """This function returns the fraction of counts in Nt with over 
    without bin condition, for each tomographic bin. Deep sample.
    """    
    
    gt_list = []

    # Compute gt for each bin (1 to 4), relative to bin 0
    for i in range(1, num_bins+1):
        gt = np.divide(deep_sample_Nt[i], deep_sample_Nt[0], 
                       np.zeros(np.shape(deep_sample_Nt[i])), where=deep_sample_Nt[0] != 0)
        gt[~np.isfinite(gt)] = 0
        gt_list.append(gt)

    return np.array(gt_list)



def return_bincondition_weight_Rzt_combined(redshift_sample_Rzt, redshift_sample_Rt, deep_sample_Rt):
    """This function returns the final average responseXshear weight in each deep cell and redshift bin: Rzc.
    Response weight = Response to shear of the balrog injection of a deep galaxy.
    Shear weight = Weight to optimize of signal to noise of some shear observable. 
    - final Rzt = <Rzt>r * <Rt>D / <Rt>r
    where: 
    - <Rzt>r: average weight in z,c in the redshift sample.
    - <Rt>r: average weight in c in the redshift sample.
    - <Rt>d: average weight in c in the deep sample.
    It basically rescales the weight in Rzt such that it matches the average weight according to the deep sample.
    """    
    
    Rzt_finals = []
    
    for i in range(1, num_bins+1): 
        Rzt_factor = np.divide(deep_sample_Rt[i], redshift_sample_Rt[i], 
                               np.zeros(np.shape(redshift_sample_Rt[i])), where=redshift_sample_Rt[i] != 0)
        Rzt_factor[~np.isfinite(Rzt_factor)] = 0
        
        Rzt_final = np.einsum('zt,t->zt', redshift_sample_Rzt[i], Rzt_factor)
        Rzt_finals.append(Rzt_final)

    return np.array(Rzt_finals)


def make_nzT(nzti, njoin, plot=False):
    zmeanti = np.zeros(nzti.shape[1])
    for i in range(nzti.shape[1]):
        try: zmeanti[i] = np.average(np.arange(len(zbinsc)),weights=nzti.T[i])
        except: zmeanti[i] = np.random.randint(len(zbinsc))
    zmeanti = np.rint(zmeanti)

    nzTi = np.zeros((len(zbinsc),int(len(zbinsc)/njoin)))
    for i in range(int(len(zbinsc)/njoin)):
        nzTi[:,i] = np.sum(nzti[:,((zmeanti>=njoin*i)&(zmeanti<njoin*i+njoin))],axis=1)

    if plot:
        plt.figure()
        for i in range(int(len(zbinsc)/njoin)):
            plt.plot(zbinsc,nzTi[:,i])
        plt.show()

    return nzTi

def make_nT(nzti, nti, njoin):
    zmeanti = np.zeros(nzti.shape[1])
    for i in range(nzti.shape[1]):
        try: zmeanti[i] = np.average(np.arange(len(zbinsc)),weights=nzti.T[i])
        except: zmeanti[i] = np.random.randint(len(zbinsc))
    zmeanti = np.rint(zmeanti)

    nTi = np.zeros(int(len(zbinsc)/njoin))
    for i in range(int(len(zbinsc)/njoin)):
        nTi[i] = np.sum(nti[((zmeanti>=njoin*i)&(zmeanti<njoin*i+njoin))])
    return nTi

def corr_metric(pzT):
    pzT = pzT/pzT.sum()
    overlap = np.zeros((pzT.shape[1],pzT.shape[1]))
    for i in range(pzT.shape[1]):
        for j in range(pzT.shape[1]):
            overlap[i,j] = np.sum(pzT[:,i]*pzT[:,j])
    overlap = overlap/np.diagonal(overlap)[:,None]
    metric = np.linalg.det(overlap)**(float(pzT.shape[1])/float(len(zbinsc)))
    return metric


def rebin_Ncz(Ncz_original):
    
    #Add two more bins 4.00 and 4.01 with all value 0 inside. This is to ensure interpolation 4.005 will success.
    zeros_column = np.zeros((Ncz_original.shape[0], 2))
    Ncz_original = np.hstack((Ncz_original, zeros_column)) #(Ngal * 402)
    
    Ncz_integrated = np.zeros((len(Ncz_original),len(zbinsc)))
    zbinsc_laigle = np.arange(0,4.02,0.01)
    Nint = 5
    #Go from zmin = -0.04 to pileup z<0.01
    #zbinsc_integrate = np.arange(min_z+delta_z/Nint/2., max_z+delta_z/Nint,delta_z/Nint)
    zbinsc_integrate = np.arange(min_z+delta_z/Nint/2. -delta_z, max_z+delta_z/Nint,delta_z/Nint)
    
    interp_func = interp1d(zbinsc_laigle, Ncz_original, kind='linear', axis=1, bounds_error=False, fill_value=0)
    values = interp_func(zbinsc_integrate)
    values = values.reshape((Ncz_original.shape[0], len(zbinsc) + 1, Nint))
    Ncz_integrated = np.sum(values,axis=2)
    # Pileup all bins in [-0.04, 0.01] (in laigle, this is the 0th bin and half of 1st bin)
    Ncz_integrated[:, 1] += Ncz_integrated[:, 0]
    Ncz_integrated = np.delete(Ncz_integrated, 0, axis=1)

    return Ncz_integrated



def rebin_cosmos(cosmos_original):
    
    #Original redshift and its bins
    redshift_original = cosmos_original[zpdfcols].values  #(Ngal * 400)
    #Add two more bins 4.00 and 4.01 with all value 0 inside. This is to ensure interpolation 4.05 will success.
    zeros_column = np.zeros((redshift_original.shape[0], 2))
    redshift_original = np.hstack((redshift_original, zeros_column)) #(Ngal * 402)
    zbinsc_laigle = np.arange(0,4.02,0.01)
    
    #Define interpolation bins
    Nint = 5
    zbinsc_integrate = np.arange(min_z+delta_z/Nint/2.,max_z+delta_z/Nint,delta_z/Nint)
    #interpolate for all galaxies
    interp_func = interp1d(zbinsc_laigle, redshift_original, kind='linear', axis=1, bounds_error=False, fill_value=0)
    values = interp_func(zbinsc_integrate)
    #sum 0.01 to 0.05
    num_galaxies = redshift_original.shape[0]
    values = values.reshape((num_galaxies, len(zbinsc), Nint))
    redshift_integrated = np.sum(values, axis=2)     #(Ngal * 80)
    
    #remake zpdf columns
    cosmos_drop = cosmos_original.drop(columns=zpdfcols)
    new_zpdfcols = ["Z{:.3f}".format(s).replace(".","_") for s in zbinsc]
    zpdf_df = pd.DataFrame(redshift_integrated, columns=new_zpdfcols)
    cosmos_new = pd.concat([cosmos_drop, zpdf_df], axis=1)
    
    return cosmos_new


#####################################################################
### Functions for assign deep SOM with Perturbed ZPU deep fluxes
#####################################################################

# Get flux and flux errors for all deep bands that is detected at least once in Balorg, and separate based on deep field == COSMOS.
def get_fluxes(balrog_file, deep_file):     
    
    bands = ['U','G','R','I','Z','J','H','K']
    
    # get deep field galaxies that is selected in balrog

    balrog_data = pickle.load(open(balrog_file, 'rb'), encoding='latin1')
    ids_b = balrog_data['ID'].values

    deep_data = pickle.load(open(deep_file, 'rb'), encoding='latin1')
    ids_d = deep_data['ID'].values

    match_d_b = np.isin(ids_d,ids_b)  #size of deep field with True if in Balrog and False otherwise
    print(ids_d[match_d_b].shape)

    fluxes_d = np.zeros((int(match_d_b.sum()),len(bands)))
    fluxerrs_d = np.zeros((int(match_d_b.sum()),len(bands)))
    
    for i,band in enumerate(bands):
        print(i,band)
        fluxes_d[:,i] = deep_data['BDF_FLUX_DERED_CALIB_%s'%band][match_d_b]
        fluxerrs_d[:,i] = deep_data['BDF_FLUX_ERR_DERED_CALIB_%s'%band][match_d_b]

    #get fields  - each deep field has different zp uncertainty
    fields = deep_data['FIELD'].values[match_d_b]
    deep_id = deep_data['ID'].values[match_d_b]
    
    assert len(fluxes_d)==len(fields)
    return fluxes_d, fluxerrs_d, fields, deep_id



###################################################
### Functions for Plotting nz realizations
###################################################

def get_means(zmeans, hists):
    """Returns means for each tomo bin
    """
    
    means = []
    for i in range(np.shape(hists)[0]):
        mean = []
        for j in range(np.shape(hists)[1]):
            mean += [np.sum(hists[i][j]*zmeans)/np.sum(hists[i][j])]
        means += [mean]

    return means

def get_mean_sigma_onenz(zmeans, hists):
    """Returns means and sigmas for each tomo bin
    """
    means = np.zeros(np.shape(hists)[0])
    sigmas = np.zeros(np.shape(hists)[0])

    for i in range(np.shape(hists)[0]):
        means[i] = np.sum(hists[i] * zmeans) / np.sum(hists[i])
        sigmas[i] = np.sqrt(np.sum(hists[i] * (zmeans - means[i]) ** 2) / np.sum(hists[i]))

    return means, sigmas

