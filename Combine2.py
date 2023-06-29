#Code to combine two netcdf files and turn them into a numpy
import numpy as np
#from netCDF4 import Dataset
#
#def load_ncdf_data(filename):
#    with Dataset(filename, 'r') as ds:
#        return {var: np.array(ds[var][:]) for var in ds.variables}
#
#def merge_ncdf_files(file1, file2):
#    data1 = load_ncdf_data(file1)
#    data2 = load_ncdf_data(file2)  
#    # Merge the two dictionaries
#    merged_data = {**data1, **data2}
#    for key in data1.keys():
#        if key in data2:
#            merged_data[key] = np.concatenate((data1[key], data2[key]))          
#    return merged_data
#
#def save_to_npz(data, filename):
#    np.savez(filename, **data)
#
#if __name__ == "__main__":
#    file1 = "./Testset/1 (2).ncdf"
#    file2 = "./Testset/1 (3).ncdf"
#    merged_data = merge_ncdf_files(file1, file2)
#    save_to_npz(merged_data, 'merged_data.npz')


import glob
from scipy.io import netcdf_file

#create an array of all the days
#days = np.arange(20100102, 20100332, 1)          #Start and finish+1 dates but I dont think this is going to work on the full dataset of

# Create empty arrays to store the data
time_yr_all = []
time_mo_all = []
time_dy_all = []
time_hr_all = []
time_mt_all = []
time_sc_all = []
ID_all = []
mlat_all = []
mlon_all = []
glat_all = []
glon_all = []
mcolat_all = []
mlt_all = []
sza_all = []
decl_all = []
dbn_nez_all = []
dbe_nez_all = []
dbz_nez_all = []
dbn_geo_all = []
dbe_geo_all = []
dbz_geo_all = []
time_extent_all = []

days = [day for day in range(0, 364)]

for day in days:
    datafiles = glob.glob('2010/*.ncdf')  # /home/w21011465/Internship/
    datafiles.sort()  # put in ascending order
    print(datafiles[day])                      #and here
    
    # Assuming 'datafiles' contains netCDF file paths
    ncdf = netcdf_file(datafiles[day], 'r')    #change the dates here
    
    time_yr = np.copy(ncdf.variables['time_yr'][:])
    time_mo = np.copy(ncdf.variables['time_mo'][:])
    time_dy = np.copy(ncdf.variables['time_dy'][:])
    time_hr = np.copy(ncdf.variables['time_hr'][:])
    time_mt = np.copy(ncdf.variables['time_mt'][:])
    time_sc = np.copy(ncdf.variables['time_sc'][:])
    ID = np.copy(ncdf.variables['id'][:])
    mlat = np.copy(ncdf.variables['mlat'][:])
    mlon = np.copy(ncdf.variables['mlon'][:])
    glat = np.copy(ncdf.variables['glat'][:])
    glon = np.copy(ncdf.variables['glon'][:])
    mcolat = np.copy(ncdf.variables['mcolat'][:])
    mlt = np.copy(ncdf.variables['mlt'][:])
    sza = np.copy(ncdf.variables['sza'][:])
    decl = np.copy(ncdf.variables['decl'][:])
    dbn_nez = np.copy(ncdf.variables['dbn_nez'][:])
    dbe_nez = np.copy(ncdf.variables['dbe_nez'][:])
    dbz_nez = np.copy(ncdf.variables['dbz_nez'][:])
    dbn_geo = np.copy(ncdf.variables['dbn_geo'][:])
    dbe_geo = np.copy(ncdf.variables['dbe_geo'][:])
    dbz_geo = np.copy(ncdf.variables['dbz_geo'][:])
    time_extent = np.copy(ncdf.variables['time_extent'][:])
    
    # Append the data to the respective storage arrays
    time_yr_all.append(time_yr)
    time_mo_all.append(time_mo)
    time_dy_all.append(time_dy)
    time_hr_all.append(time_hr)
    time_mt_all.append(time_mt)
    time_sc_all.append(time_sc)
    ID_all.append(ID)
    mlat_all.append(mlat)
    mlon_all.append(mlon)
    glat_all.append(glat)
    glon_all.append(glon)
    mcolat_all.append(mcolat)
    mlt_all.append(mlt)
    sza_all.append(sza)
    decl_all.append(decl)
    dbn_nez_all.append(dbn_nez)
    dbe_nez_all.append(dbe_nez)
    dbz_nez_all.append(dbz_nez)
    dbn_geo_all.append(dbn_geo)
    dbe_geo_all.append(dbe_geo)
    dbz_geo_all.append(dbz_geo)
    time_extent_all.append(time_extent)

# Convert the lists to numpy arrays
time_yr_all = np.concatenate(time_yr_all, axis=0)
time_mo_all = np.concatenate(time_mo_all, axis=0)
time_dy_all = np.concatenate(time_dy_all, axis=0)
time_hr_all = np.concatenate(time_hr_all, axis=0)
time_mt_all = np.concatenate(time_mt_all, axis=0)
time_sc_all = np.concatenate(time_sc_all, axis=0)
ID_all = np.concatenate(ID_all, axis=0)
mlat_all = np.concatenate(mlat_all, axis=0)
mlon_all = np.concatenate(mlon_all, axis=0)
glat_all = np.concatenate(glat_all, axis=0)
glon_all = np.concatenate(glon_all, axis=0)
mcolat_all = np.concatenate(mcolat_all, axis=0)
mlt_all = np.concatenate(mlt_all, axis=0)
sza_all = np.concatenate(sza_all, axis=0)
decl_all = np.concatenate(decl_all, axis=0)
dbn_nez_all = np.concatenate(dbn_nez_all, axis=0)
dbe_nez_all = np.concatenate(dbe_nez_all, axis=0)
dbz_nez_all = np.concatenate(dbz_nez_all, axis=0)
dbn_geo_all = np.concatenate(dbn_geo_all, axis=0)
dbe_geo_all = np.concatenate(dbe_geo_all, axis=0)
dbz_geo_all = np.concatenate(dbz_geo_all, axis=0)
time_extent_all = np.concatenate(time_extent_all, axis=0)

# Save the combined data to compressed npz files
np.savez_compressed('./2010data/data_all.npz',
                    time_yr_all=time_yr_all, time_mo_all=time_mo_all, time_dy_all=time_dy_all, 
                    time_hr_all=time_hr_all, time_mt_all=time_mt_all, time_sc_all=time_sc_all, 
                    ID_all=ID_all, mlat_all=mlat_all,
                    mlon_all=mlon_all, glat_all=glat_all, glon_all=glon_all,
                    mcolat_all=mcolat_all, mlt_all=mlt_all, sza_all=sza_all,
                    decl_all=decl_all, dbn_nez_all=dbn_nez_all,
                    dbe_nez_all=dbe_nez_all, dbz_nez_all=dbz_nez_all,
                    dbn_geo_all=dbn_geo_all, dbe_geo_all=dbe_geo_all,
                    dbz_geo_all=dbz_geo_all, time_extent_all=time_extent_all)


#archive_path = 'D:/Internship/RBSP_b_2012.npz'    #code to view a particular file
#data = np.load(archive_path, allow_pickle=True)
#print(data.files)
