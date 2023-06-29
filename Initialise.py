#The initial setup of packages on a new computer/kernel

#These all have to be ran in the console
conda create -n tensorflow_env tensorflow=2
conda activate tensorflow_env

pip install matplotlib 
pip install seaborn 
pip install scikit-learn 
pip install tensorflow 
pip install cdflib 
pip install cblind 

pip install xarray 
pip install wget 
pip install netCDF4
pip install utils
pip install dask

#pip install spacepy 