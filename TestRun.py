from helpers import extract_elements_and_positions


datafile='/Users/manocharbonnier/Desktop/Computational Material Design/Projet 1 /mof_solubility/anions/ClO4.csv'

El,Pos=extract_elements_and_positions(datafile)
X=Pos[:,0]
print(X)