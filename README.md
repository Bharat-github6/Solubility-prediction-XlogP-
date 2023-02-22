# Solubility-prediction-XlogP


import all the libraries
```from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Crippen import MolLogP
from rdkit.Chem import Descriptors
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
```

upload teh data.Data can be found here https://drive.google.com/file/d/1_rpECl9WVkY0sjOfEOsBZE2wdNHfIUy9/view?usp=share_link
```
data = Chem.SDMolSupplier('a.sdf')
```

Use RDKit to extract the information from teh data. The data was downloaded form pubchem and it is in sdf format and convert to DataFrame


```
data = []
for mol in suppl:
    #if mol is not None:
        # Get the molecular weight
        mw = Chem.rdMolDescriptors.CalcExactMolWt(mol)

        # Get the XLogP
        logp = MolLogP(mol)

        # Get the number of rotatable bonds
        rotatable_bonds = Chem.rdMolDescriptors.CalcNumRotatableBonds(mol)

        # Get the formal charge
        charge = Chem.rdmolops.GetFormalCharge(mol)

        # Get the polar surface area
        psa = Chem.rdMolDescriptors.CalcTPSA(mol)
        
        # Get the number of H-bond donors
        hbd = Descriptors.NumHDonors(mol)
        
        # Get the number of H-bond acceptors
        hba = Descriptors.NumHAcceptors(mol)

        # Get the SMILES
        smiles = Chem.MolToSmiles(mol)

        # Create a dictionary for the data
        mol_data = {
            'XLogP': logp,
            'Molecular Weight': mw,
            'smiles': smiles,
            'H-Bond Donors': hbd,
            'H-Bond Acceptors': hba,
            'charge': charge,
            'rotatable_bonds': rotatable_bonds,
            'polar surface area': psa
        }

        # Add the dictionary to the list
        data.append(mol_data)

# Create the pandas DataFrame
df = pd.DataFrame(data)

```
Split and train the model

```

X = df[['Molecular Weight', 'H-Bond Donors', 'H-Bond Acceptors', 'charge','rotatable_bonds', 'polar surface area']]
y = df['XLogP']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

```

Calculate the Mean Squared error and r-squared value and plot it.

```
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
plt.scatter(y_test, y_pred, color='tab:blue', alpha=0.5)
plt.title("Actual vs Predicted Values")
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'k--', lw=2)

print("Mean Squared Error:", mse)
print("R-squared value:", r2)
plt.show()

```
