import pandas as pd
from pathlib import Path

csv_path = Path(r"c:\Users\romeo\Desktop\CS8813\DLC\Fly32-Lorenzo-2026-03-16\labeled-data\fly32_train\CollectedData_Lorenzo.csv")
h5_path = csv_path.with_suffix('.h5')

df = pd.read_csv(csv_path, header=[0, 1, 2], index_col=0)
df.to_hdf(h5_path, key='df_with_missing', mode='w')

print(f'Wrote: {h5_path}')
print(f'Shape: {df.shape}')
