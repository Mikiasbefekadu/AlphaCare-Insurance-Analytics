import pandas as pd

def load_data(file_path):
  """Loads the data from the specified file."""
  try:
    data = pd.read_csv(filepath_or_buffer=file_path, delimiter='|', low_memory=False)
    return data
  except FileNotFoundError:
    print(f"Error: File not found at: {file_path}")
    return None
def calculate_profit(data):
  """Calculates and adds the 'Profit' column to the DataFrame."""
  data['Profit'] = data['TotalPremium'] - data['TotalClaims']
  return data
def group_zip_codes(data, n_digits = 3, subset=False, n_subset_groups = 5):
    """Creates a new column by taking the first n digits of the postal code, and can get a subset of postal codes."""
    if subset:
      zip_codes = data.groupby('PostalCode')['TotalClaims'].mean().sort_values()
      zip_codes_low = zip_codes.head(n_subset_groups).index.to_list()
      zip_codes_high = zip_codes.tail(n_subset_groups).index.to_list()
      data = data[data['PostalCode'].isin(zip_codes_low + zip_codes_high)]
      data['PostalCodeGroup'] = data['PostalCode'].astype(str)
    else:
        data['PostalCodeGroup'] = data['PostalCode'].astype(str).str[:n_digits]
    return data