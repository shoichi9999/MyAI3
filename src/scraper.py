# Import required libraries
import requests
import pandas as pd

# Constants
BASE_URL = 'https://example.com/api'

# Function to fetch data

def fetch_data():
    response = requests.get(BASE_URL)
    if response.status_code == 200:
        return response.json()
    return None

# Function to process data

def process_data(data):
    df = pd.DataFrame(data)
    return df.describe()

# Main function

if __name__ == '__main__':
    data = fetch_data()
    if data:
        stats = process_data(data)
        print(stats)