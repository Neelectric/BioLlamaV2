import dask.bag as db
import json
from time import time
from dask.distributed import Client

def load_and_process_json(index_path):
    client = Client()
    client.dashboard_link
    file_path = index_path + "lookup_table_definitive.json"
    time_before_json_load = time()
    bag = db.read_text(file_path, blocksize=100 * 1000 * 1000).map(json.loads)

    # Filter the bag to get items with specific key values
    key_to_filter = "your_key"
    value_to_filter = "your_value"
    filtered_bag = bag.filter(lambda x: x.get(key_to_filter) == value_to_filter)

    # Map the filtered bag to extract specific key-value pairs
    keys_to_extract = ["1", "2", "3"]
    extracted_data = filtered_bag.map(lambda x: {k: x.get(k) for k in keys_to_extract})

    # Compute the extracted data and return the result
    result = extracted_data.compute()
    return result

if __name__ == "__main__":
    index_path = '/root/nfs/pubmed_cleaned_index/'
    data = load_and_process_json(index_path)
    print(data)
