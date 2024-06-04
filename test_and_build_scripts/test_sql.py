import sqlite3
import glob
from tqdm import tqdm

def get_connection():
    conn = sqlite3.connect('/root/nfs/pubmed_cleaned_index/lookup_table.db')
    return conn

def get_chunk_by_id(chunk_id):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT chunk FROM chunks WHERE id = ?", (chunk_id,))
    result = cur.fetchone()
    if result:
        print(result[0])  # Print the chunk
    else:
        print("Chunk not found")
    conn.close()

# Call the function with a chunk ID
get_chunk_by_id(42)


source_files = glob.glob("/root/nfs/pubmed_cleaned/*.tsv")
# conn = get_connection()  # Establish a connection outside the loop

print("entering  main loop...")
for i, source_file in tqdm(enumerate(source_files)):
    print(f"{i}: {source_file}")


def check_missing_ids():
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        SELECT
            MIN(id) AS min_id,
            MAX(id) AS max_id,
            COUNT(*) AS total_rows,
            (MAX(id) - MIN(id) + 1) AS expected_rows,
            (MAX(id) - MIN(id) + 1) - COUNT(*) AS missing_rows
        FROM chunks;
    """)

    result = cur.fetchone()

    if result:
        min_id, max_id, total_rows, expected_rows, missing_rows = result
        print(f"Minimum ID: {min_id}")
        print(f"Maximum ID: {max_id}")
        print(f"Total rows: {total_rows}")
        print(f"Expected rows: {expected_rows}")
        print(f"Missing rows: {missing_rows}")
    else:
        print("No data found in the table")

    conn.close()

# Call the function to check missing IDs
check_missing_ids()