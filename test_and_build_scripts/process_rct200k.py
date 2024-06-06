### open the txt file in rct200k/train.txt

from tqdm import tqdm

# Path: test_and_build_scripts/process_rct200k.py
with open('rct200k/train.txt', 'r') as f:
    lines = f.readlines()


# at the end of the day, what is it i am trying to do here?
# what i really want, is to have a raw abstract_id starting at 0 mapped to an abstract, in a JSON
# we'll go from there
# how do i do that? well, empty lines are the delimiter between abstracts. so we can just split on that

abstracts = {}
current_abstract_id = 0
current_abstract = ""
for line in tqdm(lines):
    # if line starts with ###, skip
    if line.startswith("###"):
        continue
    if line == "\n":
        abstracts[current_abstract_id] = current_abstract
        current_abstract_id += 1
        current_abstract = ""
    else:
        current_abstract += line
print("all abstracts processed.")

# now we need to clean each abstract. any mentions of "\n" and "\t" should be removed
for abstract_id, abstract in abstracts.items():
    abstract = abstract.replace("\n", " ").replace("\t", "").replace("BACKGROUND", "").replace("OBJECTIVE", "").replace("METHODS", "").replace("RESULTS", "").replace("CONCLUSIONS", "")
    #if the string contains "Unique identifier NCT", then chop this off and everything after it
    if "Unique identifier NCT" in abstract:
        abstract = abstract.split("Unique identifier NCT")[0]
    if "ACTRN" in abstract:
        abstract = abstract.split("ACTRN")[0]
    abstracts[abstract_id] = abstract
    print(abstracts[abstract_id])
    print("-"*50)