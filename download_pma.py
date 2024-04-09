import os 
# print local directory
print(os.getcwd())

#print ../ directory
path = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
print(path)