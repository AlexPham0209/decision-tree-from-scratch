import os
import csv

class Person: 
    def __init__(self, args):
        num, credit_history, debt, collateral, credit_risk = args

        

PATH = os.path.join('data', '')

if __name__ == "__main__":
    data = []
    with open(PATH, mode ='r') as file:
        f = csv.reader(file)
        data_set = map(lambda entry: Entry(f), f)
        
    print()
        