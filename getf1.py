'''Run pharmit on specified database with query'''

import argparse, sys, re, subprocess

parser = argparse.ArgumentParser(description='Run pharmit on specified database with query and report F1 score')
parser.add_argument("query",help="Query file (json) to search with")
parser.add_argument("db",help="Pharmit database to search")
parser.add_argument("--actives",default="actives_final.ism",help="Name of actives smiles file",required=False)
args = parser.parse_args()

try:
    num_actives = len(open(args.actives).readlines()) # count lines
    output = subprocess.check_output(f'./data/Mpro_XChem_screen_active_site_11-May-2020/pharmit dbsearch -dbdir {args.db} -in {args.query} -extra-info -max-orient=1 -reduceconfs=1',shell=True)
    output = output.decode()
    lines = output.split('\n')
    hits = 0
    tp = 0 # true positives
    for line in lines:
        vals = line.split(',') #index,rmsd, molweight, #rotbonds,name, internal stuff
        if len(vals) != 7:
            continue #skip other rows
        hits += 1
        name = vals[4]
        if 'active' in name:
            tp += 1
            
    recall = tp/num_actives
    if recall == 0: #avoid divide by zero
        precision = 0
        f1 = 0
    else:
        precision = tp/hits
        f1 = 2*(precision*recall)/(precision+recall)

    print(f'F1: {f1:5f} Recall: {recall:5f} Precision: {precision:5f}')
except:
    print('error')
