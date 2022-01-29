import argparse
from fairseq.data import Dictionary
import pickle as pkl
def load_dict(path):
    return Dictionary.load(open(path))
    
def load_data(path):
    return pkl.load(open(path,'rb'))

def covert_to_id(data,d):
    tmps = []
    for doc in data:
        tmp = [d.encode_line(w,add_if_not_exist=False,append_eos=False) for w in doc]
        tmps.append(tmp)
    return tmps

def main(args):
    d = load_dict(args.dict)
    data = load_data(args.input)
    data = covert_to_id(data,d)
    pkl.dump(data,open(args.output,'wb'))

if __name__ == '__main__':
    params = argparse.ArgumentParser()
    params.add_argument('-i','--input')
    params.add_argument('-o','--output')
    params.add_argument('-d','--dict')
    args = params.parse_args()
    main(args)
