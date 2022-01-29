# @Time : 2020/8/30 14:43
# @Author : Scewiner, Xu, Mingzhou
# @File: map_to_bpe.py
# @Software: PyCharm
import argparse
import pickle as pkl
import codecs
import fastBPE

def main(args):
    graph = pkl.load(open(args.input,'rb'))
    for c in range(len(graph)):
        for i in range(len(graph[c])):
            w = bpe.apply([graph[c][i]])
            w = w[0].split()
            if isinstance(w,list):
                graph[c][i]=w
            else:
                graph[c][i]=[w]

    with open(args.output,'wb') as out:
        pkl.dump(graph,out)

if __name__ == '__main__':
    params = argparse.ArgumentParser()
    params.add_argument('-input')
    params.add_argument('-output')
    params.add_argument('-bpe')
    args = params.parse_args()
    bpe = fastBPE.fastBPE(args.bpe)
    main(args)
