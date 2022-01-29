# @Time : 2021/5/14 00:58
# @Author : Scewiner, Xu, Mingzhou
# @File: make_docid.py
# @Software: PyCharm
import argparse
import json

def load_doc(path):
    data = {}
    count = 0
    flag = 0
    with open(path,'r',encoding='utf8') as f:
        tmp = []
        for line in f:
            if '<title>' in line:
                if len(tmp)!=0:
                    data[count] = tmp
                    tmp = []
                    count += 1
                    flag = 0
                continue
            tmp.append(flag)
            flag += 1
        if len(tmp)!=0:
            data[count] = tmp
    return data

def make_id(data):
    docs = {}
    count = 0
    for k,v in data.items():
        print(len(v))
        for vs in v:
            docs[count] = [k,vs]
            count += 1
    return docs

def main(args):
    data = load_doc(args.input)
    data = make_id(data)
    with open(args.output,'w',encoding='utf8') as out:
        json.dump(data,out)


if __name__ == '__main__':
    params = argparse.ArgumentParser()
    params.add_argument('-i','--input')
    params.add_argument('-o','--output')
    args = params.parse_args()
    main(args)
