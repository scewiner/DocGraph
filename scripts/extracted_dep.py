# @Time : 2020/8/30 12:51
# @Author : Scewiner, Xu, Mingzhou
# @File: extracted_dep.py
# @Software: PyCharm
import argparse
import json
import sys
import os


def main(args):
    tmp = json.load(open(args.input,'r',encoding='utf8'))
    tmp_dict={}
    for n,line in enumerate(tmp['sentences']):
        sys.stdout.write('\r Line num:{}'.format(n))
        sys.stdout.flush()
        tmp_dict[n] = {'dep':line['enhancedDependencies'],'pos':line['tokens']}

    del tmp

    json.dump(tmp_dict,open(args.output,'w',encoding='utf8'),sort_keys=True,indent=4,ensure_ascii=False,separators=(',',':'))
    # os.remove(args.input)

if __name__ == '__main__':
    params = argparse.ArgumentParser()
    params.add_argument('--input')
    params.add_argument('--output')
    args = params.parse_args()

    main(args)
