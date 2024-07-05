# 处理一个包含序列数据的文件夹，并将处理后的数据保存到另一个文件夹中
# AdaptiveBiotech immuneAnalyzer生成的原始TCR库测序数据
# 提取文件中frequencyCount (%)：频率计数（%）从大到小排序的thr行数并输出到新文件.tsv
# aminoAcid：氨基酸序列、vMaxResolved：v最大解析度、frequencyCount (%)：频率计数（%）索引：1、5、3
# CASSPGTGNEKLFF	TCRBV11-02*02	1.29810800757895e-05
import os
from os.path import exists
import numpy as np
import csv
from csv import reader
import sys

"""
# 从命令行参数接收输入文件夹路径和输出文件夹路径，以及一个阈值
indir = sys.argv[1]  # 从命令行参数获取第一个参数（即输入文件夹路径）
outdir = sys.argv[2]  # 从命令行参数获取第二个参数（即输出文件夹路径）
thr = 10000
"""


# 处理一个包含TSV文件的目录，并将处理后的文件输出到另一个目录
# 输入：TCR repertoire sequencing data原始TCR库测序数据文件夹路径，输出文件夹路径，数据行阈值
def PrepareAdaptiveFile(indir, outdir, thr=10000):
    ffs = os.listdir(indir)
    for ff in ffs:
        if ".tsv" not in ff:
            continue
        ff0 = ff
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        str1 = "TestReal-"
        newff = outdir + "/" + str1 + ff0
        '''
        if exists(newff) == True:
            continue
        '''
        # 筛选出文件中满足特定条件的数据行
        csv_reader = reader(open(indir + "/" + ff, "r"),
                            delimiter="\t", quotechar='"')
        ddnew = []
        for row in csv_reader:
            # row[1]为aminoAcid氨基酸序列
            if "*" not in row[1]:
                if "X" not in row[1]:
                    if (len(row[1]) >= 10) and (len(row[1]) <= 24):
                        # row[5]为cdr3Length：CDR3长度
                        if "unresolved" not in row[5]:
                            # C开头，以F结尾，且中间包含任意字符（至少一个）的字符串
                            if (row[1][0] == "C") and (
                                row[1][-1] == "F"
                            ):  # if '^C.+F$' not in row[1]:
                                ddnew.append(row)
        ddnew = np.array(ddnew)
        # 根据ddnew二维数组中第4列（索引为3）的值进行排序#count (reads)：计数（读取次数）升序从小到大
        sorted_array = ddnew[ddnew[:, 3].astype(float).argsort()]
        # 反转，降序从大到小
        reverse_array = sorted_array[::-1]
        # 提取小于阈值数量的数据行
        if len(reverse_array) > thr:
            # 提取前thr行，不包括第thr行
            col1 = reverse_array[0:thr, 1]  # aminoAcid：氨基酸序列
            col2 = reverse_array[0:thr, 5]  # vMaxResolved：v最大解析度
            col3 = reverse_array[0:thr, 3]  # frequencyCount (%)：频率计数（%）
        else:
            col1 = reverse_array[:, 1]
            col2 = reverse_array[:, 5]
            col3 = reverse_array[:, 3]
        c = zip(col1, col2, col3)
        first_row = "amino_acid\tv_gene\tfrequency"
        f = open(newff, "w")
        f.write(first_row)
        f.write("\n")
        f.close()
        with open(newff, "a") as f:
            writer = csv.writer(f, delimiter="\t")
            writer.writerows(c)


def run_DAF(argvs):
    indir = argvs[0]
    outdir = argvs[1]
    thr = 10000
    PrepareAdaptiveFile(indir, outdir, thr)


if __name__ == "__main__":
    run_DAF(sys.argv[1:])
