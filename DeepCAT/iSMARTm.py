# 进行T细胞受体（TCR）序列的重叠群分析，以识别具有相似CDR3序列的T细胞克隆，计算V基因之间的相似性，并对两个TCR序列簇进行比较以确定它们之间的相似性。
"""
读取TCR序列数据并将其存储在字典中。
对TCR序列进行重叠群分析，以识别具有相似CDR3序列的T细胞克隆。
计算V基因之间的相似性，并将结果存储在字典中。
对两个TCR序列簇进行比较，以确定它们之间的相似性。
"""
import numpy as np
from Bio import pairwise2

# from Bio.SubsMat.MatrixInfo import blosum62
from itertools import chain
from optparse import OptionParser
import random
import time
from functools import partial
from multiprocessing import Pool

# from statsmodels.sandbox.stats.multicomp import multipletests as mlt
import sys
import os
import re

# import resource
import psutil  # windows上使用以替代linus的resource模块

import init

blosum62 = {
    ("W", "F"): 1,
    ("L", "R"): -2,
    ("S", "P"): -1,
    ("V", "T"): 0,
    ("Q", "Q"): 5,
    ("N", "A"): -2,
    ("Z", "Y"): -2,
    ("W", "R"): -3,
    ("Q", "A"): -1,
    ("S", "D"): 0,
    ("H", "H"): 8,
    ("S", "H"): -1,
    ("H", "D"): -1,
    ("L", "N"): -3,
    ("W", "A"): -3,
    ("Y", "M"): -1,
    ("G", "R"): -2,
    ("Y", "I"): -1,
    ("Y", "E"): -2,
    ("B", "Y"): -3,
    ("Y", "A"): -2,
    ("V", "D"): -3,
    ("B", "S"): 0,
    ("Y", "Y"): 7,
    ("G", "N"): 0,
    ("E", "C"): -4,
    ("Y", "Q"): -1,
    ("Z", "Z"): 4,
    ("V", "A"): 0,
    ("C", "C"): 9,
    ("M", "R"): -1,
    ("V", "E"): -2,
    ("T", "N"): 0,
    ("P", "P"): 7,
    ("V", "I"): 3,
    ("V", "S"): -2,
    ("Z", "P"): -1,
    ("V", "M"): 1,
    ("T", "F"): -2,
    ("V", "Q"): -2,
    ("K", "K"): 5,
    ("P", "D"): -1,
    ("I", "H"): -3,
    ("I", "D"): -3,
    ("T", "R"): -1,
    ("P", "L"): -3,
    ("K", "G"): -2,
    ("M", "N"): -2,
    ("P", "H"): -2,
    ("F", "Q"): -3,
    ("Z", "G"): -2,
    ("X", "L"): -1,
    ("T", "M"): -1,
    ("Z", "C"): -3,
    ("X", "H"): -1,
    ("D", "R"): -2,
    ("B", "W"): -4,
    ("X", "D"): -1,
    ("Z", "K"): 1,
    ("F", "A"): -2,
    ("Z", "W"): -3,
    ("F", "E"): -3,
    ("D", "N"): 1,
    ("B", "K"): 0,
    ("X", "X"): -1,
    ("F", "I"): 0,
    ("B", "G"): -1,
    ("X", "T"): 0,
    ("F", "M"): 0,
    ("B", "C"): -3,
    ("Z", "I"): -3,
    ("Z", "V"): -2,
    ("S", "S"): 4,
    ("L", "Q"): -2,
    ("W", "E"): -3,
    ("Q", "R"): 1,
    ("N", "N"): 6,
    ("W", "M"): -1,
    ("Q", "C"): -3,
    ("W", "I"): -3,
    ("S", "C"): -1,
    ("L", "A"): -1,
    ("S", "G"): 0,
    ("L", "E"): -3,
    ("W", "Q"): -2,
    ("H", "G"): -2,
    ("S", "K"): 0,
    ("Q", "N"): 0,
    ("N", "R"): 0,
    ("H", "C"): -3,
    ("Y", "N"): -2,
    ("G", "Q"): -2,
    ("Y", "F"): 3,
    ("C", "A"): 0,
    ("V", "L"): 1,
    ("G", "E"): -2,
    ("G", "A"): 0,
    ("K", "R"): 2,
    ("E", "D"): 2,
    ("Y", "R"): -2,
    ("M", "Q"): 0,
    ("T", "I"): -1,
    ("C", "D"): -3,
    ("V", "F"): -1,
    ("T", "A"): 0,
    ("T", "P"): -1,
    ("B", "P"): -2,
    ("T", "E"): -1,
    ("V", "N"): -3,
    ("P", "G"): -2,
    ("M", "A"): -1,
    ("K", "H"): -1,
    ("V", "R"): -3,
    ("P", "C"): -3,
    ("M", "E"): -2,
    ("K", "L"): -2,
    ("V", "V"): 4,
    ("M", "I"): 1,
    ("T", "Q"): -1,
    ("I", "G"): -4,
    ("P", "K"): -1,
    ("M", "M"): 5,
    ("K", "D"): -1,
    ("I", "C"): -1,
    ("Z", "D"): 1,
    ("F", "R"): -3,
    ("X", "K"): -1,
    ("Q", "D"): 0,
    ("X", "G"): -1,
    ("Z", "L"): -3,
    ("X", "C"): -2,
    ("Z", "H"): 0,
    ("B", "L"): -4,
    ("B", "H"): 0,
    ("F", "F"): 6,
    ("X", "W"): -2,
    ("B", "D"): 4,
    ("D", "A"): -2,
    ("S", "L"): -2,
    ("X", "S"): 0,
    ("F", "N"): -3,
    ("S", "R"): -1,
    ("W", "D"): -4,
    ("V", "Y"): -1,
    ("W", "L"): -2,
    ("H", "R"): 0,
    ("W", "H"): -2,
    ("H", "N"): 1,
    ("W", "T"): -2,
    ("T", "T"): 5,
    ("S", "F"): -2,
    ("W", "P"): -4,
    ("L", "D"): -4,
    ("B", "I"): -3,
    ("L", "H"): -3,
    ("S", "N"): 1,
    ("B", "T"): -1,
    ("L", "L"): 4,
    ("Y", "K"): -2,
    ("E", "Q"): 2,
    ("Y", "G"): -3,
    ("Z", "S"): 0,
    ("Y", "C"): -2,
    ("G", "D"): -1,
    ("B", "V"): -3,
    ("E", "A"): -1,
    ("Y", "W"): 2,
    ("E", "E"): 5,
    ("Y", "S"): -2,
    ("C", "N"): -3,
    ("V", "C"): -1,
    ("T", "H"): -2,
    ("P", "R"): -2,
    ("V", "G"): -3,
    ("T", "L"): -1,
    ("V", "K"): -2,
    ("K", "Q"): 1,
    ("R", "A"): -1,
    ("I", "R"): -3,
    ("T", "D"): -1,
    ("P", "F"): -4,
    ("I", "N"): -3,
    ("K", "I"): -3,
    ("M", "D"): -3,
    ("V", "W"): -3,
    ("W", "W"): 11,
    ("M", "H"): -2,
    ("P", "N"): -2,
    ("K", "A"): -1,
    ("M", "L"): 2,
    ("K", "E"): 1,
    ("Z", "E"): 4,
    ("X", "N"): -1,
    ("Z", "A"): -1,
    ("Z", "M"): -1,
    ("X", "F"): -1,
    ("K", "C"): -3,
    ("B", "Q"): 0,
    ("X", "B"): -1,
    ("B", "M"): -3,
    ("F", "C"): -2,
    ("Z", "Q"): 3,
    ("X", "Z"): -1,
    ("F", "G"): -3,
    ("B", "E"): 1,
    ("X", "V"): -1,
    ("F", "K"): -3,
    ("B", "A"): -2,
    ("X", "R"): -1,
    ("D", "D"): 6,
    ("W", "G"): -2,
    ("Z", "F"): -3,
    ("S", "Q"): 0,
    ("W", "C"): -2,
    ("W", "K"): -3,
    ("H", "Q"): 0,
    ("L", "C"): -1,
    ("W", "N"): -4,
    ("S", "A"): 1,
    ("L", "G"): -4,
    ("W", "S"): -3,
    ("S", "E"): 0,
    ("H", "E"): 0,
    ("S", "I"): -2,
    ("H", "A"): -2,
    ("S", "M"): -1,
    ("Y", "L"): -1,
    ("Y", "H"): 2,
    ("Y", "D"): -3,
    ("E", "R"): 0,
    ("X", "P"): -2,
    ("G", "G"): 6,
    ("G", "C"): -3,
    ("E", "N"): 0,
    ("Y", "T"): -2,
    ("Y", "P"): -3,
    ("T", "K"): -1,
    ("A", "A"): 4,
    ("P", "Q"): -1,
    ("T", "C"): -1,
    ("V", "H"): -3,
    ("T", "G"): -2,
    ("I", "Q"): -3,
    ("Z", "T"): -1,
    ("C", "R"): -3,
    ("V", "P"): -2,
    ("P", "E"): -1,
    ("M", "C"): -1,
    ("K", "N"): 0,
    ("I", "I"): 4,
    ("P", "A"): -1,
    ("M", "G"): -3,
    ("T", "S"): 1,
    ("I", "E"): -3,
    ("P", "M"): -2,
    ("M", "K"): -1,
    ("I", "A"): -1,
    ("P", "I"): -3,
    ("R", "R"): 5,
    ("X", "M"): -1,
    ("L", "I"): 2,
    ("X", "I"): -1,
    ("Z", "B"): 1,
    ("X", "E"): -1,
    ("Z", "N"): 0,
    ("X", "A"): 0,
    ("B", "R"): -1,
    ("B", "N"): 3,
    ("F", "D"): -3,
    ("X", "Y"): -1,
    ("Z", "R"): 0,
    ("F", "H"): -1,
    ("B", "F"): -3,
    ("F", "L"): 0,
    ("X", "Q"): -1,
    ("B", "B"): 4,
}


t0 = time.time()
sys.setrecursionlimit(1000000)

"""
从给定的序列中提取指定长度的motif
    seq：要处理的序列，类型为字符串。
    m：指定motif的长度，默认为6。
    gap：指定motif之间的间隔，默认为1。
    strip：指定是否对序列进行去头去尾处理，默认为True。
"""


# 直接调用：IndexSeqByMotif
def SplitMotif(seq, m=6, gap=1, strip=True):
    # gap is either 0 or 1
    if strip:
        ns = len(seq)
        if ns >= 10:
            seq = seq[2: (ns - 2)]
        else:
            return []
    ns = len(seq)
    if ns <= 6:
        return []
    motifList = [seq[xx: (xx + m)] for xx in range(0, ns - m + 1)]
    if gap == 1:
        for ii in range(1, m):
            motifList += [
                seq[xx: (xx + ii)] + "." + seq[(xx + ii + 1): (xx + m + 1)]
                for xx in range(0, ns - m)
            ]
    return motifList


"""
根据给定的序列和序列ID对序列进行索引，以便根据motif对序列进行分组
    seqs：一个包含多个序列的字符串列表。
    seqIDs：一个与seqs长度相同的序列ID列表。
    m：指定motif的长度。默认值为6。
    gap：指定motif之间的gap大小。默认值为1。
"""


# 直接调用：ObtainCL
def IndexSeqByMotif(seqs, seqIDs, m=6, gap=1):
    Ns = len(seqs)
    seqDict = {}
    for ii in range(0, Ns):
        ss = seqs[ii]
        MM = SplitMotif(ss, m=m, gap=gap)
        seqDict[seqIDs[ii]] = MM
    motifDict = {}
    for kk in seqDict.keys():
        vv = seqDict[kk]
        for mm in vv:
            if mm in motifDict:
                motifDict[mm].append(kk)
            else:
                motifDict[mm] = [kk]
    motifDictNew = {}
    for kk in motifDict:
        if len(motifDict[kk]) == 1:
            continue
        motifDictNew[kk] = motifDict[kk]
    return motifDictNew


"""
用于生成一个互share motif图（Motif Graph），用于分析序列之间的相似性和相关性
    mD：一个包含多个motif的列表。
    seqs：一个包含多个序列的字符串列表。
    seqIDs：一个与seqs长度相同的序列ID列表。
"""


# 直接调用：ObtainCL
def GenerateMotifGraph(mD, seqs, seqIDs):
    SeqShareGraph = {}
    mDL = {}
    for kk in mD:
        vv = mD[kk]
        LL = []
        for v in vv:
            LL.append(len(seqs[v]))
        mDL[kk] = LL
    for kk in mD:
        vv = mD[kk]
        LL = mDL[kk]
        nv = len(vv)
        for ii in range(0, nv):
            id_1 = vv[ii]
            L1 = LL[ii]
            for jj in range(ii, nv):
                if jj == ii:
                    continue
                id_2 = vv[jj]
                L2 = LL[jj]
                if L2 != L1:
                    continue
                if id_1 not in SeqShareGraph:
                    SeqShareGraph[id_1] = [id_2]
                elif id_2 not in SeqShareGraph[id_1]:
                    SeqShareGraph[id_1].append(id_2)
                if id_2 not in SeqShareGraph:
                    SeqShareGraph[id_2] = [id_1]
                elif id_1 not in SeqShareGraph[id_2]:
                    SeqShareGraph[id_2].append(id_1)
    return SeqShareGraph


"""
非递归的深度优先搜索（DFS）算法
    graph：表示图的数据结构。
    start：搜索的起始节点。
非递归的深度优先搜索（DFS）算法，针对无向图的
因为图中相邻的顶点是由边决定的，而不是由顶点决定的。
如果图是有向的，那么应该使用递归的DFS算法，因为需要考虑顶点的访问顺序。
"""


# 直接调用：IdentifyMotifCluster
def dfs(graph, start):
    """
    Non-resursive depth first search
    """
    visited = set()
    stack = [start]
    while stack:
        vertex = stack.pop()
        if vertex not in visited:
            visited.add(vertex)
            stack.extend(set(graph[vertex]) - visited)

    return visited


"""
用于识别给定稀疏矩阵表示的序列共享图（SSG）中的motif聚类
    SSG：表示序列共享图的稀疏矩阵表示。
"""


# 直接调用：obtainCL
def IdentifyMotifCluster(SSG):
    # Input SeqShareGraph dictionary representation of sparse matrix
    POS = list(SSG.keys())
    NP = len(POS)
    ClusterList = []
    tmpL = list(chain(*ClusterList))
    count = 0
    """
        def LoadComm(STACK,cur_ii):
            if cur_ii in STACK:
                    return
            else:
                    STACK.append(cur_ii)
                    vv=SSG[cur_ii]
                    for v in vv:
                        #v_idx=POS.index(v)
                        if v not in STACK:
                            LoadComm(STACK,v)
            return STACK
    """
    for ii in POS:
        if ii not in tmpL:
            # STACK=LoadComm([],ii)
            STACK = dfs(SSG, ii)
            ClusterList.append(list(STACK))
            tmpL = list(chain(*ClusterList))
            count += 1
            if count % 200 == 0:
                print("    Solved %d clusters" % (count))
    """
        ClusterList_ss=[]
        for cc in ClusterList:
            CL=[]
            for pp in cc:
                CL.append(POS[pp])
            ClusterList_ss.append(CL)
    """
    return ClusterList


"""
解析一个名为fname的FASTA文件
FASTA是一种用于表示核苷酸或蛋白质序列的格式，其中每个序列都有一个标题（以'>'开头）和一个序列字符串。
"""


# 直接调用：PreCalculateVgeneDist
def ParseFa(fname):
    InputStr = open(fname).readlines()
    FaDict = {}
    seq = ""
    for line in InputStr:
        if line.startswith(">"):
            if len(seq) > 0:
                FaDict[seqHead] = seq
                seq = ""
            seqHead = line.strip()
        else:
            seq += line.strip()
    if seqHead not in FaDict:
        FaDict[seqHead] = seq
    return FaDict


cur_dir = os.path.dirname(os.path.realpath(__file__)) + "/"
print(cur_dir)

"""
计算人类TRAV和TRBV基因库中V基因之间的相似性，这段代码仅在需要时运行一次
读取一个名为Imgt_Human_TRBV.fasta的FASTA文件，其中包含人类TRAV和TRBV基因库的V基因序列。
将VScore字典中的所有键（V基因对）写入一个名为VgeneScores.txt的文件中
格式为V1\tV2\tscore，其中V1和V2是V基因对，score是它们之间的相似性分数。
"""


# 直接调用：main
def PreCalculateVgeneDist(VgeneFa="Imgt_Human_TRBV.fasta"):
    # Only run one time if needed
    FaDict = ParseFa(cur_dir + VgeneFa)
    VScore = {}
    CDR1Dict = {}
    CDR2Dict = {}
    for kk in FaDict:
        if "|" in kk:
            VV = kk.split("|")[1]
        else:
            VV = kk[1:]
        CDR1Dict[VV] = FaDict[kk][26:37]  # Imgt CDR1: 27 - 38
        CDR2Dict[VV] = FaDict[kk][55:64]  # Imgt CDR2: 56 - 65
    Vkeys = []
    Vkeys = CDR1Dict.keys()
    nn = len(Vkeys)
    for ii in range(0, nn):
        V1 = list(Vkeys)[ii]
        s1_CDR1 = CDR1Dict[V1]
        s1_CDR2 = CDR2Dict[V1]
        for jj in range(ii, nn):
            V2 = list(Vkeys)[jj]
            s2_CDR1 = CDR1Dict[V2]
            s2_CDR2 = CDR2Dict[V2]
            score1 = SeqComparison(s1_CDR1, s2_CDR1)
            score2 = SeqComparison(s2_CDR2, s2_CDR2)
            # print score1+score2
            VScore[(V1, V2)] = score1 + score2
    gg = open("VgeneScores.txt", "w")
    for kk in VScore:
        vv = VScore[kk]
        line = kk[0] + "\t" + kk[1] + "\t" + str(vv) + "\n"
        gg.write(line)
    gg.close()


"""
在给定的序列Seq中插入最多两个间隔
    Seq表示要插入间隔的序列，n表示要插入的间隔数量。
    返回值是一个列表，其中包含所有可能的插入结果。
"""


# 直接调用：NHLocalAlignment，DefineDDcor
def InsertGap(Seq, n):
    # Insert n gaps to Seq; n<=2
    if n == 0:
        return [Seq]
    ns = len(Seq)
    SeqList = []
    if n == 1:
        for kk in range(0, ns + 1):
            SeqNew = Seq[0:kk] + "-" + Seq[kk:]
            SeqList.append(SeqNew)
    if n == 2:
        for kk in range(0, ns + 1):
            SeqNew = Seq[0:kk] + "-" + Seq[kk:]
            for jj in range(0, ns + 2):
                SeqNew0 = SeqNew[0:jj] + "-" + SeqNew[jj:]
                SeqList.append(SeqNew0)
    return SeqList


# 双氨基酸基序相似性字典
# 对于每对双氨基酸序列，计算它们在所有样本中的频率
# 使用频率计算相关系数

DDcor = init.DDcor

COR_0005 = 0.005

"""
计算两个双氨基酸基序之间的相关系数。读取预定义字典DDcor，其中键是氨基酸序列对，值是相关系数。
    输入参数是两个氨基酸序列m1和m2，输出是相似分数。
    m1、m2为双氨基酸基序。
"""


# 直接调用：SeqComparison_Di
def GetCor(m1, m2):
    # Given di amino acid motifs m1 and m2, get correlation
    if "-" in m1 or "-" in m2:
        return (0.1, 0.9)
    if "*" in m1 or "*" in m2:
        return (0.05, 0.95)  # Arbitrary low score for gap
    if "." in m1 or "." in m2:
        return (0.05, 0.95)  # Arbitrary low score for gap
    COR = DDcor[(m1, m2)]
    return COR


"""
比较两个序列（如氨基酸序列或核苷酸序列）之间的相似性。
    输入参数是两个序列s1和s2，以及一个可选的gap分数gap
    输出是一个整数，表示两个序列之间的相似度。
"""


# 直接调用：PreCalculateVgeneDist、NHLocalAlignment
def SeqComparison(s1, s2, gap=-6):
    n = len(s1)
    CorList = []
    score = 0
    for kk in range(0, n):
        aa = s1[kk]
        bb = s2[kk]
        if aa in [".", "-", "*"] or bb in [".", "-", "*"]:
            if aa != bb:
                score += gap
            continue
        if aa == bb:
            score += min(4, blosum62[(aa, aa)])
            continue
        KEY = (aa, bb)
        if KEY not in blosum62:
            KEY = (bb, aa)
        if KEY not in blosum62:
            print(KEY)
            raise "Non-standard amino acid coding!"
        score += blosum62[KEY]
        CorList.append(blosum62[KEY])
    return score


"""
计算两个序列之间的相似度，其中允许进行氨基酸替换，采用双氨基酸基序对比
BLOSUM62矩阵是一个用于比较两个氨基酸的相似性的表格，其中包含了不同氨基酸之间的相似度分数
    输入参数是两个氨基酸序列s1和s2，以及一个可选的gap分数gap
    输出是一个整数，表示两个序列之间的相似度。
"""


# 直接调用：NHLocalAlignment
def SeqComparison_Di(s1, s2, gap=-6):
    # Older version that allows di amino acid replacement.
    n = len(s1)
    CorList = []
    score = 0
    for kk in range(0, n - 1):
        m1 = s1[kk: (kk + 2)]
        m2 = s2[kk: (kk + 2)]
        Cor = GetCor(m1, m2)
        CorList.append(Cor)
        aa = s1[kk]
        if kk == 0:
            if Cor >= COR_0005:
                bb = s1[kk]
            else:
                bb = s2[kk]
        else:
            Cor1 = CorList[kk - 1]
            if Cor1 >= COR_0005 or Cor >= COR_0005:
                bb = s1[kk]
            else:
                bb = s2[kk]
        if aa == "-" or bb == "-":
            score += gap
            continue
        if aa == "*" or bb == "*":
            score += gap
            continue
        if aa == "." or bb == ".":
            if aa == "." and bb == ".":
                continue
            else:
                score += gap
                continue
        if aa == bb:
            score += min(4, blosum62[(aa, aa)])
            continue
        KEY = (aa, bb)
        if KEY not in blosum62:
            KEY = (bb, aa)
        if KEY not in blosum62:
            print(KEY)
            raise "Non-standard amino acid coding!"
        score += blosum62[KEY]
    aa = s1[n - 1]
    bb = s2[n - 1]
    if aa in [".", "-", "*"] or bb in [".", "-", "*"]:
        if not (aa == "." and bb == "."):
            score += gap
        else:
            score += 0
    else:
        if aa == bb:
            # score+= min(4,blosum62[(aa,aa)])
            score += blosum62[(aa, aa)]
        else:
            KEY = (aa, bb)
            if KEY not in blosum62:
                KEY = (bb, aa)
            if KEY not in blosum62:
                print(KEY)
                raise "Non-standard amino acid coding!"
            score += blosum62[KEY]
    return score


"""
计算两个序列之间的局部对齐得分。
    输入参数序列Seq1和Seq2，gap_thr（gap threshold，空格阈值）、gap（gap penalty，空格惩罚）和Di（是否使用Di值）。
    输出参数为局部对齐得分。
"""


# 直接调用：falign、PWalign、CompareClusters
def NHLocalAlignment(Seq1, Seq2, gap_thr=1, gap=-6, Di=False):
    n1 = len(Seq1)
    n2 = len(Seq2)
    if n1 < n2:
        Seq = Seq1
        Seq1 = Seq2
        Seq2 = Seq
        nn = n2 - n1
    else:
        nn = n1 - n2
    if nn > gap_thr:
        return -1
    # alns=pairwise2.align.localms(Seq1,Seq2,m,s,g,ge)
    # 给Seq2添加gap，使之与Seq1长度相同
    SeqList1 = [Seq1]
    SeqList2 = InsertGap(Seq2, nn)
    alns = []
    SCOREList = []
    for s1 in SeqList1:
        for s2 in SeqList2:
            if Di:
                SCOREList.append(SeqComparison_Di(s1, s2, gap))
            else:
                SCOREList.append(SeqComparison(s1, s2, gap))
    """
                alns.append((s1,s2))
        SCOREList=[]
        for seq in SeqList:
            s1=aln[0]
            s2=aln[1]
            SCORE=SeqComparison(s1,s2)
            SCOREList.append(SCORE)
    """
    maxS = max(SCOREList)
    # ALN=alns[np.where(np.array(SCOREList)==maxS)[0][0]]
    return maxS


# 未完成，无调用，封装函数
def fun_map(p, f):
    # Fake function for passing multiple arguments to Pool.map()
    return f(*p)


"""
对两个序列进行本地序列比对
    xx：一个包含两个整数的列表，表示要进行比对的两个序列的索引。
    st：一个整数，表示序列比对开始的位置。
    VScore：一个字典，存储了所有可能的V基因对之间的评分。
    eqs：一个包含所有序列的字符串列表。
    Vgene：一个包含所有V基因的字符串列表。
    UseV：一个布尔值，表示是否使用V基因评分。
    gapn：一个整数，表示允许的gap数量。
    gap：一个浮点数，表示gap的分数。
    输出：一个浮点数，表示两个序列在指定位置开始时的得分。
计算两个序列在指定位置开始时的得分。这个得分包括序列比对得分和V基因评分。
"""


# 无调用
def falign(xx, st, VScore={}, Seqs=[], Vgene=[], UseV=True, gapn=1, gap=-6):
    ii = xx[0]
    jj = xx[1]
    V1 = Vgene[ii]
    V2 = Vgene[jj]
    mid1 = Seqs[ii][st:-2]
    mid2 = Seqs[jj][st:-2]
    if UseV:
        if V2 == V1:
            V_score = 4
        else:
            Vkey = (V1, V2)
            if Vkey not in VScore:
                Vkey = (V2, V1)
            if Vkey not in VScore:
                # print("V gene not found!")
                VScore = 0
            else:
                V_score = VScore[Vkey] / 20.0
    else:
        V_score = 4.0
    aln = NHLocalAlignment(mid1, mid2, gapn, gap)
    score = aln / float(max(len(mid1), len(mid2))) + V_score
    return score


"""
对给定的序列进行比对，并返回一个字典，其中键是两个序列的索引，值是它们之间的比对得分
    Seqs：一个包含所有序列的字符串列表。
    Vgene：一个包含所有V基因的字符串列表。
    ID：一个包含所有序列的索引的字符串列表。
    VScore：一个字典，存储了所有可能的V基因对之间的评分。
    gap：一个浮点数，表示gap的分数。
    gapn：一个整数，表示允许的gap数量。
    UseV：一个布尔值，表示是否使用V基因评分。
    cutoff：一个整数，表示要保留的比对得分阈值。
    Nthread：一个整数，表示要使用的线程数量。
    Di：一个布尔值，表示是否进行双向比对。
    输出：一个字典，其中键是两个序列的索引，值是它们之间的比对得分。
"""


# 直接调用：ObtainCL
def PWalign(
    Seqs,
    Vgene=[],
    ID=[],
    VScore={},
    gap=-6,
    gapn=1,
    UseV=True,
    cutoff=7,
    Nthread=1,
    Di=False,
):
    # Wrapper function
    ns = len(Seqs)
    if ns != len(Vgene):
        if len(Vgene) == 0:
            Vgene = [""] * ns
            ID = range(0, ns)
        else:
            raise "Incompatible variable gene number!"
    # sorted函数对z进行排序，key参数设置为一个函数，该函数接收一个元组作为输入，并返回该元组的第一个元素的长度。最后，len(pair[0])作为排序的依据，对z进行排序
    # pair为匿名参数(Seqs, Vgene, ID)
    z = sorted(zip(Seqs, Vgene, ID), key=lambda pair: len(pair[0]))
    Seqs = [x for x, y, t in z]
    Vgene = [x for y, x, t in z]
    ID = [x for t, y, x in z]
    del z
    PWscore = {}
    st = 4
    if not UseV:
        st = 2
    t1 = time.time()
    if Nthread == 1:
        # 序列对两两进行比对
        for ii in range(0, ns):
            V1 = Vgene[ii]
            if ii % 100 == 0:
                t2 = time.time()
                # print('%d: Time elapsed %f' %(ii, t2-t1))
            for jj in range(ii, ns):
                if ii == jj:
                    continue
                V2 = Vgene[jj]
                mid1 = Seqs[ii][st:-2]
                mid2 = Seqs[jj][st:-2]
                if UseV:
                    if V2 == V1:
                        V_score = 4
                    else:
                        Vkey = (V1, V2)
                        if Vkey not in VScore:
                            Vkey = (V2, V1)
                        if Vkey not in VScore:
                            # print("V gene not found!")
                            continue
                        else:
                            V_score = (
                                VScore[Vkey] / 20.0
                            )  # Take the floor of the float number
                else:
                    V_score = 4.0
                aln = NHLocalAlignment(mid1, mid2, gapn, gap, Di)
                # print aln
                #            J_score=NHLocalAlignment(Jend1,Jend2,gap=False)[0]
                score = aln / float(max(len(mid1), len(mid2))) + V_score
                if score >= cutoff:
                    PWscore[(ii, jj)] = 1
    else:
        # Multi-thread processing
        p = Pool(Nthread)
        XX = []
        for ii in range(0, ns):
            for jj in range(ii, ns):
                if ii == jj:
                    continue
                else:
                    XX.append([ii, jj])
        para = []
        for xx in XX:
            para.append((xx, st, VScore, Seqs, Vgene, UseV, gapn, gap))
        pl_out = p.map(partial(fun_map, f=falign), para)
        p.close()
        p.join()
        # End multiple processing
        for kk in range(0, len(XX)):
            score = pl_out[kk]
            if score >= cutoff:
                PWscore[(XX[kk][0], XX[kk][1])] = 1

    return (PWscore, Seqs, Vgene, ID)


"""
识别CDR3序列对之间的相似性，并将它们分组到cluster中。通过设置相似性得分的阈值，可以控制cluster的大小。
    PWscore: 比对结果
    cutoff: 阈值
    输出: 识别出的clusterList
"""


# 直接调用：ObtainCL
def IdentifyCDR3Clusters(PWscore, cutoff=7):
    POS = np.array(list(PWscore.keys()))[
        np.where(np.array(list(PWscore.values())) == 1)
    ]
    if len(POS) <= 0:
        # print("Too few clustered CDR3s! Please check your repertoire data.")
        return []
    POS = list(POS)
    POS = np.array([list(map(lambda x: x[0], POS)),
                   list(map(lambda x: x[1], POS))])
    uniquePos = list(set(list(POS[0]) + list(POS[1])))
    ClusterList = []
    tmpL = list(chain(*ClusterList))

    def LoadComm(STACK, cur_ii):
        if cur_ii in STACK:
            return
        else:
            STACK.append(cur_ii)
            vv = list(POS[1][np.where(POS[0] == cur_ii)]) + list(
                POS[0][np.where(POS[1] == cur_ii)]
            )
            for v in vv:
                LoadComm(STACK, v)
        return STACK

    for ii in uniquePos:
        if ii in tmpL:
            continue
        else:
            STACK = LoadComm([], ii)
            ClusterList.append(STACK)
            tmpL = list(chain(*ClusterList))
    return ClusterList


"""
将两个聚类结果（CLinfo1和CLinfo2）进行比较，找出可能的合并方案
    CLinfo1: 聚类结果1
    CLinfo2: 聚类结果2
    VScore: 比对结果
    gapn: 允许的gap数量
    gap: 允许的gap长度
    cutoff: 阈值
    输出: 可能的合并方案
"""


# 直接调用：main
def CompareClusters(
    CLinfo1, CLinfo2, VScore, gapn=1, gap=-6, cutoff=7, UseV=True, Di=False
):
    CL1 = CLinfo1[0]
    CL2 = CLinfo2[0]
    Seqs1 = CLinfo1[1]
    Seqs2 = CLinfo2[1]
    Vgene1 = CLinfo1[2]
    Vgene2 = CLinfo2[2]
    n1 = len(CL1)
    n2 = len(CL2)
    # print "Processing %d * %d clusters" %(n1,n2)
    MergedCL = []
    MergedSeq = []
    MergedVgene = []
    for ii in range(0, n1):
        # print ii
        seqs1 = list(np.array(Seqs1)[CL1[ii]])
        VG1 = list(np.array(Vgene1)[CL1[ii]])
        L1 = np.median(list(map(lambda x: len(x), seqs1)))
        for jj in range(0, n2):
            seqs2 = list(np.array(Seqs2)[CL2[jj]])
            VG2 = list(np.array(Vgene2)[CL2[jj]])
            L2 = np.median(list(map(lambda x: len(x), seqs2)))
            if L2 <= L1 - 1 or L2 >= L1 + 1:
                continue
            Scores = []
            st = 4
            if not UseV:
                st = 2
            for tt1 in zip(seqs1, VG1):
                ss1 = tt1[0]
                vv1 = tt1[1]
                mid1 = ss1[st:-2]
                for tt2 in zip(seqs2, VG2):
                    ss2 = tt2[0]
                    vv2 = tt2[1]
                    mid2 = ss2[st:-2]
                    Score = NHLocalAlignment(mid1, mid2, gapn, gap, Di) / float(
                        max(len(mid1), len(mid2))
                    )
                    if UseV:
                        if vv1 == vv2:
                            V_score = 4
                        else:
                            Vkey = (vv1, vv2)
                            if Vkey not in VScore:
                                Vkey = (vv2, vv1)
                            if Vkey not in VScore:
                                # print("V gene not found!")
                                V_score = 0
                            else:
                                V_score = (
                                    VScore[Vkey] / 20.0
                                )  # Take the floor of the float number
                    else:
                        V_score = 4.0
                    Score += V_score
                    # print ss1, ss2, Score
                    Scores.append(Score)
            Scores_sorted = sorted(Scores, reverse=True)
            if Scores_sorted[0] >= cutoff and Scores_sorted[1] >= cutoff:
                # print [ii,jj]
                MergedCL.append([ii, jj])
                MergedSeq.append((seqs1, seqs2))
                MergedVgene.append((VG1, VG2))
    return (MergedCL, MergedSeq, MergedVgene)


"""
从输入文件中提取CDR3序列，并根据CDR3序列之间的相似性进行聚类
    InputFile：输入文件路径，格式为TSV，其中包含CDR3序列及其对应的重链V基因。
    VScore：V基因之间的相似性分数。
    gap：允许的gap数量。
    gapn：gap的分数。
    cutoff=7：聚类中的最小相似性阈值。
    UseV=True：是否使用V基因进行聚类。
    outDir="./"：输出文件夹路径。
    Nthread=1：并行处理线程数。
    Di=False：是否使用DI算法进行聚类。
    返回值：一个包含聚类结果的列表。
    CL, Seqs, Vgene, PWscore, CDR3Dict
"""


# 直接调用：main
def ObtainCL(
    InputFile, VScore, gap, gapn, cutoff=7, UseV=True, outDir="./", Nthread=1, Di=False
):
    outDir = outDir + "/motif_group/"
    if not os.path.exists(outDir):
        os.makedirs(outDir)
    ff = open(InputFile)
    OutF = outDir + "/" + re.sub(".txt", "", InputFile.split("/")[-1])
    Seqs0 = []
    Vgene0 = []
    ID = []
    count = 0
    ALLLines = ff.readlines()
    CDR3Dict = {}
    for line in ALLLines[1:]:  # get rid of header line
        ww = line.strip().split("\t")
        if ww[0] == "":
            continue
        Seqs0.append(ww[0])
        if UseV:
            Vgene0.append(ww[1])
        else:
            Vgene0.append("")
        CDR3Dict[count] = ww[1:]
        ID.append(count)
        count += 1
    print("Building motif index")
    mD = IndexSeqByMotif(Seqs0, ID)
    print("Generating motif sharing graph")
    SSG = GenerateMotifGraph(mD, Seqs0, ID)
    print("Dividing motif sharing graph")
    mClusters = IdentifyMotifCluster(SSG)
    g = open(OutF + "_ClusteredCDR3s_" + str(cutoff) + ".txt", "w")
    g.write(ALLLines[0].strip() + "\t" + "Group" + "\n")
    gr = 0
    CL = []
    Seqs = []
    Vgene = []
    PWscore = {}
    for mID in mClusters:
        mSeqs = []
        mVgene = []
        for mm in mID:
            mSeqs.append(Seqs0[mm])
            mVgene.append(Vgene0[mm])
            # print("  Processing %d sequences." %(len(mSeqs)))
        TMP = PWalign(
            mSeqs,
            mVgene,
            mID,
            VScore,
            gap,
            gapn,
            UseV=UseV,
            cutoff=cutoff,
            Nthread=Nthread,
            Di=Di,
        )
        PWscore = TMP[0]
        Seqs = TMP[1]
        Vgene = TMP[2]
        ID = TMP[3]
        CL = IdentifyCDR3Clusters(PWscore, cutoff=cutoff)
        for cl in CL:
            gr += 1
            for ss in cl:
                cdr3 = Seqs[ss]
                tmpID = ID[ss]
                Line = (
                    "\t".join([cdr3] + CDR3Dict[tmpID] + [str(gr)]) + "\n"
                )  # Must add statistical significance estimation
                g.write(Line)
    g.close()
    return (CL, Seqs, Vgene, PWscore, CDR3Dict)


"""
解析一个名为CLfile的文件。这个文件包含了从Clonify软件生成的CDR3序列和它们对应的V基因信息。
将文件中的数据分割成三个部分：V基因序列列表（Vgene）、CDR3序列列表（Seqs）和组信息列表（CL）。
    输出：CL（组信息列表）、Seqs（CDR3序列列表）、Vgene（V基因序列列表）
"""


# 直接调用：main
def ParseCLFile(CLfile):
    ff = open(CLfile)
    ALLLines = ff.readlines()
    Seqs = []
    Vgene = []
    CL = []
    gp = 1
    groups = []
    count = 0
    for line in ALLLines[1:]:
        ww = line.strip().split("\t")
        cdr3 = ww[0]
        vv = ww[1]
        Seqs.append(cdr3)
        Vgene.append(vv)
        ID = int(ww[-1])  # 序列编号ID
        if ID > gp:
            CL.append(groups)
            groups = []
            gp += 1
        groups.append(count)
        count += 1
    CL.append(groups)
    return (CL, Seqs, Vgene)


# 解析命令行参数
# 指定输入文件夹、单个文件、输入文件列表、阈值、输出目录、gap penalty、最大gap数量、变量基因序列文件、是否保留对齐矩阵、是否比较个体之间的交互、参考cohort和线程数等参数
# 直接调用：main
def CommandLineParser():
    parser = OptionParser()
    print(
        """
iSMART is a highly specific tools for dividing TCR beta chain repertoire sequencing
data into antigen-specific groups. Similarity between different repertoires is also
compared through commonly shared CDR3 groups. iSMART is developed by Li lab at UTSW.
All rights reserved.
Input columns:
1. CDR3 amino acid sequence (Starting from C, ending with the first F/L in motif [FL]G.G)
2. Variable gene name in Imgt format: TRBVXX-XX*XX
3. Joining gene name (optional)
4. Frequency (optional)
5. Other information (optional)
"""
    )
    # -d   Input repertoire sequencing file directory重排测序文件目录.\iSMARTm_Input
    # Directory
    parser.add_option(
        "-d",
        "--directory",
        dest="Directory",
        default=".\iSMARTm_Input",
        help="Input repertoire sequencing file directory. Please make sure that all the files in the directory are input files.",
    )
    # -f   Input single file of CDR3 sequences for grouping用于分组的 CDR3 序列单个文件
    # File
    parser.add_option(
        "-f",
        "--file",
        dest="File",
        default="",
        help="Input single file of CDR3 sequences for grouping",
    )
    # -F    a file containing the full path to all the files包含所有文件完整路径的文件
    # files
    parser.add_option(
        "-F",
        "--fileList",
        dest="files",
        default="",
        help="Alternative input: a file containing the full path to all the files. If given, overwrite -d and -f option",
    )
    # -t    Threshold for calling similar CDR3 groups相似 CDR3 组的阈值7.5
    # thr
    parser.add_option(
        "-t",
        "--threshold",
        dest="thr",
        default=7.5,
        help="Threshold for calling similar CDR3 groups. The higher the more specific.",
    )
    # -o    Output directory for intermediate and final outputs中间和最终输出的输出目录.\iSMARTm_Output
    # OutDir
    parser.add_option(
        "-o",
        "--output",
        dest="OutDir",
        default=".\iSMARTm_Output",
        help="Output directory for intermediate and final outputs.",
    )
    # -g    Gap penalty for CDR3 alignment间隙惩罚-6
    # Gap
    parser.add_option(
        "-g", "--GapPenalty", dest="Gap", default=-6, help="Gap penalty,default= -6"
    )
    # -n    Maximum number of gaps allowed when performing alignment最大间隙数1
    # GapN
    parser.add_option(
        "-n",
        "--GapNumber",
        dest="GapN",
        default=1,
        help="Maximum number of gaps allowed when performing alignment. Max=1, default=1",
    )
    # -V    IMGT Human beta variable gene sequencesIMGT_Human_TRBV可变基因序列Imgt_Human_TRBV.fasta
    # VFa
    parser.add_option(
        "-V",
        "--VariableGeneFa",
        dest="VFa",
        default="Imgt_Human_TRBV.fasta",
        help="IMGT Human beta variable gene sequences",
    )
    # -W    KeepPairwiseMatrix.保留配对得分矩阵False
    # PW
    parser.add_option(
        "-W",
        "--KeepPairwiseMatrix",
        dest="PW",
        default=False,
        action="store_true",
        help="If true, iSMART will keep the pairwise alignment score matrix. Make sure you have enough disk space when dealing with large samples. Default: False",
    )
    # -I    CrossInteraction.交叉交互False
    # I
    parser.add_option(
        "-I",
        "--CrossInteraction",
        dest="I",
        default=False,
        action="store_true",
        help="If true, iSMART takes the clonal group files to compute sharing between individuals.",
    )
    # -C    CrossComparison.交叉比较False
    # C
    parser.add_option(
        "-C",
        "--CrossComparison",
        dest="C",
        default=False,
        action="store_true",
        help="If true, iSMART compares all the CDR3 clusters in the input to the directory specified in -r.",
    )
    # -r    ReferenceCohort.C的比较参考群体
    # R
    parser.add_option(
        "-r", "--referenceCohort", dest="R", default="", help="See -C option"
    )
    # -v    VariableGene.可变基因True
    # V
    parser.add_option(
        "-v",
        "--VariableGene",
        dest="V",
        default=True,
        action="store_false",
        help="If False, iSMART will omit variable gene information and use CDR3 sequences only. This will yield reduced specificity. The cut-off will automatically become the current value-4.0",
    )
    # -N    NumberOfThreads.线程数1
    # NN
    parser.add_option(
        "-N",
        "--NumberOfThreads",
        dest="NN",
        default=1,
        help="Number of threads for multiple processing. Not working so well.",
    )
    # -D    UseDiAAmat.使用二元氨基酸矩阵False
    # Di
    parser.add_option(
        "-D",
        "--UseDiAAmat",
        dest="Di",
        default=False,
        action="store_true",
        help="If True, iSMART will use a predefined di-amino acid substitution matrix in sequence comparison.",
    )
    return parser.parse_args()


def iSMARTm_main():
    # 解析命令行参数
    (opt, _) = CommandLineParser()
    FileDir = opt.Directory
    if len(FileDir) > 0:
        files = os.listdir(FileDir)
        files0 = []
        for ff in files:
            if os.path.splitext(ff)[1] == ".tsv":
                ff = FileDir + "/" + ff
                files0.append(ff)
        files = files0
    else:
        files = []
    File = opt.File
    if len(File) > 0:
        files = [File]
    FileList = opt.files
    if len(FileList) > 0:
        files = []
        fL = open(FileList)
        for ff in fL.readlines():
            files.append(ff.strip())
    VFa = opt.VFa
    # 人类TRAV和TRBV基因库中V基因之间的相似性
    PreCalculateVgeneDist(VFa)
    vf = open("./VgeneScores.txt")  # Use tcrDist's Vgene 80-score calculation
    VScore = {}
    while 1:
        line = vf.readline()
        if len(line) == 0:
            break
        ww = line.strip().split("\t")
        VScore[(ww[0], ww[1])] = int(ww[2])
    Gap = int(opt.Gap)
    Gapn = int(opt.GapN)
    cutoff = float(opt.thr)
    OutDir = opt.OutDir
    if not os.path.exists(OutDir):
        os.mkdir(OutDir)
    PW = opt.PW
    II = opt.I
    CC = opt.C
    RR = opt.R
    VV = opt.V
    NN = int(opt.NN)
    Di = False
    DataDict = {}
    # CrossComparison.交叉比较
    if CC:
        print("Compare input file with reference data")
        RefFiles = os.listdir(RR)
        gg = open(OutDir + "CrossReference.txt", "w")
        gg.write("CDR3\tVgene\tIndividualGroupID\tCrossGroupID\tSampleID\n")
        for f1 in files:
            TMP1 = ParseCLFile(f1)
            print("Processing %s" % (f1))
            for fr in RefFiles:
                TMPr = ParseCLFile(RR + fr)
                sys.stdout.write(".")
                sys.stdout.flush()
                MC = CompareClusters(TMP1, TMPr, VScore,
                                     Gapn, Gap, cutoff, VV, Di)
                n = len(MC[0])
                for kk in range(0, n):
                    gg1 = MC[0][kk][0]
                    gg2 = MC[0][kk][1]
                    ww1 = MC[1][kk][0]
                    ww2 = MC[1][kk][1]
                    vv1 = MC[2][kk][0]
                    vv2 = MC[2][kk][1]
                    nw1 = len(ww1)
                    nw2 = len(ww2)
                    ii = files.index(f1)
                    jj = RefFiles.index(fr)
                    for ss in range(0, nw1):
                        line = (
                            ww1[ss]
                            + "\t"
                            + vv1[ss]
                            + "\t"
                            + str(gg1)
                            + "\t"
                            + str(groupID)
                            + "\t"
                            + files[ii]
                            + "\n"
                        )
                        gg.write(line)
                    for ss in range(0, nw2):
                        line = (
                            ww2[ss]
                            + "\t"
                            + vv2[ss]
                            + "\t"
                            + str(gg2)
                            + "\t"
                            + str(groupID)
                            + "\t"
                            + files[jj]
                            + "\n"
                        )
                        gg.write(line)
                    groupID += 1
            print("")
        gg.close()
        return
    for ff in files:
        print("Processing %s" % (ff))
        # CrossInteraction.交叉交互False
        if II:
            TMP = ParseCLFile(ff)
        else:
            TMP = ObtainCL(ff, VScore, Gap, Gapn, cutoff, VV, OutDir, NN, Di)
        if PW:
            PWscore = TMP[0]
            OutF = OutDir + ff + "_PWscores.txt"
            gg = open(OutF, "w")
            line = "\t".join(TMP[1]) + "\n"
            gg.write(line)
            line = "\t".join(TMP[2]) + "\n"
            gg.write(line)
            for ii in range(0, len(PWscore)):
                line = "\t".join(map(str, list(PWscore[ii]))) + "\n"
                gg.write(line)
            gg.close()
        DataDict[ff] = TMP

    gg = open(
        OutDir + "/CrossComparison.txt", "w"
    )  # Must add statistical significance estimation later
    gg.write("CDR3\tVgene\tIndividualGroupID\tCrossGroupID\tSampleID\n")
    if len(files) >= 2:
        nn = len(files)
        groupID = 0
        print("Pairwise comparison of %d repertoires" % (nn))
        for ii in range(0, nn):
            print(ii)
            CLinfo1 = DataDict[files[ii]]
            for jj in range(ii, nn):
                if jj == ii:
                    continue
                CLinfo2 = DataDict[files[jj]]
                MC = CompareClusters(
                    CLinfo1, CLinfo2, VScore, Gapn, Gap, cutoff)
                n = len(MC[0])
                for kk in range(0, n):
                    gg1 = MC[0][kk][0]
                    gg2 = MC[0][kk][1]
                    ww1 = MC[1][kk][0]
                    ww2 = MC[1][kk][1]
                    vv1 = MC[2][kk][0]
                    vv2 = MC[2][kk][1]
                    nw1 = len(ww1)
                    nw2 = len(ww2)
                    for ss in range(0, nw1):
                        line = (
                            ww1[ss]
                            + "\t"
                            + vv1[ss]
                            + "\t"
                            + str(gg1)
                            + "\t"
                            + str(groupID)
                            + "\t"
                            + files[ii]
                            + "\n"
                        )
                        gg.write(line)
                    for ss in range(0, nw2):
                        line = (
                            ww2[ss]
                            + "\t"
                            + vv2[ss]
                            + "\t"
                            + str(gg2)
                            + "\t"
                            + str(groupID)
                            + "\t"
                            + files[jj]
                            + "\n"
                        )
                        gg.write(line)
                    groupID += 1
    gg.close()


def get_memory_usage():
    process = psutil.Process()  # 获取当前进程对象
    mem_info = process.memory_info()  # 获取内存使用情况
    return mem_info.rss / 1024 / 1024  # 字节转换为MB


if __name__ == "__main__":
    iSMARTm_main()
    print("Total time elapsed: {:2f}".format(time.time() - t0))
    print(
        "Maximum memory usage: {:2f} MB".format(get_memory_usage())
        # % (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1000000)
    )
