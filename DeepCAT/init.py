import os


"""
在给定的序列Seq中插入最多两个间隔
    Seq表示要插入间隔的序列，n表示要插入的间隔数量。
    返回值是一个列表，其中包含所有可能的插入结果。
"""


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
def DefineDDcor(
    train_file=["TrainingData\TumorCDR3.txt",
                "TrainingData\TumorCDR3_test.txt"],
    result_file="DDcor.txt",
):
    # 定义一个包含所有可能双氨基酸序列对的列表
    amino_acids = "ACDEFGHIKLMNPQRSTVWY"
    all_pairs = [
        (x + y, a + b)
        for x in amino_acids
        for y in amino_acids
        for a in amino_acids
        for b in amino_acids
    ]

    # 计算每对氨基酸序列之间的相关系数
    DDcor = {}
    # 初始化字典
    for pair in all_pairs:
        DDcor[pair] = 0
    # 读取样本数据
    samples = []
    for fp in train_file:
        filename = fp
        with open(filename, "r") as f:
            lines = f.readlines()
            # 去除每行末尾的换行符，将lines中的每一行字符串转为列表
            sample_datas = [line.strip().split() for line in lines]  # 对应lines
            samples.append(sample_datas)  # 对应fp

    # 计算每对双氨基酸基序之间的相关系数
    ns=0
    for sample_datas in samples:
        for sample_data in sample_datas:
            if not sample_data:
                continue
            for i in range(len(sample_data[0])-4):
                ns+=1
                if (sample_data[0][i:i+2], sample_data[0][i+2:i+4]) in all_pairs:
                    DDcor[(sample_data[0][i:i+2], sample_data[0][i+2:i+4])] +=1
    for pair in all_pairs:
        DDcor[pair] = DDcor[pair] / (ns if ns > 0 else 100000)
    '''
    for pair in all_pairs:
        freqs = []
        # 计算每对双氨基酸基序在所有样本中的频率
        for sample_data in samples:
            freq = 0
            freqn = 0
            ns = len(sample_data)
            for seqs1 in range(0, ns):
                if seqs1 == ns - 1:
                    break
                for seqs2 in range(seqs1 + 1, ns):
                    seq1 = sample_data[seqs1][0]
                    seq2 = sample_data[seqs2][0]
                    n1 = len(seq1)
                    n2 = len(seq2)
                    if pair[0] not in seq1 or pair[1] not in seq2:
                        freqn += 1
                        continue
                    # seq1和seq2补长
                    if n1 - n2 > 2 or n2 - n1 > 2:
                        continue
                    elif n1 > n2:
                        seq1 = [seq1]
                        seq2 = InsertGap(seq2, n1 - n2)
                    else:
                        seq1 = InsertGap(seq1, n2 - n1)
                        seq2 = [seq2]
                    for seq11 in seq1:
                        for seq22 in seq2:
                            freqn += 1
                            # 判断seq1和seq2中pair[0]和pair[1]的位置是否相同
                            # 如果相同，则将freq加1
                            if seq11.find(pair[0]) == seq22.find(pair[1]):
                                freq += 1
            freq = freq / (freqn if freqn > 0 else 10000)
            freqs.append(freq)
        # 使用频率替代相似性系数
        correlation = freqs[-1]
        DDcor[pair] = correlation
        print(pair, correlation)
    '''
    # 保存结果到文件
    with open(result_file, "w") as f:
        for pair, correlation in DDcor.items():
            f.write(f"{pair[0]}\t{pair[1]}\t{correlation}\n")

    return DDcor


def get_DDcor(result_file="DDcor.txt"):
    DDcor = {}
    if os.path.exists(result_file):
        with open(result_file, "r") as f:
            lines = f.readlines()
            for line in lines:
                temp = line.strip().split('\t')
                DDcor[temp[0], temp[1]] = float(temp[2])
        return DDcor
    else:
        return None


DDcor=get_DDcor()
if DDcor==None:
    DDcor = DefineDDcor()