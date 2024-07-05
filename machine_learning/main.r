# demo
inputcsv <- "./ExampleData.csv"
k <- 4
outputgz <- "Output/Positional_4mer.csv.gz"

source("Generating_kmer_matrices.R")
# 这将识别 CDR3 中的所有 kmer，并在每个样本中的计数，然后将它们写入 OutputFile；运行时间约 15 分钟
kmers <- NonPos_Matrix(inputcsv, k = k, OutputFile = outputgz)

source("Clustering_function.R")
# 由 1 和 2 组成的向量；1 代表病例（乳糜泻患者），2 代表对照组，其顺序与上一步生成的 kmer 矩阵/CDR3 矩阵列的顺序相同。
class <- c(rep(1, 11), rep(2, 11))
# 使用主成分 1:10 的所有可能组合对数据进行聚类，并找出最佳组合。
# 生成最优聚类图，并生成一个 csv 文件，详细说明所有可能的主成分组合的准确性，保存在输出目录中。
# 这项工作大约需要 5 分钟。
clusters <- ClusterOptim(outputgz, class, outDir = "Output")
