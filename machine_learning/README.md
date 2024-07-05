1. 系统要求
需要 R（版本 >=3.6）和软件包：

- dendextend
- dbscan
- ggbiplot
- ggm
- limma
- 数据表
- data.table

本代码已使用 R v.3.6 和 v.4.0.0 进行了测试。


2. 安装指南
应从 https://cran.r-project.org/ 安装 R 和相关软件包。


3. 演示
将 Generating_kmer_matrices.R、Clustering_function.R 和 ExampleData.csv 文件保存在一个文件夹（工作目录）中，并创建一个子文件夹 "Output"。
启动 R，将工作目录设置为包含脚本和数据的文件夹（否则请使用完整文件路径）。在 R 中运行以下命令 

source('Generating_kmer_matrices.R')
kmers<-NonPos_Matrix('ExampleData.csv',k=4,OutputFile='Output/Positional_4mer.csv.gz')
#这将识别 CDR3 中存在的所有 kmer 及其在每个样本中的计数，然后将它们写入输出文件；运行时间约为 15 分钟


source('Clustering_function.R')

class<-c(rep(1,11),rep(2,11))  #由 1 和 2 组成的向量；1 代表病例（乳糜泻患者），2 代表对照组，其顺序与上一步生成的 kmer 矩阵/CDR3 矩阵列的顺序相同。
Clusters<-ClusterOptim('Output/Positional_4mer.csv.gz',class,outDir='Output') #使用主成分 1:10 的所有可能组合对数据进行聚类，并找出最佳组合。生成最优聚类图，并生成一个 csv 文件，详细说明所有可能的主成分组合的准确性，保存在输出目录中。这项工作大约需要 5 分钟。


4. 使用说明
要在其他数据集上运行代码，请确保原始输入 csv 文件是 CDR3 氨基酸序列 x 样本的矩阵，其中包含每个序列在每个样本中出现的次数计数。按上述方法生成不同 k 值范围（建议范围为 3-7）的 kmer 矩阵，以及位置 kmers 矩阵（用 NonPos_Matrix 代替 Pos_Matrix，并根据情况修改输出文件）。在生成的每个千分表矩阵上运行聚类步骤以确定最佳参数，并比较结果以确定最佳千分表长度和输入类型。


5. 结果
使用上述方法生成的 kmer 矩阵/CDR3 矩阵的聚类结果将保存在输出目录中。
