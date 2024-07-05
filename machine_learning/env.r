chooseCRANmirror() # 选择一个CRAN镜像
setRepositories() # 设置软件包，直接全选
install.packages("ggm")#然后就搭配好环境了，接着就可以开始运行文件测试了
library()
.packages(all.available = TRUE)
system.file(package = "dendextend")
system.file(package = "dbscan")
system.file(package = "ggbiplot")
system.file(package = "ggm")
system.file(package = "limma")
system.file(package = "tidyverse")

library(dendextend)
library(dbscan)
library(ggbiplot)
library(ggm)
library(limma)
library(tidyverse)
library(tibble)
library(dplyr)
