DeepCAT,python=3.7
pip install biopython tensorflow==1.14 matplotlib scikit-learn psutil keras
运行start.py，可选择回车选取默认参数

可能出现的BUG/Warning：
FutureWarning: Passing (type, 1) or ‘1type’ as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / ‘(1,)type’.
TensorFlow版本太低了，不兼容高版本的numpy
解决办法：
pip uninstall numpy#卸载高版本numpy
pip install numpy==1.16#安装低版本numpy

PS：iSMARTm.py在本报告中并未涉及


mathine_learning,R=4.0,4.4
环境配置查看env.r，可选用4.4版本R程序
运行main.r文件