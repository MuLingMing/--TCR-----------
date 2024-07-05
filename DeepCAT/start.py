# pip install biopython tensorflow==1.14 matplotlib scikit-learn psutil keras

import os
import tensorflow as tf
from keras import backend as K
import subprocess
import DeepCAT
import json
import pickle

# Warning:Backend TkAgg is interactive backend. Turning interactive mode on.
import matplotlib.pyplot as plt

plt.ioff()  # 关闭交互模式

# 设置GPU数量，加速
NUM_PARALLEL_EXEC_UNITS = 6

config = tf.ConfigProto(intra_op_parallelism_threads=NUM_PARALLEL_EXEC_UNITS,
                        inter_op_parallelism_threads=1,
                        allow_soft_placement=True,
                        device_count={'CPU': NUM_PARALLEL_EXEC_UNITS})

session = tf.Session(config=config)

K.set_session(session)


os.environ["OMP_NUM_THREADS"] = str(NUM_PARALLEL_EXEC_UNITS)
os.environ["KMP_BLOCKTIME"] = "30"
os.environ["KMP_SETTINGS"] = "1"
os.environ["KMP_AFFINITY"] = "granularity=fine,verbose,compact,1,0"

#正式开始
def run_script(script_name, args):
    # 使用subprocess.Popen启动脚本并获取其stdout
    process = subprocess.Popen(
        ["python", script_name] + args, stdout=subprocess.PIPE)
    # 等待脚本执行完成
    process.wait()
    # 获取脚本的输出
    if process.stdout:
        output, error = process.communicate()
        return output, error
    else:
        print("Error: Script did not produce any output")
        exit(1)


def train(argvs):
    # 调用DeepCAT.py中的batchTrain函数进行训练
    # 参数为argvs中的四个文件路径
    defult_args = [
        "./TrainingData/TumorCDR3.txt",
        "./TrainingData/TumorCDR3_test.txt",
        "./TrainingData/NormalCDR3.txt",
        "./TrainingData/NormalCDR3_test.txt",
    ]
    if argvs == []:
        argvs = defult_args
    while len(argvs) < len(defult_args):
        argvs.append("")
    for i in range(len(argvs)):
        if argvs[i] == "" or i < len(defult_args):
            argvs[i] = defult_args[i]
    for i in range(len(argvs), 0, -1):
        if argvs[i - 1] == "":
            argvs.pop(i - 1)

    ftumor = argvs[0]
    feval_tumor = argvs[1]
    fnormal = argvs[2]
    feval_normal = argvs[3]
    PredictClassList, PredictLabelList, AUCDictList = DeepCAT.batchTrain(
        ftumor=ftumor,
        fnormal=fnormal,
        feval_tumor=feval_tumor,
        feval_normal=feval_normal,
        n=10,
        STEPs=10000,
        rate=0.33,
    )
    # 保存结果
    if not os.path.exists("./tmp/dictlist/"):
        os.makedirs("./tmp/dictlist/", exist_ok=True)
    with open("./tmp/dictlist/PredictClassList.pkl", "wb") as f:
        pickle.dump(PredictClassList, f)
    with open("./tmp/dictlist/PredictLabelList.pkl", "wb") as f:
        pickle.dump(PredictLabelList, f)
    with open("./tmp/dictlist/AUCDictList.pkl", "wb") as f:
        pickle.dump(AUCDictList, f)


def predict(argvs):
    # 调用DeepCAT.py中的predict函数进行预测
    defult_args = [
        "-t",
        "./DeepCAT_CHKP/tmp/",
        "./SampleData/Control",
        "./SampleData/Cancer",
    ]
    if argvs == []:
        argvs = defult_args
    while len(argvs) < len(defult_args):
        argvs.append("")
    for i in range(len(argvs)):
        if argvs[i] == "" or i < len(defult_args):
            argvs[i] = defult_args[i]
    for i in range(2, len(argvs)):
        if i == 2:
            if type(argvs[i]) != list:
                argvs[i] = [argvs[i]]
            continue
        if type(argvs[i]) == list:
            argvs[2] = argvs[2] + argvs[i]
        else:
            argvs[2].append(argvs[i])
    for i in range(len(argvs), 0, -1):
        if argvs[i - 1] == "":
            argvs.pop(i - 1)

    for i in argvs[2]:
        run_script("DeepCAT.py", [i, argvs[1], argvs[0]])


def iSMARTm(argvs):
    # 调用iSMARTm.py中的main函数进行预测
    defult_args = ["./TCR_repertoire_sequencing_data/"]
    if argvs == []:
        argvs = defult_args
    while len(argvs) < len(defult_args):
        argvs.append("")
    for i in range(len(argvs)):
        if argvs[i] == "" or i < len(defult_args):
            argvs[i] = defult_args[i]
    for i in range(len(argvs), 0, -1):
        if argvs[i - 1] == "":
            argvs.pop(i - 1)

    run_script("Script_DeepCAT.py", ["-r"] + argvs)


def main():
    # 主函数
    # 调用train函数进行训练
    # 调用predict函数进行预测
    # 调用iSMARTm进行预测
    function_dict = {
        "1": "predict",
        "2": "train",
        "3": "iSMARTm",
        "4": "exit",
    }
    panel_dict = {
        "initial": "Please input the command you want to run:\n"
        + "DeepCAT: 数据预测，请输入1\n"
        + "DeepCAT: 数据训练，请输入2\n"
        + "iSMARTm: T细胞重叠群分析，请输入3\n"
        + "退出程序，请输入4",
        "predict": "   用户没有原始的 TCR 库测序数据：\n"
        + "使用 SampleData 文件夹中的数据作为示例输入:\n"
        + "要处理输入文件，只需调用Script_DeepCAT.sh：\n"
        + "   -t,SampleData/Control\n"
        + "   -t,SampleData/Cancer\n"
        + "DeepCAT 将输出两个文件，Cancer_score_Control.txt 和 Cancer_score_Cancer.txt。\n"
        + "默认参数：-t,./DeepCAT_CHKP/tmp/,./SampleData/Control,./SampleData/Cancer\n"
        + "请输入参数：-t/-r,mod_path,input_path\n"
        + "选择返回：请输入back\n"
        + "选择退出：请输入exit\n",
        "train": "要从头开始训练 DeepCAT，请使用 TrainingData 文件夹中的示例数据。\n"
        + "此文件夹包含两个文件，每个文件都是来自癌症或健康个体的 CDR3 列表\n"
        + "\n"
        + "肿瘤样本序列的文件路径、评估肿瘤样本序列的文件路径、正常样本序列的文件路径、评估正常样本序列的文件路径\n"
        + "\n"
        + "默认参数：TrainingData/TumorCDR3.txt,TrainingData/TumorCDR3_test.txt,TrainingData/NormalCDR3.txt,TrainingData/NormalCDR3_test.txt,\n"
        + "请输入参数：path,path,path,path\n"
        + "选择返回：请输入back\n"
        + "选择退出：请输入exit\n",
        "iSMARTm": "T细胞受体（TCR）序列的重叠群分析\n"
        + "   用户拥有原始的 TCR 库测序数据：\n"
        + "DeepCAT目录中创建一个文件夹（your_folder_name），并将输入的“tsv”文件放在该位置\n"
        + "mkdir your_folder_name\n"
        + "   -r your_folder_name\n"
        + "运行Script_DeepCAT.sh后，将创建一个输出文件Cancer_score.txt，其中包含输入文件的名称和相应的癌症评分。\n"
        + "使用方法：\n"
        + "   文件夹名称\n"
        + "默认参数：已配置\n"
        + "参数请参照iSMARTm.py的说明，使用英文逗号划分。\n"
        + "选择返回：请输入back\n"
        + "选择退出：请输入exit\n",
        "back": "back",
        "exit": "Exit",
    }
    while True:
        print(panel_dict["initial"])
        command = input("请输入命令：")
        if command in function_dict:
            if function_dict[command] == "exit":
                exit(0)
            print(panel_dict[function_dict[command]])
            argvs = input("请输入参数：")
            if argvs == "back":
                continue
            elif argvs == "json":
                with open("config.json", "r") as file:
                    data = json.load(file)
                    selected_data = data.get(function_dict[command], [])
                    selected_list = [value for key,
                                     value in selected_data.items()]
                    argvs = selected_list
                eval(function_dict[command])(argvs)
            elif argvs == "exit":
                exit(0)
            else:
                eval(function_dict[command])(argvs.split(",") if argvs else [])
                print("Success")


if __name__ == "__main__":
    main()
