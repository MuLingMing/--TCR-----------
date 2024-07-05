"""
    A、用户没有原始的 TCR 库测序数据：
使用 SampleData 文件夹中的数据作为示例输入:
要处理输入文件，只需调用Script_DeepCAT.sh：
  bash  Script_DeepCAT.sh -t SampleData/Control
  bash  Script_DeepCAT.sh -t SampleData/Cancer
DeepCAT 将输出两个文件，Cancer_score_Control.txt 和 Cancer_score_Cancer.txt。

    B、用户拥有原始的 TCR 库测序数据：
DeepCAT目录中创建一个文件夹（your_folder_name），并将输入的“tsv”文件放在该位置
  mkdir your_folder_name
  bash  Script_DeepCAT.sh -r your_folder_name
运行Script_DeepCAT.sh后，将创建一个输出文件Cancer_score.txt，其中包含输入文件的名称和相应的癌症评分。
"""
"""
使用方法：
    python Script_DeepCAT.py -t/-r 文件夹名称
"""

# If raw TCR repertoire sequencing data are available please place data
# in the "Input" folder
import os
import sys
import subprocess
import json
var1 = "./iSMARTm_Input"
# If raw TCR repertoire sequencing data are not available
# please use our Sample Data as an example
# python DeepCAT_modif.py $var3  $var4
var2 = "./iSMARTm_Output"
var3 = "DeepCAT_CHKP/tmp/"
#


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


if __name__ == "__main__":
    # argv[0]是脚本名称，argv[1:]是传入的参数
    # argv[1]是参数-t或-r，argv[2]是文件夹名称
    args = sys.argv[1:]  # args[0]是参数-t或-r，args[1]是文件夹名称
    if args[0] == "-r":
        if os.listdir(args[1]):
            if not os.path.exists(var2):
                os.makedirs(var2)
            run_script("PrepareAdaptiveFile.py", [args[1], var1])
            with open('config.json', 'r') as file:
                data = json.load(file)
                opt = data.get("iSMARTm", [])["iSMARTm_options"]
                argvs = []
                for k, v in opt.items():
                    if v:
                        argvs.extend([k, v])
            run_script("iSMARTm.py", argvs)
        else:
            print("Error! The", args[1], "directory is empty")
            exit(1)
    elif args[0] == "-t":
        if not os.path.exists("DeepCAT_CHKP/"):
            print(
                "Directory DeepCAT_CHKP DOES NOT exists. Please unzip  DeepCAT_CHKP.zip file"
            )
        run_script("DeepCAT.py", [args[1], var3, "-t"])
