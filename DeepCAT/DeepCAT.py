# CNN model for tumor-specific CDR3 sequence prediction

import sys
import os
import re
import csv
import pathlib
import tensorflow as tf
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

# import skimage

tf.compat.v1.logging.set_verbosity(
    tf.compat.v1.logging.ERROR)  # 设置TensorFlow的日志级别。
AAs = np.array(list("WFGAVILMPYSTNQCKRHDE"))  # 一个包含所有可能氨基酸的字符串数组，20序列对应字符串数组。
curPath = os.getcwd()  # 获取当前工作目录
# AAidx_file='AAindexNormalized.txt' ## AA index reached AUC about 61% for L=14. Worse than AdaBoost
# AAidx_file='AtchleyFactors.txt'  ## Atchley factors work worse than using 544 AA index
AAidx_file = "AAidx_PCA.txt"  # works like a charm!!!
with open(AAidx_file) as gg:
    AAidx_Names = gg.readline().strip().split("\t")  # 氨基酸的名称数组，即对应字符与数字的列表
    AAidx_Dict = {}  # 字典，存储AAindex文件中的氨基酸特征
    for ll in gg.readlines():
        ll = ll.strip().split("\t")
        AA = ll[0]
        tag = 0
        vv = []
        # 拼接文件中的特征
        for xx in ll[1:]:
            vv.append(float(xx))
        if tag == 1:
            continue
        AAidx_Dict[AA] = vv  # 键值对，存储氨基酸特征

Nf = len(AAidx_Dict["C"])  # C的特征数

pat = re.compile("[\\*_XB]")  # non-productive CDR3 patterns，匹配非产品线氨基酸


# 对输入的序列进行独热编码
# 输入：序列，列表
# 输出：独热编码，列表
def OneHotEncoding(Seq):
    if type(Seq) != list:
        Seq_aa = list(Seq)
    Ns = len(Seq_aa)
    OHE = np.zeros([20, Ns])  # 20*Ns的矩阵，假设20个类别，每个类别用Ns维的向量表示
    for ii in range(Ns):
        aa = Seq_aa[ii]
        vv = np.where(AAs == aa)  # 查找字符在表中的位置
        OHE[vv, ii] = 1
    OHE = OHE.astype(np.float32)  # 将OHE转换为float32类型
    return OHE


# 对序列进行编码
# 输入：序列，列表
# 输出：编码，列表
# 调用于GetFeatureLabels、PredictCancer
def AAindexEncoding(Seq):
    Ns = len(Seq)
    AAE = np.zeros([Ns, Nf])
    for kk in range(Ns):
        ss = Seq[kk]
        AAE[kk,] = AAidx_Dict[ss]
    AAE = np.transpose(AAE.astype(np.float32))  # 将AAE转置，并将其转换为np.float32类型
    return AAE


# 获取特征和标签
# 输入：TumorCDR3s和NonTumorCDR3s，列表
# 输出：特征和标签，字典
# 调用于batchTrain
def GetFeatureLabels(TumorCDR3s, NonTumorCDR3s):  # 肿瘤CDR3s和非肿瘤CDR3s
    # 序列长度
    nt = len(TumorCDR3s)
    nc = len(NonTumorCDR3s)
    # 氨基酸长度
    LLt = [len(ss) for ss in TumorCDR3s]
    LLt = np.array(LLt)
    LLc = [len(ss) for ss in NonTumorCDR3s]
    LLc = np.array(LLc)
    # 特征和标签
    NL = range(12, 17)
    FeatureDict = {}
    LabelDict = {}
    for LL in NL:
        # 检索LL长度的序列
        vvt = np.where(LLt == LL)[0]
        vvc = np.where(LLc == LL)[0]
        # 标签
        Labels = [1] * len(vvt) + [0] * len(vvc)
        Labels = np.array(Labels)
        Labels = Labels.astype(np.int32)
        data = []
        for ss in TumorCDR3s[vvt]:
            if len(pat.findall(ss)) > 0:
                continue
            data.append(AAindexEncoding(ss))
            # data.append(OneHotEncoding(ss))
        for ss in NonTumorCDR3s[vvc]:
            if len(pat.findall(ss)) > 0:
                continue
            data.append(AAindexEncoding(ss))
            # data.append(OneHotEncoding(ss))
        data = np.array(data)
        features = {"x": data, "LL": LL}
        FeatureDict[LL] = features
        LabelDict[LL] = Labels
    return FeatureDict, LabelDict


# 基于卷积神经网络（CNN）的模型，用于识别给定特征（CDR3序列）的标签（是否为HLA配对）
# 输入：特征和标签，字典
# 输出：模型，Estimator
def cnn_model_CDR3_LL12(features, labels, mode):
    """Model function for CNN."""
    # Input Layer输入层：将输入特征（CDR3序列）转换为适合卷积层输入的格式
    data = features["x"]
    # LL=features['LL']
    input_layer = tf.reshape(data, [-1, Nf, 12, 1])
    # Convolutional Layer #1卷积层1：使用8个3x2的卷积核（kernel size为Nf x 2）对输入数据进行卷积操作，并使用ReLU激活函数
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=8,
        kernel_size=[Nf, 2],
        padding="valid",
        activation=tf.nn.relu,
    )
    # Pooling Layer #1池化层1：使用最大池化（max pooling）操作对卷积结果进行降采样，以减少计算复杂度和参数数量
    pool1 = tf.layers.max_pooling2d(
        inputs=conv1, pool_size=[1, 2], strides=[1, 1]
    )  # stride used to be 2
    # Convolutional Layer #2 and Pooling Layer #2卷积层2：使用16个1x2的卷积核对池化结果进行卷积操作，并使用ReLU激活函数，池化层2：使用最大池化操作对卷积结果进行降采样
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=16,
        kernel_size=[1, 2],
        padding="valid",
        activation=tf.nn.relu,
    )
    pool2 = tf.layers.max_pooling2d(
        inputs=conv2, pool_size=[1, 2], strides=[1, 1])
    SHAPE = pool2.shape
    pool2_flat = tf.reshape(pool2, [-1, int(SHAPE[1] * SHAPE[2] * SHAPE[3])])
    # 全连接层：将卷积结果展平并输入全连接层，使用ReLU激活函数
    dense = tf.layers.dense(inputs=pool2_flat, units=10, activation=tf.nn.relu)
    dropout = tf.layers.dropout(
        inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN
    )
    # Logits Layer：在训练过程中随机丢弃一定比例的神经元，以防止过拟合
    logits = tf.layers.dense(inputs=dropout, units=2)
    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=logits, axis=1),
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
        # `logging_hook`.
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor"),
    }
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
    # Calculate Loss (for both TRAIN and EVAL modes)计算损失
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
    # Configure the Training Op (for TRAIN mode)训练
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss=loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)
    # Add evaluation metrics (for EVAL mode)评估
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predictions["classes"]
        )
    }
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops
    )


# 基于卷积神经网络（CNN）的模型，用于识别和分类CDR3序列
# 输入：特征和标签，字典
# 输出：模型，Estimator
def cnn_model_CDR3_LL13(features, labels, mode):
    """Model function for CNN."""
    # Input Layer
    data = features["x"]
    # LL=features['LL']
    input_layer = tf.reshape(data, [-1, Nf, 13, 1])
    # Convolutional Layer #1
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=8,
        kernel_size=[Nf, 2],
        padding="valid",
        activation=tf.nn.relu,
    )
    # Pooling Layer #1
    pool1 = tf.layers.max_pooling2d(
        inputs=conv1, pool_size=[1, 2], strides=[1, 1]
    )  # stride used to be 2
    # Convolutional Layer #2 and Pooling Layer #2
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=16,
        kernel_size=[1, 2],
        padding="valid",
        activation=tf.nn.relu,
    )
    pool2 = tf.layers.max_pooling2d(
        inputs=conv2, pool_size=[1, 2], strides=[1, 1])
    SHAPE = pool2.shape
    pool2_flat = tf.reshape(pool2, [-1, int(SHAPE[1] * SHAPE[2] * SHAPE[3])])
    dense = tf.layers.dense(inputs=pool2_flat, units=10, activation=tf.nn.relu)
    dropout = tf.layers.dropout(
        inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN
    )
    # Logits Layer
    logits = tf.layers.dense(inputs=dropout, units=2)
    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=logits, axis=1),
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
        # `logging_hook`.
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor"),
    }
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
    # Calculate Loss (for both TRAIN and EVAL modes)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss=loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)
    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predictions["classes"]
        )
    }
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops
    )


# 基于卷积神经网络（CNN）的模型，用于识别和分类CDR3序列
# 输入：特征和标签，字典
# 输出：模型，Estimator
def cnn_model_CDR3_LL14(features, labels, mode):
    """Model function for CNN."""
    # Input Layer
    data = features["x"]
    # LL=features['LL']
    input_layer = tf.reshape(data, [-1, Nf, 14, 1])
    # Convolutional Layer #1
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=8,
        kernel_size=[Nf, 2],
        padding="valid",
        activation=tf.nn.relu,
    )
    # Pooling Layer #1
    pool1 = tf.layers.max_pooling2d(
        inputs=conv1, pool_size=[1, 2], strides=[1, 1]
    )  # stride used to be 2
    # Convolutional Layer #2 and Pooling Layer #2
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=16,
        kernel_size=[1, 2],
        padding="valid",
        activation=tf.nn.relu,
    )
    pool2 = tf.layers.max_pooling2d(
        inputs=conv2, pool_size=[1, 2], strides=[1, 1])
    SHAPE = pool2.shape
    pool2_flat = tf.reshape(pool2, [-1, int(SHAPE[1] * SHAPE[2] * SHAPE[3])])
    dense = tf.layers.dense(inputs=pool2_flat, units=10, activation=tf.nn.relu)
    dropout = tf.layers.dropout(
        inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN
    )
    # Logits Layer
    logits = tf.layers.dense(inputs=dropout, units=2)
    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=logits, axis=1),
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
        # `logging_hook`.
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor"),
    }
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
    # Calculate Loss (for both TRAIN and EVAL modes)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss=loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)
    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predictions["classes"]
        )
    }
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops
    )


# 基于卷积神经网络（CNN）的模型，用于识别和分类CDR3序列
# 输入：特征和标签，字典
# 输出：模型，Estimator
def cnn_model_CDR3_LL15(features, labels, mode):
    """Model function for CNN."""
    # Input Layer
    data = features["x"]
    # LL=features['LL']
    input_layer = tf.reshape(data, [-1, Nf, 15, 1])
    # Convolutional Layer #1
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=8,
        kernel_size=[Nf, 2],
        padding="valid",
        activation=tf.nn.relu,
    )
    # Pooling Layer #1
    pool1 = tf.layers.max_pooling2d(
        inputs=conv1, pool_size=[1, 2], strides=[1, 1]
    )  # stride used to be 2
    # Convolutional Layer #2 and Pooling Layer #2
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=16,
        kernel_size=[1, 2],
        padding="valid",
        activation=tf.nn.relu,
    )
    pool2 = tf.layers.max_pooling2d(
        inputs=conv2, pool_size=[1, 2], strides=[1, 1])
    SHAPE = pool2.shape
    pool2_flat = tf.reshape(pool2, [-1, int(SHAPE[1] * SHAPE[2] * SHAPE[3])])
    dense = tf.layers.dense(inputs=pool2_flat, units=10, activation=tf.nn.relu)
    dropout = tf.layers.dropout(
        inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN
    )
    # Logits Layer
    logits = tf.layers.dense(inputs=dropout, units=2)
    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=logits, axis=1),
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
        # `logging_hook`.
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor"),
    }
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
    # Calculate Loss (for both TRAIN and EVAL modes)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss=loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)
    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predictions["classes"]
        )
    }
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops
    )


# 基于卷积神经网络（CNN）的模型，用于识别和分类CDR3序列
# 输入：特征和标签，字典
# 输出：模型，Estimator
def cnn_model_CDR3_LL16(features, labels, mode):
    """Model function for CNN."""
    # Input Layer
    data = features["x"]
    # LL=features['LL']
    input_layer = tf.reshape(data, [-1, Nf, 16, 1])
    # Convolutional Layer #1
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=8,
        kernel_size=[Nf, 2],
        padding="valid",
        activation=tf.nn.relu,
    )
    # Pooling Layer #1
    pool1 = tf.layers.max_pooling2d(
        inputs=conv1, pool_size=[1, 2], strides=[1, 1]
    )  # stride used to be 2
    # Convolutional Layer #2 and Pooling Layer #2
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=16,
        kernel_size=[1, 2],
        padding="valid",
        activation=tf.nn.relu,
    )
    pool2 = tf.layers.max_pooling2d(
        inputs=conv2, pool_size=[1, 2], strides=[1, 1])
    SHAPE = pool2.shape
    pool2_flat = tf.reshape(pool2, [-1, int(SHAPE[1] * SHAPE[2] * SHAPE[3])])
    dense = tf.layers.dense(inputs=pool2_flat, units=10, activation=tf.nn.relu)
    dropout = tf.layers.dropout(
        inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN
    )
    # Logits Layer
    logits = tf.layers.dense(inputs=dropout, units=2)
    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=logits, axis=1),
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
        # `logging_hook`.
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor"),
    }
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
    # Calculate Loss (for both TRAIN and EVAL modes)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss=loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)
    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predictions["classes"]
        )
    }
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops
    )


# 根据输入的整数 version 选择合适的模型，实现模型版本的切换
ModelDict = {
    12: cnn_model_CDR3_LL12,
    13: cnn_model_CDR3_LL13,
    14: cnn_model_CDR3_LL14,
    15: cnn_model_CDR3_LL15,
    16: cnn_model_CDR3_LL16,
}


# 训练一个CNN模型并对输入的特征数据进行评估
# 输入：训练特征数据，字典、训练标签数据，字典、评估特征数据，字典、评估标签数据，字典、训练步数，整数、模型版本，标识符、模型保存路径
# 输出：评估结果
# 调用于batchTrain
def TrainAndEvaluate(
    TrainFeature,
    TrainLabels,
    EvalFeature,
    EvalLabels,
    STEPs=10000,
    ID="",
    dir_prefix="/tmp/",
):
    # Train CNN model:
    for LL in range(12, 17):
        # 创建一个CNN模型
        CDR3_classifier = tf.estimator.Estimator(
            model_fn=ModelDict[LL],
            model_dir=dir_prefix
            + "/CDR3_classifier_PCA_LL"
            + str(LL)
            + "_L2_k2f8d10_"
            + ID
            + "/",
        )
        # 从训练特征数据中生成批量训练数据
        train_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": TrainFeature[LL]["x"]},
            y=TrainLabels[LL],
            batch_size=100,
            num_epochs=None,
            shuffle=True,
        )
        # 训练模型
        CDR3_classifier.train(input_fn=train_input_fn, steps=STEPs)
        # 评估模型
        eval_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": EvalFeature[LL]["x"]}, y=EvalLabels[LL], num_epochs=1, shuffle=False
        )
        eval_results = CDR3_classifier.evaluate(input_fn=eval_input_fn)
        # 输出评估结果
        with open(dir_prefix + "/eval_results.txt", "a") as f:
            f.write("CDR3_classifier_PCA_LL{}_L2_k2f8d10_{}:".format(
                LL, ID)+str(eval_results)+"\n")
        print(eval_results)


# 预测和评估CDR3序列的标签
# 输入：评估特征数据，字典、评估标签数据，字典、ROC曲线绘制，bool、预测评估标识符、模型保存路径
# 输出：预测结果、标签、AUC值
# 调用于batchTrain
def PredictEvaluation(
    EvalFeature, EvalLabels=None, makePlot=False, ID="", dir_prefix=curPath + "/tmp/"
):
    PredictClass = {}
    PredictLabels = {}
    AUCDict = {}
    for LL in range(12, 17):
        # 创建一个预测评估对象
        CDR3_classifier = tf.estimator.Estimator(
            model_fn=ModelDict[LL],
            model_dir=dir_prefix
            + "/CDR3_classifier_PCA_LL"
            + str(LL)
            + "_L2_k2f8d10_"
            + ID
            + "/",
        )
        if EvalLabels is None:
            # 从EvalFeature中获取特征
            eval_input_fn = tf.estimator.inputs.numpy_input_fn(
                x={"x": EvalFeature[LL]["x"]}, num_epochs=1, shuffle=False
            )
            eval_results = CDR3_classifier.predict(input_fn=eval_input_fn)
            xx = []
            for x in eval_results:
                xx.append(x["probabilities"][1])
            AUC = None
            YY = None
        else:
            # 从EvalFeature和EvalLabels中获取特征和标签
            eval_input_fn = tf.estimator.inputs.numpy_input_fn(
                x={"x": EvalFeature[LL]["x"]},
                y=EvalLabels[LL],
                num_epochs=1,
                shuffle=False,
            )
            # 预测
            eval_results = CDR3_classifier.predict(input_fn=eval_input_fn)
            xx = []
            for x in eval_results:
                xx.append(x["probabilities"][1])
            YY = EvalLabels[LL]
            xy = list(zip(xx, YY))
            xy.sort()
            xs = [x for x, y in xy]
            ys = [y for x, y in xy]
            AUC = roc_auc_score(ys, xs)
        # 将预测结果和标签保存
        PredictClass[LL] = xx
        AUCDict[LL] = AUC
        PredictLabels[LL] = YY
    if makePlot:
        # 绘制ROC曲线
        LLcolors = ["b", "g", "r", "c", "m"]
        LegendLabels = []
        plt.figure(figsize=(7, 7))
        font = {"family": "Arial", "weight": "normal", "size": 22}
        mpl.rc("font", **font)
        hhList = []
        for LL in range(12, 17):
            xx = PredictClass[LL]
            yy = PredictLabels[LL]
            ycurve = roc_curve(yy, xx)
            (hh,) = plt.plot(ycurve[0], ycurve[1], LLcolors[LL - 12], lw=2)
            hhList.append(hh)
            LegendLabels.append(
                str(LL) + " (" + str(np.round(AUCDict[LL], 2)) + ")")
        plt.plot([0, 1], [0, 1], ls="dashed", lw=2)
        plt.xlabel("False Positive Rate", fontsize=22)
        plt.ylabel("True Positive Rate", fontsize=22)
        legend = plt.legend(hhList, LegendLabels,
                            fontsize=22, title="Length (AUC)")
        # plt.show()
        plt.savefig(dir_prefix + "/ROC-" + ID + ".png", dpi=300)
    return PredictClass, PredictLabels, AUCDict


# 批量训练和评估CDR3序列
# 输入：肿瘤样本的CDR3序列的文件路径、正常样本的CDR3序列的文件路径、评估肿瘤样本的CDR3序列的文件路径、评估正常样本的CDR3序列的文件路径、交叉验证的比例即训练集占整个数据集的比例、随机抽取的子样本的数量、训练过程中的迭代次数、临时文件的存储目录前缀
# 输出：预测类别、预测标签、AUC字典
def batchTrain(
    ftumor,
    fnormal,
    feval_tumor,
    feval_normal,
    rate=0.33,
    n=100,
    STEPs=10000,
    dir_prefix=curPath + "/tmp",
):
    # rate: cross validation ratio: 0.2 means 80% samples will be used for training
    # n: number of subsamplings
    # 创建存储临时文件的目录
    pathlib.Path(dir_prefix).mkdir(parents=True, exist_ok=True)
    tumorCDR3s = []
    # 读取肿瘤和正常样本的CDR3序列文件，并将其存储在列表中
    g = open(ftumor)
    ftumor_skipping = 0
    for ll in g.readlines():
        rr = ll.strip()
        if (not rr.startswith("C")) or (not rr.endswith("F")) or (len(pat.findall(rr)) > 0):
            # print("Non-standard CDR3s. Skipping.")
            ftumor_skipping += 1
            continue
        tumorCDR3s.append(rr)
    if ftumor_skipping > 0:
        print("Non-standard CDR3s. Skipping %d samples." % (ftumor_skipping))
    print("Number of tumor samples: %d" % (len(tumorCDR3s)))
    g.close()
    normalCDR3s = []
    g = open(fnormal)
    fnormal_skipping = 0
    for ll in g.readlines():
        rr = ll.strip()
        if (not rr.startswith("C")) or (not rr.endswith("F")) or (len(pat.findall(rr)) > 0):
            # print("Non-standard CDR3s. Skipping.")
            fnormal_skipping += 1
            continue
        normalCDR3s.append(rr)
    if fnormal_skipping > 0:
        print("Non-standard CDR3s. Skipping %d samples." % (fnormal_skipping))
    print("Number of normal samples: %d" % (len(normalCDR3s)))
    g.close()
    # 初始化
    count = 0
    nt = len(tumorCDR3s)
    nn = len(normalCDR3s)
    vt_idx = range(0, nt)
    vn_idx = range(0, nn)
    nt_s = int(np.ceil(nt * (1 - rate)))
    nn_s = int(np.ceil(nn * (1 - rate)))
    PredictClassList = []
    PredictLabelList = []
    AUCDictList = []
    while count < n:
        print("==============Training cycle %d.=============" % (count))
        ID = str(count)
        # 随机抽取肿瘤和正常样本的训练集和测试集
        vt_train = np.random.choice(vt_idx, nt_s, replace=False)
        vt_test = [x for x in vt_idx if x not in vt_train]
        vn_train = np.random.choice(vn_idx, nn_s, replace=False)
        vn_test = [x for x in vn_idx if x not in vn_train]
        # 将训练集和测试集分别写入文件
        sTumorTrain = np.array(tumorCDR3s)[vt_train]
        sNormalTrain = np.array(normalCDR3s)[vn_train]
        sTumorTest = np.array(tumorCDR3s)[vt_test]
        sNormalTest = np.array(normalCDR3s)[vn_test]
        ftrain_tumor = dir_prefix + "/sTumorTrain-" + str(ID) + ".txt"
        ftrain_normal = dir_prefix + "/sNormalTrain-" + str(ID) + ".txt"
        feval_tumor = dir_prefix + "/sTumorTest-" + str(ID) + ".txt"
        feval_normal = dir_prefix + "/sNormalTest-" + str(ID) + ".txt"
        h = open(ftrain_tumor, "w")
        _ = [h.write(x + "\n") for x in sTumorTrain]
        h.close()
        h = open(ftrain_normal, "w")
        _ = [h.write(x + "\n") for x in sNormalTrain]
        h.close()
        h = open(feval_tumor, "w")
        _ = [h.write(x + "\n") for x in sTumorTest]
        h.close()
        h = open(feval_normal, "w")
        _ = [h.write(x + "\n") for x in sNormalTest]
        h.close()
        # 读取训练集和测试集
        g = open(ftrain_tumor)
        Train_Tumor = []
        for line in g.readlines():
            Train_Tumor.append(line.strip())
        Train_Tumor = np.array(Train_Tumor)
        g = open(ftrain_normal)
        Train_Normal = []
        for line in g.readlines():
            Train_Normal.append(line.strip())
        Train_Normal = np.array(Train_Normal)
        TrainFeature, TrainLabels = GetFeatureLabels(Train_Tumor, Train_Normal)
        g = open(feval_tumor)
        Eval_Tumor = []
        for line in g.readlines():
            Eval_Tumor.append(line.strip())
        Eval_Tumor = np.array(Eval_Tumor)
        g = open(feval_normal)
        Eval_Normal = []
        for line in g.readlines():
            Eval_Normal.append(line.strip())
        Eval_Normal = np.array(Eval_Normal)
        EvalFeature, EvalLabels = GetFeatureLabels(Eval_Tumor, Eval_Normal)
        # 训练和评估
        TrainAndEvaluate(
            TrainFeature,
            TrainLabels,
            EvalFeature,
            EvalLabels,
            STEPs=STEPs,
            ID=ID,
            dir_prefix=dir_prefix,
        )
        # 预测评估
        PC, PL, AD = PredictEvaluation(
            EvalFeature,
            EvalLabels=EvalLabels,
            makePlot=True,
            ID=ID,
            dir_prefix=dir_prefix,
        )
        PredictClassList.append(PC)
        PredictLabelList.append(PL)
        AUCDictList.append(AD)
        count += 1
    return PredictClassList, PredictLabelList, AUCDictList


# 预测输入的iSMART结果文件中是否存在癌症
# 输入：iSMART结果文件、模型文件所在的目录
# 输出：单长度平均预测分数，列表、全长度预测结果，列表、全长预测分数、癌症预测分数
# 调用于PredictBatch
def PredictCancer(f, dir_prefix):
    # f: input iSMART result file
    # N: top N most frequent CDR3s will be included in the analysis
    gf = open(f)
    # 存储CDR3序列
    CDR3s = []
    for ll in gf.readlines():
        cc = ll.strip().split("\t")[0]
        if not cc.startswith("C") or not cc.endswith("F"):
            continue
        CDR3s.append(cc)
    # AAindex编码
    CDR3sDict = {}
    for cc in CDR3s:
        if len(pat.findall(cc)) > 0:
            continue
        ll = len(cc)
        ccF = AAindexEncoding(cc)
        if ll not in CDR3sDict:
            CDR3sDict[ll] = [ccF]
        else:
            CDR3sDict[ll].append(ccF)
    # 存储所有长度预测的癌症预测分数
    ScoreDict = {}
    # 所有长度的预测结果
    XX = []
    for LL in range(12, 17):
        CDR3_classifier = tf.estimator.Estimator(
            model_fn=ModelDict[LL],
            model_dir=dir_prefix
            + "/CDR3_classifier_PCA_LL"
            + str(LL)
            + "_L2_k2f8d10_tCi01"
            + "/",
        )
        if LL in CDR3sDict:
            eval_input_fn = tf.estimator.inputs.numpy_input_fn(
                x={"x": np.array(CDR3sDict[LL])}, num_epochs=1, shuffle=False
            )
        else:
            continue
        eval_results = CDR3_classifier.predict(input_fn=eval_input_fn)
        xx = []
        for x in eval_results:
            xx.append(x["probabilities"][1])
        ScoreDict[LL] = xx
        XX += xx
    # 每个长度计算平均预测分数
    mms = []
    for kk in ScoreDict:
        mms.append((kk, np.mean(ScoreDict[kk])))
    # 所有预测结果的平均值，作为癌症预测分数
    CancerScore = np.mean(XX)
    # return CancerScore, XX
    return mms, XX, ScoreDict, CancerScore


# 批量预测癌症
# 输入：要预测的文件夹路径、临时文件夹的路径
# 输出：预测结果的文件名列表、癌症预测分数列表、癌症预测分数标准差列表、预测结果的分布列表
def PredictBatch(DIR, dir_prefix=curPath + "/tmp/"):
    ffs = os.listdir(DIR)
    mmsList = []
    SDList = []
    XXList = []
    for ff in ffs:
        mms, XX, SD = PredictCancer(DIR + "/" + ff, dir_prefix=dir_prefix)
        mmsList.append(mms)
        SDList.append(SD)
        XXList.append(XX)
    return ffs, mmsList, SDList, XXList


# DeepCAT启动部分，预测癌症
if __name__ == "__main__":
    if len(sys.argv) > 1:  # 检查命令行参数的数量是否大于1
        # argv[0]为DeepCAT.py，argv[1]为输入文件夹路径，argv[2]为模型文件夹的路径，argv[3]为-t/-r
        DIR = sys.argv[1]
        DIR1 = os.path.basename(DIR)
        ffs = os.listdir(DIR)
        dir_prefix = sys.argv[2]
        CC = []
        ffss = []
        for ff in ffs:
            if ff == "README.md":
                continue
            else:
                _, XX1, _, score = PredictCancer(
                    DIR + "/" + ff, dir_prefix
                )  # return mms, XX, ScoreDict, CancerScore
                CC.append(score)
                ffss.append(ff)
        CC = np.array(CC)
        ffss = np.array(ffss)
        # 检查命令行参数3的值，如果为'-t'，则将结果保存到名为'Cancer_score_'+DIR1+'.txt'的文本文件中，使用制表符分隔。
        # 如果为'-r'，则将结果保存到名为'Cancer_score.txt'的文本文件中，使用制表符分隔
        if sys.argv[3] == "-t":
            with open("Cancer_score_" + DIR1 + ".txt", "w") as f:
                writer = csv.writer(f, delimiter="\t")
                writer.writerows(zip(ffss, CC))
        elif sys.argv[3] == "-r":
            with open("Cancer_score.txt", "w") as f:
                writer = csv.writer(f, delimiter="\t")
                writer.writerows(zip(ffss, CC))
