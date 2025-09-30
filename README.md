# PassBert_torch
基于torch实现的PassBert口令预训练

paper:https://www.usenix.org/conference/usenixsecurity23/presentation/xu-ming

用KIMI2的Ok Computer功能进行了简单的论文解读，网址如下：https://3nywnaj62ipm4.ok.kimi.link/

环境：
python:3.11.13

torch:2.8.0+cu128

正在全力编写中~

更新日志：

9.28：完成了passbert中的PasswordTokenize(测试无问题)

9.29:完成了Passbert模型部分的初步搭建

9.30:完成了pretrain部分的数据集构建，暂时为测试

参考仓库:

[Bert4pytorch](https://github.com/MuQiuJun-AI/bert4pytorch/)

[Bert4keras](https://github.com/bojone/bert4keras/)

[PassBertStengthMeter-1](https://github.com/Ming-Xu-research/PassBertStrengthMeter-1) (PassBert的官方实现)