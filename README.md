# PassBert_torch
基于torch实现的PassBert口令预训练

paper:https://www.usenix.org/conference/usenixsecurity23/presentation/xu-ming

用KIMI2的Ok Computer功能进行了简单的论文解读，网址如下：https://3nywnaj62ipm4.ok.kimi.link/

### 环境：

python:3.11.13

torch:2.8.0+cu128

正在全力编写中~

### 更新日志：

9.28：完成了passbert中的PasswordTokenize(测试无问题)

9.29:完成了Passbert模型部分的初步搭建

9.30:完成了pretrain部分的数据集构建，暂时为测试

10.3:完成了pretrain部分的数据集构造的测试，目前process_data可以处理出满足需求的token_id以及mask序列，可以使用根目录下的test_passwords.txt进行测试

10.4:目前已经完成了所有训练流程的迁移，可以正常进行训练，但是可能存在一些Bug，需要进行一系列调试

10.8:修复了一些小Bug，增加了Type_Embedding,Span Mask掩码,以及FlashAttention

### Todo List

- [x] attention改为flashatten

- [ ] 写一个训练效果优劣的metric

- [ ] 将Span Mask结合WWM做针对口令问题的调整，加入PCFG，例如Password1！,那么其PCFG格式为L8D1S1，在这个层面我们使用WWM级别的掩码方式，假设我们选择掩饰L8那么在L8中我们使用SM的方式进行掩码

- [ ] KL,SinglePassword,PassSim,理论最优

参考仓库:

[Bert4pytorch](https://github.com/MuQiuJun-AI/bert4pytorch/)

[Bert4keras](https://github.com/bojone/bert4keras/)

[PassBertStengthMeter-1](https://github.com/Ming-Xu-research/PassBertStrengthMeter-1) (PassBert的官方实现)
