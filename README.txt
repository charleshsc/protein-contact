数据读取参考dataset.py
评估你的测试集参考eval.py

1. 给你的label是L*L基于真实结构生成的实数，所以在dataset.py进行了划分
2. 建议在计算loss图之后loss*=mask再开始backward
3. 有问题随时可提问or私聊