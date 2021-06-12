# 工科创III-E蛋白质关联图预测

## 使用方法
下面是使用案例，请替换其中的参数。
### 训练
``` sh
python3 train.py --model attention --train_dir ~/Data/train --valid_dir ~/Data/valid
```

### 测试
``` sh
python3 test.py --model attention --checkpoint run/experiment_24/best_model.pth.tar --test_dir ~/Data/test
```

### 预测
``` sh
python3 predict.py --model attention --checkpoint run/experiment_24/best_model.pth.tar --test_dir ~/test_data --target_dir predict
```

预测时，输入的文件为npz类型，保存在`[test_dir]/feature`文件夹内。输出的蛋白质预测文件位于`predict`文件夹内。