# 蛋白质关联图预测

该仓库主要包含对蛋白质关联图的预测，并且实现了多种模型。



----

## Table of Contents

- [Environment](#Environment)
- [Usage](#Usage)
- [Citations](#Citations)

-----



## Environment

CUDA version 10.1

> pytorch >= 1.7.1, torchvision==0.8.2, torchaudio==0.7.2, cudatoolkit=10.1



## Usage

我们一共实现了如下模型：

+ respre
+ dilation
+ deep
+ attention
+ lstm
+ realvalue
+ GAN



下面给出了运行程序的样例代码，若想运行不同的模型，只需更改对应的model参数即可。

**注意**  GAN模型需要单独进入到GAN文件夹下，之后再运行下述代码，并且model参数只可选择GAN 

### Train
``` sh
python3 train.py --model attention --train_dir path_to_train --valid_dir path_to_valid
```

### Test
``` sh
python3 test.py --model attention --checkpoint attention.pth.tar --test_dir path_to_test
```

### predict
``` sh
python3 predict.py --model attention --checkpoint attention.pth.tar --test_dir path_to_test --target_dir path_to_predict
```

预测时，输入的文件为npz类型，保存在`[test_dir]/feature`文件夹内。输出的蛋白质预测文件位于`predict`文件夹内。

### Ensemble
如果需要集成多个模型输出的结果，请先运行predict。之后使用`utils/ensemble.py`，按照提示配置其中的输入参数至指定文件夹，运行后即可得到集成结果。

### 附件
附件中包含了Unbalanced模型与Attention模型对应的参数。在测试Unbalanced模型参数时，请将`model`参数修改为`dilation`。在数据集目录下，应分别包含`feature`和`label`两个子目录。

## Citations

```bibtex
@misc{hu2021TNTimple,
	auther = 		{Zhe Xie, Shuo Yu, Shengchao Hu, Chang Liu},
	title = 		{蛋白质关联图预测},
	howpublished = 	        {\url{https://gitee.com/xz2000/protein}},
	year = 			{2021}
}
```

