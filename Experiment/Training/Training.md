模型训练 (以DeepCrack为例)
###### model
模型结构定义
![[Pasted image 20240627160408.png|500]]
`train.py` 训练主代码
`trainer.py` 函数
![[Pasted image 20240627160655.png|500]]
`config.py` 超参 索引
![[Pasted image 20240627161400.png|500]]
`infer.py` 推断
![[Pasted image 20240627162256.png|500]]
1440 * 1440 实际，最好裁成这样

改服务器对应关系
```
screen -R train_3
```

```
export CUDA_VISIBLE_DEVICES=3
```

###### 开始训练
```
python train.py
```


[[Dataset]]
