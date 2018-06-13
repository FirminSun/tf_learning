# tf_learning
tensorflow learining and application



## 第四章
### 交叉熵

### 防止过拟合
* 增加数据集
* 正则化方法
* Dropout

### 优化器
* tensorflow中的优化器
    * tf.train.GradientDescentOptimizer
    * tf.train.AdadeltaOptimizer
    * tf.train.AdagradOptimizer
    * tf.train.MomentumOptimizer
    * tf.train.AdamOptimizer
    * tf.train.FtrlOptimizer
    * tr.train.RMSPropOptimizer
    
GradientDescentOptimizer 使用广泛收敛慢，稳定性好
AdadeltaOptimizer和RMSPropOptimizer  收敛速度快,稳定
MomentumOptimizer 速度最快，不过初始值不好时会向错误的方向搜索