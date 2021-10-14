环境更改
* env：最初环境（包括discrete，continue，continue_cnn三个版本）
* env_sp：在continue条件上加入夏普比率
* env_sp1：加入手续费计算
* env_sp2: 加入单日的交易限制
* zzenv：改为zz500的环境

执行文件更改
* version 3.1：最初始版本（用于选择模型）
* version 3.2：适用于加入金融指标/手续费/单日交易限制

当前版本：zz_ppo.py, zz_ddpg.py, zz_td3.py

* visualize：可视化执行文件
* normalize：数据标准化
