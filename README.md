# yjs_algorithm
yjs project about algorithm

# 配置
1. 项目目录不完整，还需要下载datas文件夹和完成render项目的配置，最终的完整后端目录如下,他们都是并列的目录或文件
```
.idea
__pycache__
datas （需要下载）
redner （需要自己配）
saved_data
apis.py
demo.py
model.py
```
2. datas文件夹我已经压缩成zip了，下面说说redner项目的配置
- 在此项目的根目录下，打开git bash，然后运行下面命令(最好翻墙)
```
git clone https://github.com/BachiLi/redner.git
cd redner
git submodule update --init --recursive
```
- 然后在有pytorch的python环境中安装render包，如下所示
```
pip install redner
```
- 然后在render的根目录下（注意不是本项目的根目录）运行python文件:
```
python setup.py install
```
- 这样这个redner项目就配好了，很简单吧，然后回到本云计算项目的根目录

# 运行
- 直接运行这个apis.py文件就行了，里面有这个confs字典变量，看看怎么和后端对接上
- apis.py的主函数很简单，先根据confs初始化模型，然后训练，在训练过程中把图片和obj文件给自动保存到confs里指定面的文件夹了
