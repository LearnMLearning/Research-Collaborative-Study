#### Step 1: 把人家代码搞下来 并导入 (这边示例为Pycharm)
![[Pasted image 20240723131700.png]]

![[Pasted image 20240723131709.png]]

![[Pasted image 20240723131800.png]]

![[Pasted image 20240723131806.png]]

![[Pasted image 20240723131813.png]]

#### Step 2: 环境配置
###### 1. 安装 Git:
如果你还没有安装 Git，可以通过以下方式安装：
- **Windows**：下载并安装 [Git for Windows](https://gitforwindows.org/)
- **macOS**：使用 Homebrew 安装 `brew install git`
- **Linux**：使用包管理器安装，例如 `sudo apt-get install git`

###### 2. 安装 Python 和 pip：
- 下载并安装 [Python](https://www.python.org/downloads/)，确保安装过程中勾选 "Add Python to PATH" 选项。
- Python 安装后会自带 `pip`，你可以通过 `pip --version` 检查是否成功安装。

###### 3. 安装 virtualenv：
```bash
pip install virtualenv
```

###### 4. 创建虚拟环境
```bash
virtualenv venv
# 激活虚拟环境
# Windows
.\venv\Scripts\activate
# macOS/Linux
source venv/bin/active
```
![[Pasted image 20240723131827.png]]

#### Step 3: 克隆目标仓库
``` bash
git clone https://github.com/jjbrophy47/dare_rf git clone https://github.com/schelterlabs/hedgecut
```

#### Step 4: 安装依赖
```bash
pip install -r requirements.txt
```

#### Step 5: 运行代码并复现结果

1. **查找主要脚本**： 通常 README 文件中会提到如何运行主要脚本。例如：
```bash
python main.py
```
2. **运行主要脚本**： 根据项目的具体要求运行脚本。如果项目需要配置参数或环境变量，按照说明进行设置。
3. **生成可视化结果**： 查找生成图表和可视化结果的代码部分，通常在 `main.py` 或 `analysis.py` 中。运行这些脚本以生成论文中的图表。

![[Pasted image 20240723131911.png]]

#### Step 5: 分析和调试
1. **分析代码**：
	- 使用注释和文档来理解代码的每一步。注释可以帮助你更好地理解代码的功能。
	- 如果遇到问题，可以使用 `print` 或调试工具来检查变量和输出。
2. **调试和修复问题：**
	- 如果代码无法运行或生成的结果不正确，检查错误消息并尝试解决问题。
	- 在 GitHub Issues 页面查找是否有类似的问题和解决方案。

#### Step 7: 记录和总结
1. **记录过程**：
	- 记录你在运行和调试过程中遇到的问题和解决方案。
	- 写下每一步的详细操作，以便将来参考
2. **总结和改进**：
	- 总结你在整个过程中学到的知识和技能。
	- 思考是否有改进和优化代码的空间。
	- 
![[Pasted image 20240723131919.png]]

![[Pasted image 20240723131930.png]]
