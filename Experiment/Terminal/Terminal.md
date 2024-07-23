###### 导出文件夹中的文件和目录结构：
1. 打开终端。
2. 导航到你想要的列出目录结构的文件夹：
```bash
cd path/to/your/folder
```
3. 使用 `tree` 命令生成目录结构（如果没有安装 `tree`，你可以通过 `brew install tree` 或 `sudo apt-get install tree` 安装）：
```bash
tree -a > directory_structure.txt
```
这将生成包含文件和目录结构的 `directory_structure.txt` 文件。