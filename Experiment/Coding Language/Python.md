https://www.learnpython.org

以 DaRE 为例

###### `__init__.py`
dare 是一个module，被导入时 [[__init__.py]] 优先被执行

![[Pasted image 20240723161604.png]]
如果不是这样，得写
```python
from dare._classes import Tree,Forest
```

这个文件里有一个特殊变量
```python
__all__ = {
	'Tree',
	'Forest'
}
```


###### `__classes__.py`
```python
class Forest(object):
```

**`__init__`**
![[Pasted image 20240723162338.png]]
```python
a = Forest (k=a,b,c)
```

**`__str__`**
![[Pasted image 20240723162645.png]]

![[Pasted image 20240723162707.png]]