# Python：知识库资料

> 目标读者：掌握计算机基础知识的本科生。资料以 Python 3.10 及以上版本的常用行为为基线；涉及解释器差异、操作系统差异或性能结论时会单独说明前提。示例默认使用 CPython，代码中的文件路径可按实际环境调整。

## 目录

1. 变量、对象与常用数据类型
2. 可变对象、不可变对象与复制
3. 条件判断与循环
4. 函数、参数与作用域
5. 列表与元组
6. 字典与集合
7. 类、对象与封装
8. 继承、多态与组合
9. 迭代器与生成器
10. 装饰器
11. 异常处理
12. 文件读写与 JSON
13. 模块、包与虚拟环境
14. 进程、线程、协程与 GIL
15. 常见错误与调试方法

## 1. 变量、对象与常用数据类型

### 1.1 概念

Python 的变量是绑定到对象的名字，不是存放固定类型数据的内存格子。执行 `count = 3` 后，名字 `count` 引用一个整数对象；再执行 `count = "三"`，只是让同一个名字改为引用字符串对象。对象本身具有类型和值，这体现了 Python 的动态类型特征。

常用内置类型包括：

- 数值：`int` 表示任意精度整数，`float` 通常表示双精度浮点数，`complex` 表示复数，`bool` 表示布尔值。
- 文本与字节：`str` 表示 Unicode 文本，`bytes` 表示不可变字节序列。网络传输和二进制文件通常使用字节，业务文本通常使用字符串。
- 序列与容器：`list`、`tuple`、`dict`、`set` 分别用于动态列表、不可变序列、键值映射和不重复元素集合。
- 空值：`None` 是 `NoneType` 类型的唯一实例，常表示“没有值”或“尚未提供值”。

### 1.2 原理或使用方法

赋值默认复制的是对象引用，而不是完整对象。`type(obj)` 返回对象的实际类型；`isinstance(obj, SomeType)` 还会接受子类实例，因此在面向对象代码中通常更适合做类型检查。

浮点数采用二进制近似表示，不能精确表示所有十进制小数。金额等要求十进制精确计算的场景可以考虑 `decimal.Decimal`，并用字符串构造初始值。Decimal 仍有精度和舍入规则，并不意味着任意运算都无限精确。

### 1.3 示例

```python
from decimal import Decimal

count = 7
name = "Python"
tags = ["编程", "动态类型"]
point = (3, 4)
student = {"name": "小林", "score": 90}
unique_numbers = {1, 2, 2}

print(type(count).__name__)       # int
print(isinstance(True, int))      # True：bool 是 int 的子类
print(unique_numbers)             # {1, 2}，集合自动去除重复元素
print(0.1 + 0.2 == 0.3)           # False，二进制浮点误差
print(Decimal("0.1") + Decimal("0.2") == Decimal("0.3"))  # True
```

### 1.4 易错点和注意事项

- `==` 比较值是否相等，`is` 比较两个引用是否指向同一对象。判断 `None` 推荐写 `value is None`，一般不要用 `is` 比较数字或字符串。
- 不要把 `bool` 与 `int` 的子类关系误当成“布尔值适合参与所有整数运算”。做数值参数校验时，如有需要应单独排除布尔值。
- `str` 与 `bytes` 不能直接拼接或混用文本方法，应通过 `encode()` 和 `decode()` 显式转换，并选择正确编码。
- 类型转换可能失败，例如 `int("3.5")` 会抛出 `ValueError`，应先校验或捕获异常。

### 1.5 常见问题

**问：Python 变量需要先声明类型吗？**  
答：不需要。变量在使用时通过赋值绑定对象，也可以添加类型注解帮助阅读与静态检查，但普通类型注解不会自动执行运行时类型校验。

**问：判断对象类型应该使用 `type()` 还是 `isinstance()`？**  
答：要求“必须是这个精确类型”时可用 `type(x) is T`；需要兼容子类时使用 `isinstance(x, T)`。很多场景还可以依靠对象提供的接口，而不必过度检查类型。

**问：为什么 `0.1 + 0.2` 不严格等于 `0.3`？**  
答：这是二进制浮点表示造成的舍入误差，不是 Python 独有的语法错误。显示时控制小数位、比较时使用合理容差，或按业务需要选择 Decimal。

## 2. 可变对象、不可变对象与复制

### 2.1 概念

可变对象允许在对象身份不变的情况下修改内容，例如 `list`、`dict`、`set`；不可变对象创建后不能改变自身内容，例如 `int`、`str`、`tuple`、`frozenset`。对不可变对象执行“修改”操作，通常是创建新对象并让变量重新绑定。

可变性与可哈希性相关但不完全等价。作为字典键或集合元素的对象必须可哈希，并且其参与相等性比较的内容在作为键使用期间不应变化。元组虽然不可变，但若内部含有列表，整个元组仍不可哈希。

### 2.2 原理或使用方法

`b = a` 只增加一个引用。浅拷贝创建新的外层容器，但内层对象仍共享；`copy.deepcopy()` 递归复制内部对象，通常使嵌套结构彼此独立，但成本更高，而且文件句柄等资源不能被随意深拷贝。

函数参数也采用共享对象的方式传递：函数内通过形参修改可变对象会影响调用方；给形参重新赋值则只改变局部绑定。默认参数在函数定义时求值，因此不应使用共享的可变对象保存每次调用独立的状态。

### 2.3 示例

```python
import copy

original = [[1, 2], [3, 4]]
alias = original
shallow = original.copy()
deep = copy.deepcopy(original)

alias.append([5, 6])
shallow[0].append(99)
deep[1].append(88)

print(original)  # [[1, 2, 99], [3, 4], [5, 6]]
print(shallow)   # [[1, 2, 99], [3, 4]]
print(deep)      # [[1, 2], [3, 4, 88]]


def add_item(item, bucket=None):
    if bucket is None:
        bucket = []
    bucket.append(item)
    return bucket


print(add_item("A"))  # ['A']
print(add_item("B"))  # ['B']，两次调用没有共享列表
```

### 2.4 易错点和注意事项

- `list.copy()`、`a[:]` 和 `copy.copy(a)` 都是浅拷贝，不会复制嵌套列表中的内层列表。
- 不可变容器不代表其中所有对象都不可变，例如 `([], [])` 是元组，但内部列表仍可修改。
- 不要在遍历同一个列表或集合时随意增删元素，这可能跳过元素或抛出运行时错误；可遍历副本或构建新容器。
- 深拷贝不是越多越好。结构很大或含有外部资源时，应明确哪些层级需要复制。

### 2.5 常见问题

**问：字符串真的是不可变对象吗？**  
答：是。`text += "!"` 会生成新字符串并重新绑定 `text`，不是修改原字符串的内部字符。

**问：为什么字典键不能使用列表？**  
答：列表内容可改变，若其哈希值随内容变化，字典将无法稳定定位原键。需要序列作为键时，可在元素都可哈希的前提下使用元组。

**问：函数默认参数写成 `items=[]` 有什么问题？**  
答：该列表只在函数定义时创建一次，多次调用会共享它。需要每次调用独立列表时，默认使用 `None`，再在函数体中创建列表。

## 3. 条件判断与循环

### 3.1 概念

条件判断根据表达式的布尔结果选择执行分支，使用 `if / elif / else`。循环用于重复处理数据：`for` 适合遍历可迭代对象，`while` 适合在条件成立期间持续执行。

Python 有空值语义：`None`、`False`、数值零、空字符串和空容器在布尔上下文中为假，大多数非空对象为真。显式比较通常比依赖隐式转换更清晰。

### 3.2 原理或使用方法

`elif` 分支按顺序检查，命中后不再检查后续分支，因此范围条件应从窄到宽或按明确优先级排列。`break` 结束当前循环，`continue` 跳过本次循环的剩余部分。循环后的 `else` 只在循环未被 `break` 中断时执行，常用于“查找失败”分支。

`range(start, stop, step)` 生成惰性整数序列，结束值不包含在结果中。需要索引和元素时可使用 `enumerate()`，并行遍历多个序列可使用 `zip()`；`zip()` 默认在最短输入结束时停止。

### 3.3 示例

```python
def level_of(score):
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        raise TypeError("score 必须是数值")
    if not 0 <= score <= 100:
        raise ValueError("score 必须在 0 到 100 之间")
    if score >= 90:
        return "优秀"
    if score >= 60:
        return "合格"
    return "待提高"


print(level_of(85))  # 合格


def find_even(numbers):
    for index, value in enumerate(numbers):
        if value % 2 == 0:
            return index, value
    else:
        return None


print(find_even([1, 3, 6, 8]))  # (2, 6)
print(find_even([1, 3, 5]))     # None
```

### 3.4 易错点和注意事项

- `if value is not None` 与 `if value` 不同，后者会把 `0`、空字符串等合法值也当成假。
- 浮点数直接判断相等可能受误差影响，应根据问题选择容差或十进制计算。
- `while` 循环必须存在可达成退出条件的路径，否则可能形成死循环。
- 不要在 `for item in items` 中直接删除 `items` 的当前元素；可以改为 `items = [x for x in items if ...]`。

### 3.5 常见问题

**问：`for` 循环后面的 `else` 是“没有进入循环就执行”吗？**  
答：不是。只要循环没有通过 `break` 退出，`else` 就会执行，包括正常遍历结束和一开始就没有元素的情况。

**问：什么时候选择 `while` 而不是 `for`？**  
答：已知要遍历的数据序列时优先 `for`；重复次数由动态状态决定、没有明确迭代对象时更适合 `while`，同时应设计退出条件。

**问：如何同时获取下标和元素？**  
答：使用 `enumerate(items, start=0)`。需要从一个开始编号时可传 `start=1`，不必手动维护计数器。

## 4. 函数、参数与作用域

### 4.1 概念

函数把输入、处理步骤和输出组织为可复用单元。`def` 定义函数，`return` 返回结果并结束本次调用；没有显式 `return` 或只写 `return` 时返回 `None`。

参数可分为仅位置参数、位置或关键字参数、默认参数、可变位置参数 `*args`、仅关键字参数和可变关键字参数 `**kwargs`。局部变量在函数调用期间创建，全局变量属于模块级作用域。

### 4.2 原理或使用方法

函数定义中的 `/` 表示其左侧参数只能用位置传递；`*` 表示其右侧参数只能用关键字传递。`*args` 把多余位置实参收集为元组，`**kwargs` 把多余关键字实参收集为字典。调用时使用 `*iterable` 和 `**mapping` 可以展开参数。

名称查找遵循局部、外层函数、全局、内置的 LEGB 顺序。函数内给变量赋值默认创建局部变量；`global` 和 `nonlocal` 可改变绑定位置，但应谨慎使用，优先通过参数和返回值传递状态。

### 4.3 示例

```python
def make_profile(name, /, role="student", *skills, active=True, **extra):
    """创建简单的用户资料字典。"""
    return {
        "name": name,
        "role": role,
        "skills": list(skills),
        "active": active,
        **extra,
    }


profile = make_profile(
    "小林",                 # name 只能按位置传递
    "admin",               # role
    "Python", "SQL",       # 收集到 skills
    active=False,           # active 只能用关键字传递
    level=2,                # 收集到 extra
)
print(profile)
# {'name': '小林', 'role': 'admin', 'skills': ['Python', 'SQL'],
#  'active': False, 'level': 2}


def append_log(message, logs=None):
    if logs is None:
        logs = []
    logs.append(message)
    return logs
```

### 4.4 易错点和注意事项

- 默认参数只在定义时求值一次，不仅列表，任何依赖调用时状态的默认值都应注意这一点。
- 函数参数过多会增加误用风险，可把相关配置封装为对象，或使用仅关键字参数提高可读性。
- 不要为简洁而写很长的 lambda 或嵌套表达式；复杂逻辑应使用普通函数并取有意义的名称。
- 类型注解如 `def add(a: int, b: int) -> int` 主要用于文档与静态检查，CPython 不会仅凭注解阻止传入其他类型。

### 4.5 常见问题

**问：Python 函数是传值还是传引用？**  
答：更准确地说，形参与实参共享同一个对象。函数内修改可变对象会影响外部；给形参重新赋值不会让外部变量改指新对象。

**问：一个函数能返回多个值吗？**  
答：可以写 `return x, y`，实际返回一个元组，调用方常通过 `x, y = func()` 解包。返回含义较多时，也可使用字典或数据类提高可读性。

**问：什么时候使用 `*args` 和 `**kwargs`？**  
答：适合编写包装器或确实需要可变数量参数的通用接口。普通业务函数若参数含义固定，显式列出参数更清楚。

## 5. 列表与元组

### 5.1 概念

列表 `list` 是有序、可变、允许重复元素的序列，适合需要增删改的数据集合。元组 `tuple` 是有序、不可变的序列，适合表示结构固定的一组值，例如坐标或函数的多值返回。

“不可变”指元组不能替换、添加或删除自身槽位中的引用；若元组元素是可变对象，该元素内部仍可能改变。

### 5.2 原理或使用方法

列表通常基于动态数组实现，按下标访问为 O(1)，末尾追加通常具有摊销 O(1) 成本；在中间插入或删除需要移动元素，通常为 O(n)。具体扩容策略属于解释器实现细节，不应依赖固定增长倍数。

切片 `items[start:stop:step]` 不包含 `stop`，并创建新的浅层序列。`list.sort()` 原地排序并返回 `None`；`sorted(iterable)` 返回新列表，可处理任意可迭代对象。列表推导式适合清晰地转换和筛选数据，复杂处理仍应使用普通循环。

### 5.3 示例

```python
scores = [72, 90, 85, 90]
scores.append(60)
first_three = scores[:3]
high_scores = [score for score in scores if score >= 85]

scores.sort(reverse=True)       # 修改原列表
original_order = sorted(scores) # 创建升序新列表

point = (3, 4)
x, y = point
single = (1,)                   # 单元素元组必须保留逗号

print(first_three)   # [72, 90, 85]
print(high_scores)   # [90, 85, 90]
print(scores)        # [90, 90, 85, 72, 60]
print(original_order) # [60, 72, 85, 90, 90]
print(x, y, single)  # 3 4 (1,)
```

### 5.4 易错点和注意事项

- `result = items.sort()` 会让 `result` 得到 `None`，应直接调用 `items.sort()` 或使用 `result = sorted(items)`。
- `a * 3` 对列表会重复其中的引用，例如 `rows = [[]] * 3` 的三个内层列表是同一对象；独立行可用 `[[] for _ in range(3)]`。
- 列表适合顺序存储，但用 `item in list` 做成员判断需要线性扫描；大量存在性查询可考虑集合。
- 元组不等于“只读列表”的唯一选择。需要命名字段时可以考虑 `collections.namedtuple`、`typing.NamedTuple` 或数据类。

### 5.5 常见问题

**问：列表和元组应该怎么选？**  
答：内容需要动态增删时使用列表；结构固定、希望避免被重新绑定元素时使用元组。选择应基于语义和需求，不是因为元组在任何场景都更快。

**问：列表切片会复制嵌套对象吗？**  
答：不会。切片只创建新的外层列表，内部元素仍引用原对象，属于浅拷贝。

**问：为什么 `(1)` 不是元组？**  
答：括号在这里只表示表达式分组，`type((1,))` 才是单元素元组，关键符号是逗号。

## 6. 字典与集合

### 6.1 概念

字典 `dict` 保存键到值的映射，集合 `set` 保存不重复且可哈希的元素。字典适合按唯一键快速查找记录，集合适合去重、成员判断以及交集、并集、差集运算。

从 Python 3.7 起，字典保留插入顺序是语言规范的一部分；集合不是有序序列，不应依赖其遍历顺序。保留插入顺序也不等于按键排序。

### 6.2 原理或使用方法

字典和集合通常基于哈希表，平均查找、插入和删除成本为 O(1)，但极端哈希冲突或实现调整会影响实际表现。键必须可哈希，并且相等键的哈希值必须相同。

`d[key]` 在键不存在时抛出 `KeyError`；`d.get(key, default)` 返回默认值；`setdefault()` 会在键缺失时写入默认值；只读场景不希望修改字典时，应考虑 `get()` 或先判断。

### 6.3 示例

```python
student = {"name": "小林", "scores": [80, 90]}
student["active"] = True
student.setdefault("skills", []).append("Python")

print(student.get("age", 0))  # 0，不会因为缺少 age 报错

python_students = {"小林", "小周", "小陈"}
sql_students = {"小周", "小吴"}

print(python_students & sql_students)  # 交集：{'小周'}
print(python_students | sql_students)  # 并集：{'小林', '小周', '小陈', '小吴'}
print(python_students - sql_students)  # 差集：{'小林', '小陈'}

words = ["python", "code", "python"]
counts = {}
for word in words:
    counts[word] = counts.get(word, 0) + 1
print(counts)  # {'python': 2, 'code': 1}
```

集合打印顺序可能因环境而不同，示例输出只表示元素内容，不代表固定排列顺序。

### 6.4 易错点和注意事项

- 不要用可变列表作为字典键或集合元素。自定义对象作为键时，相等性与哈希实现必须一致。
- 在遍历字典时直接增删键可能导致 `RuntimeError`，可先遍历 `list(d)` 或构建新字典。
- `d.get(key)` 返回 `None` 既可能表示键不存在，也可能表示键对应的值就是 `None`；需要区分时使用 `key in d`。
- 集合去重会丢失原始顺序语义；若需要去重且保持首次出现顺序，可使用 `list(dict.fromkeys(items))`。

### 6.5 常见问题

**问：字典的键可以是元组吗？**  
答：当元组及其所有嵌套元素都可哈希时可以，例如 `(2026, "Python")`；含有列表的元组不可以。

**问：字典和集合谁更适合频繁成员判断？**  
答：只关心元素是否存在时使用集合；存在键之外还需要保存对应值时使用字典。两者平均成员判断都较快。

**问：两个字典如何合并？**  
答：Python 3.9 及以上可用 `merged = a | b` 创建新字典，右侧同键值覆盖左侧；`a |= b` 会原地更新 `a`。

## 7. 类、对象与封装

### 7.1 概念

类定义对象共有的数据结构和行为，对象是类的具体实例。`__init__()` 是实例创建后的初始化方法，不是真正分配对象的构造函数；`self` 是当前实例的约定名称，调用实例方法时通常由 Python 自动传入。

封装把数据与相关操作放在一起，并控制外部修改方式。Python 的封装更多依靠命名约定和接口设计，而不是绝对禁止访问：单下划线 `_balance` 表示内部使用，`@property` 可把属性访问与方法逻辑结合。

### 7.2 原理或使用方法

实例属性通常属于某个对象，类属性由类和未覆盖它的实例共享。可变类属性若被所有实例共同修改，容易造成意外共享，因此实例自己的列表或字典应在 `__init__()` 中创建。

方法通过实例或类调用时存在绑定规则：普通实例方法接收 `self`，`@classmethod` 接收类对象，`@staticmethod` 不自动接收实例或类。静态方法不是“性能更高的方法”，只是逻辑上与类相关但不需要实例状态。

### 7.3 示例

```python
class BankAccount:
    interest_rate = 0.01  # 类属性：所有账户共享的默认利率配置

    def __init__(self, owner, balance=0):
        if balance < 0:
            raise ValueError("初始余额不能为负数")
        self.owner = owner
        self._balance = float(balance)

    @property
    def balance(self):
        return self._balance

    def deposit(self, amount):
        if amount <= 0:
            raise ValueError("存款金额必须大于 0")
        self._balance += amount
        return self._balance


account = BankAccount("小林", 100)
account.deposit(50)
print(account.owner)    # 小林
print(account.balance)  # 150.0
```

### 7.4 易错点和注意事项

- 在类体中写 `items = []` 会创建共享的类属性；每个实例需要独立列表时，应在 `__init__()` 中写 `self.items = []`。
- `obj.balance = ...` 对只有读取器的 property 会抛出 `AttributeError`，这是限制外部修改的一种方式。
- 双下划线开头会触发名称改写，不等于安全权限控制，不能用来保护敏感信息。
- 类不应承担所有功能。只包含一组相关函数时，模块级函数可能比类更直接。

### 7.5 常见问题

**问：`__init__()` 必须有返回值吗？**  
答：不能返回实例以外的业务值，应返回 `None`。它负责初始化已经创建的实例，而不是返回新实例。

**问：实例属性和类属性如何区分？**  
答：`self.name` 通常是实例属性；类体中直接定义的 `interest_rate` 是类属性。修改类属性时通过类名操作更清楚，避免意外创建同名实例属性。

**问：什么时候使用 `@property`？**  
答：需要让访问形式保持属性风格，同时在读取或写入时执行校验、计算等逻辑时使用。简单数据容器不必为每个字段都添加 property。

## 8. 继承、多态与组合

### 8.1 概念

继承让子类复用并扩展父类行为，适合表达稳定的“是一种”关系。方法重写是子类提供同名方法实现；`super()` 用于按照方法解析顺序调用父级实现。

多态是不同对象对同一操作提供各自实现。例如多种图形都提供 `area()`，调用方无需关心具体图形类。组合则是让一个对象包含其他对象，通过协作完成功能，适合“拥有”或“使用”关系。

### 8.2 原理或使用方法

Python 支持鸭子类型：只要对象提供所需接口，即使没有共同父类，也能被同一函数处理。需要明确抽象接口和约束子类时，可使用 `abc.ABC` 与 `@abstractmethod`。

多重继承的方法解析顺序可用 `ClassName.__mro__` 查看。复杂继承会增加理解成本，尤其不应仅为了复用少量代码而建立多层继承；组合往往更容易替换和测试。

### 8.3 示例

```python
import math
from abc import ABC, abstractmethod


class Shape(ABC):
    @abstractmethod
    def area(self):
        """返回图形面积。"""


class Rectangle(Shape):
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def area(self):
        return self.width * self.height


class Circle(Shape):
    def __init__(self, radius):
        if radius < 0:
            raise ValueError("半径不能为负数")
        self.radius = radius

    def area(self):
        return math.pi * self.radius ** 2


def total_area(shapes):
    return sum(shape.area() for shape in shapes)


shapes = [Rectangle(2, 3), Circle(1)]
print(total_area(shapes))  # 9.141592653589793
```

`total_area()` 只要求每个对象有可调用的 `area()`，体现了多态。若子类没有实现抽象方法，实例化时会抛出 `TypeError`。

### 8.4 易错点和注意事项

- 子类重写 `__init__()` 后不会自动执行父类初始化逻辑，需要时显式调用 `super().__init__(...)`。
- 继承应遵守可替换性：子类不应随意破坏父类公开行为的合理约定。
- 不要仅为了共享工具函数而继承；工具函数可直接放在模块中，协作者可通过组合注入。
- `isinstance()` 能检查继承关系，但过度检查具体类型会削弱多态，应优先依赖接口。

### 8.5 常见问题

**问：继承和组合如何选择？**  
答：对象确实属于父类概念的一种，并需要复用其稳定接口时可继承；只是需要另一个对象的能力时优先组合。不存在适用于所有设计的固定答案。

**问：多态是否必须通过继承实现？**  
答：不必须。Python 中不同类只要提供兼容的方法，就可以被同一调用代码使用；抽象基类主要用于表达接口约束。

**问：`super()` 是否总是调用直接父类？**  
答：不总是。它按照当前类的方法解析顺序寻找下一实现，在多重继承中尤其不能简单理解为“直接父类”。

## 9. 迭代器与生成器

### 9.1 概念

可迭代对象能够通过 `iter()` 返回迭代器；迭代器实现 `__iter__()` 和 `__next__()`，每次产生一个值，耗尽时抛出 `StopIteration`。`for` 循环会自动处理该异常，不需要调用方手动捕获。

生成器是使用 `yield` 编写的迭代器。调用生成器函数不会立即执行整个函数体，而是返回生成器对象；每次请求下一个值时执行到 `yield`，保存局部状态并暂停。

### 9.2 原理或使用方法

生成器采用惰性求值，适合大文件逐行处理、数据流转换和无限序列。它不必一次性把全部结果放入内存，但通常只能遍历一次，且随机访问不如列表方便。

生成器表达式写为 `(x * x for x in data)`，与列表推导式相比不立即构建完整列表。若结果需要反复遍历、排序或按索引访问，可以物化为列表，但应评估内存成本。

### 9.3 示例

```python
def read_non_empty_lines(path):
    """逐行读取文本，跳过空白行，不把整个文件加载到内存。"""
    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            text = line.strip()
            if text:
                yield text


def squares(numbers):
    for number in numbers:
        yield number * number


result = squares(range(4))
print(next(result))   # 0
print(list(result))   # [1, 4, 9]
print(list(result))   # []，生成器已经耗尽
```

### 9.4 易错点和注意事项

- 生成器耗尽后继续调用 `next()` 会抛出 `StopIteration`；重复遍历需要重新创建生成器或保存结果。
- 列表推导式并非一定更好。数据量大且只需遍历一次时，生成器通常更节省内存。
- 生成器内部使用 `with` 管理资源时，应确保生成器被完整消费或关闭，避免长期不消费导致资源迟迟不释放。
- 不要在生成器中隐藏过多副作用；数据处理流程复杂时，应拆成命名清晰的步骤。

### 9.5 常见问题

**问：可迭代对象和迭代器是一回事吗？**  
答：不是。列表是可迭代对象但不是迭代器，`iter([1, 2])` 会返回迭代器；迭代器本身也是可迭代对象。

**问：生成器为什么能节省内存？**  
答：生成器按请求计算并产生元素，通常只保留当前执行状态，而不是同时保存全部结果。若内部本身构建了大型列表，则不能因此减少该部分内存。

**问：如何让生成器提前释放资源？**  
答：可以调用生成器的 `close()`，或使用适当的上下文管理封装。不要在长期任务中创建生成器后既不消费也不关闭。

## 10. 装饰器

### 10.1 概念

装饰器是接收一个可调用对象并返回可调用对象的高阶函数，用于在不修改原函数主体的情况下增加日志、计时、权限检查、缓存等行为。`@decorator` 基本等价于 `func = decorator(func)`。

装饰发生在函数定义完成后，而不是每次调用时才创建装饰逻辑。装饰器返回的包装函数会替代原名字，因此应保留原函数元数据。

### 10.2 原理或使用方法

包装函数通常使用 `*args` 和 `**kwargs` 接收任意参数，再转发给原函数。`functools.wraps()` 复制名称、文档字符串等元数据，有利于调试和工具识别。

装饰器适合横切逻辑，但不宜隐藏核心业务规则。带参数装饰器会多一层工厂函数；缓存、重试等装饰器还应处理异常、线程安全和缓存键冲突，不能只关注语法。

### 10.3 示例

```python
from functools import wraps
from time import perf_counter


def timed(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = perf_counter()
        try:
            return func(*args, **kwargs)
        finally:
            elapsed = perf_counter() - start
            print(f"{func.__name__} 耗时 {elapsed:.6f} 秒")
    return wrapper


@timed
def add(a, b):
    """返回两个数的和。"""
    return a + b


print(add(2, 3))   # 5，同时输出耗时
print(add.__name__)  # add，functools.wraps 保留了原名称
```

`finally` 确保原函数抛出异常时也能记录耗时，但不会吞掉异常。生产项目通常使用日志系统而不是直接 `print()`。

### 10.4 易错点和注意事项

- 忘记在包装函数中 `return func(...)`，会使被装饰函数的结果变成 `None`。
- 不使用 `functools.wraps` 时，`__name__`、文档字符串和某些内省工具会得到包装器信息。
- 装饰器叠加有顺序，靠近函数的装饰器先应用，外层装饰器后包装；涉及缓存和异常处理时应明确顺序。
- 计时结果受机器负载、预热和输入规模影响，一次运行不能作为稳定性能结论。

### 10.5 常见问题

**问：装饰器会修改原函数内部代码吗？**  
答：通常不会。装饰器用包装函数替换原名称，在调用前后执行附加逻辑，原函数对象仍由闭包等方式引用。

**问：为什么装饰器中的包装函数常用 `*args, **kwargs`？**  
答：这样包装器可以接收并转发不同签名的位置参数和关键字参数，适合通用装饰场景。

**问：装饰器适合实现缓存吗？**  
答：可以，但要明确缓存键、生命周期、并发和失效策略。简单纯函数可考虑 `functools.lru_cache`，不应把共享缓存用于依赖用户权限或频繁变化的数据而不加区分。

## 11. 异常处理

### 11.1 概念

异常表示程序执行中无法正常继续的事件。`try` 放置可能失败的代码，`except` 处理指定异常，`else` 在没有异常时执行，`finally` 无论是否异常都执行，适合清理资源。

异常处理的目标是让程序在可恢复的边界采取明确行动，并保留足够诊断信息，而不是让所有错误都消失。

### 11.2 原理或使用方法

应捕获能够具体处理的异常类型，如 `ValueError`、`FileNotFoundError`，避免裸 `except`。裸 `except` 还会捕获 `KeyboardInterrupt`、`SystemExit` 等不应被普通业务逻辑吞掉的事件。

使用 `raise NewError(...) from exc` 可以保留异常链；直接 `raise` 可重新抛出当前异常。自定义业务异常通常继承 `Exception`，并携带必要上下文。`with` 通常是文件、锁等资源清理的首选，不必每次手写 `finally`。

### 11.3 示例

```python
def parse_age(text):
    try:
        age = int(text)
    except (TypeError, ValueError) as exc:
        raise ValueError("年龄必须是整数字符串") from exc
    else:
        if not 0 <= age <= 150:
            raise ValueError("年龄必须在 0 到 150 之间")
        return age


print(parse_age("20"))  # 20

try:
    parse_age("二十")
except ValueError as exc:
    print(f"输入无效：{exc}")  # 输入无效：年龄必须是整数字符串
    print(f"原始异常：{type(exc.__cause__).__name__}")  # ValueError
```

### 11.4 易错点和注意事项

- `except Exception: pass` 会隐藏真实故障，使后续排查困难；至少应记录完整堆栈，并决定是否继续传播。
- `try` 块不宜包裹过多无关语句，否则可能把其他位置的同类异常误认为当前操作的错误。
- 不要用异常替代所有普通条件判断；但在文件打开等存在竞态的操作中，尝试执行并处理异常通常比先检查再执行更可靠。
- `finally` 中的 `return` 会覆盖 `try` 或 `except` 中的返回值或异常，通常应避免。

### 11.5 常见问题

**问：`else` 和 `finally` 有什么区别？**  
答：`else` 只在 `try` 没有异常时执行；`finally` 无论是否发生异常都会执行。两者用途不同。

**问：什么时候应该自定义异常？**  
答：需要区分业务错误类型、在更高层统一处理或携带业务上下文时可以自定义。若内置异常已能准确表达问题，不必重复定义。

**问：捕获异常后应该继续还是退出？**  
答：取决于是否能真正恢复。输入可修正时可提示重试；关键依赖不可用且无法继续时，应记录并向上报告，不能假装操作成功。

## 12. 文件读写与 JSON

### 12.1 概念

文件读写通过文件对象完成，文本模式处理 `str`，二进制模式处理 `bytes`。常见模式有 `r` 读取、`w` 覆盖写入、`a` 追加和 `x` 独占创建；加 `b` 表示二进制模式。

JSON 是一种跨语言文本数据格式，适合对象、数组、字符串、数值、布尔值和 null。Python 的 `json` 模块负责 Python 数据与 JSON 文本之间的转换，但集合、自定义对象和日期时间默认不能直接序列化。

### 12.2 原理或使用方法

`with open(...)` 会在离开代码块时关闭文件，即使发生异常也能触发清理。文本文件应显式指定编码，跨平台项目常用 UTF-8；`pathlib.Path` 提供面向对象的路径操作。

`json.dump()` 和 `json.load()` 面向文件对象，`json.dumps()` 和 `json.loads()` 面向字符串。写 JSON 时 `ensure_ascii=False` 可保留可读中文，`indent=2` 便于查看，但会增加文件体积。

### 12.3 示例

```python
import json
from pathlib import Path

path = Path("student.json")
data = {
    "name": "小林",
    "skills": ["Python", "SQL"],
    "active": True,
    "score": 90,
}

try:
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)

    with path.open("r", encoding="utf-8") as file:
        loaded = json.load(file)

    print(loaded["name"])    # 小林
    print(loaded["skills"])  # ['Python', 'SQL']
finally:
    path.unlink(missing_ok=True)  # 清理示例文件；Python 3.8+ 支持 missing_ok
```

### 12.4 易错点和注意事项

- 使用 `w` 打开已有文件会立即清空内容，需要保留原数据时应选择追加、另存或安全替换策略。
- JSON 解析失败会抛出 `json.JSONDecodeError`；不要把不受信任文件直接当成有效 JSON。
- JSON 的 `null`、`true`、`false` 分别对应 Python 的 `None`、`True`、`False`。JSON 标准不支持注释和任意 Python 对象。
- 大文件不应无必要地使用 `read()` 一次读入，可逐行或分块处理；写重要数据时还需考虑中途失败导致文件不完整的问题。

### 12.5 常见问题

**问：`json.dump` 和 `json.dumps` 如何选择？**  
答：写入已打开的文件对象使用 `dump`；需要得到 JSON 字符串用于网络传输或进一步处理时使用 `dumps`。

**问：如何保存包含 `datetime` 的对象？**  
答：先明确序列化格式，例如转换为 ISO 8601 字符串，读取后再按约定解析。不要依赖不透明的默认字符串而不记录格式。

**问：为什么中文 JSON 会出现 `\uXXXX`？**  
答：这是默认的 ASCII 转义，语义没有丢失。需要便于人工查看时可设置 `ensure_ascii=False`，并确保文件使用 UTF-8。

## 13. 模块、包与虚拟环境

### 13.1 概念

模块通常是一个 `.py` 文件，包是组织多个模块的目录。导入机制让代码按职责拆分，避免所有逻辑集中在单个文件中。

虚拟环境为项目提供独立的 Python 解释器入口和依赖安装位置，减少不同项目之间的版本冲突。虚拟环境隔离依赖，不等于容器或完整操作系统隔离。

### 13.2 原理或使用方法

导入时会按模块搜索路径查找并执行模块顶层代码，模块通常只初始化一次并缓存到 `sys.modules`。顶层应避免执行耗时任务或产生不必要副作用，可复用代码放在函数和类中。

包内相对导入如 `from .report import build_report` 依赖包上下文，直接运行模块文件可能失败。更稳妥的方式是从项目根目录使用 `python -m package.module` 启动。

虚拟环境可用标准库 `venv` 创建。安装依赖应使用当前环境的 `python -m pip`，避免误用其他解释器对应的 pip。

### 13.3 示例

目录结构：

```text
my_project/
├─ main.py
└─ school/
   ├─ __init__.py
   └─ report.py
```

`school/report.py`：

```python
def build_report(scores):
    if not scores:
        return "暂无成绩"
    return f"平均分：{sum(scores) / len(scores):.1f}"
```

`main.py`：

```python
from school.report import build_report

print(build_report([80, 90, 100]))  # 平均分：90.0
```

创建并激活虚拟环境：

```bash
# Windows PowerShell
py -m venv .venv
.venv\Scripts\Activate.ps1

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate

# 激活后安装与导出依赖
python -m pip install requests
python -m pip freeze > requirements.txt
```

从项目根目录运行 `python main.py` 即可执行示例。若使用模块化入口，可运行 `python -m school.report`，但该文件需要自行添加入口代码才会有输出。

### 13.4 易错点和注意事项

- 不要让本地文件名与依赖模块同名，例如项目中的 `random.py` 可能遮蔽标准库 `random`，造成难以理解的导入错误。
- 修改已导入模块后，普通交互环境不一定自动重新加载；长期开发应重启解释器或使用明确的开发工具，不依赖随意 `reload()`。
- 虚拟环境目录通常不应提交到版本库；应提交依赖声明，并在新环境重新安装。
- `pip freeze` 会包含间接依赖，适合记录完整环境，但不一定是维护项目直接依赖声明的最佳方式，较大项目可使用规范的依赖配置。

### 13.5 常见问题

**问：`__init__.py` 有什么作用？**  
答：它可把目录标记为常规包并执行包初始化逻辑。较新版本也支持命名空间包，但普通项目保留简洁的 `__init__.py` 通常更直观。

**问：为什么直接运行包内文件时相对导入报错？**  
答：直接运行文件可能缺少包上下文。可从项目根目录使用 `python -m 包名.模块名`，或调整入口结构，而不是随意修改搜索路径。

**问：虚拟环境能解决所有依赖冲突吗？**  
答：不能。它隔离不同项目的安装位置，但同一项目中不兼容的依赖仍需解析；系统库、环境变量和外部服务也不由虚拟环境自动隔离。

## 14. 进程、线程、协程与 GIL

### 14.1 概念

进程具有相对独立的内存空间，线程共享进程内数据，协程是由程序在等待点主动让出执行权的轻量任务。三者都可表达并发，但成本、隔离性和适用场景不同。

传统 CPython 使用全局解释器锁 GIL，使同一进程中的多个线程通常不能同时执行 Python 字节码。I/O 等待期间线程可以释放 GIL，因此多线程仍可能改善 I/O 密集任务的吞吐量。CPython 3.13 开始出现可选的自由线程构建，不能把传统 GIL 行为无条件套到所有版本和所有解释器上。

### 14.2 原理或使用方法

- I/O 密集任务：网络请求、文件或数据库等待较多时，可使用线程或异步 I/O。线程模型直观，协程适合大量等待任务，但必须使用非阻塞库或把阻塞调用移出事件循环。
- CPU 密集任务：在传统 CPython 中通常考虑多进程，使不同进程利用多个核心。进程创建和跨进程数据传输有成本，小任务未必受益。
- 共享状态：线程需要考虑锁和竞态；多进程需要显式共享或通信；协程虽常运行于单线程，也应在 `await` 边界考虑状态一致性。

并发是组织多个任务的能力，并行是多个任务实际同时执行。使用多线程不等于所有代码都能并行加速。

### 14.3 示例

多进程计算示例，适用于传统 CPython 的 CPU 密集任务：

```python
from concurrent.futures import ProcessPoolExecutor


def square(number):
    return number * number


if __name__ == "__main__":
    with ProcessPoolExecutor() as executor:
        results = list(executor.map(square, range(5)))
    print(results)  # [0, 1, 4, 9, 16]
```

`if __name__ == "__main__":` 对 Windows 等使用进程启动方式的环境尤其重要，可避免子进程启动时重复执行入口逻辑。传给进程池的函数和参数需要支持相应序列化要求。

异步 I/O 结构示例：

```python
import asyncio


async def fetch_item(name):
    await asyncio.sleep(0.1)  # 示例中的非阻塞等待，实际项目应使用异步 I/O 库
    return f"完成：{name}"


async def main():
    results = await asyncio.gather(
        fetch_item("A"),
        fetch_item("B"),
        fetch_item("C"),
    )
    print(results)


asyncio.run(main())
```

### 14.4 易错点和注意事项

- 在协程中直接调用同步阻塞请求或长时间 CPU 计算，会阻塞事件循环；可使用合适异步库，或用 `asyncio.to_thread()` 处理部分阻塞调用，但 CPU 密集任务仍需评估进程方案。
- 多线程共享列表、计数器等状态时，简单操作也不能一律视为业务层面的原子事务；需要锁、队列或更高层同步设计。
- 进程不是越多越好，线程和协程也不是。数量应结合核心数、任务类型、下游服务容量和实测结果确定。
- 使用基准测试比较方案时，应包含数据传递和任务创建成本，不能只测单个计算函数。

### 14.5 常见问题

**问：有 GIL 是否意味着多线程没有价值？**  
答：不是。传统 CPython 的 GIL 限制 Python 字节码的多核并行，但 I/O 等待时可切换线程，因此多线程仍能改善许多 I/O 密集任务。

**问：协程是否一定比线程快？**  
答：没有这种绝对结论。协程减少部分线程切换和连接管理成本，但要求合适的异步生态；性能取决于任务、库、阻塞点和资源限制。

**问：CPU 密集任务为什么选择多进程？**  
答：传统 CPython 中每个进程有独立解释器状态，可绕过单进程 GIL 对 Python 字节码并行的限制。同时还要考虑进程开销、内存和通信成本。

## 15. 常见错误与调试方法

### 15.1 概念

调试是通过可复现输入、错误信息和程序状态定位问题的过程。语法错误使代码无法开始执行，运行时异常在执行中抛出，逻辑错误则可能正常结束但结果错误。

有效调试应先稳定复现，再缩小范围、提出假设并验证，而不是随机修改代码。堆栈信息、最小复现、断点和日志是互补工具。

### 15.2 原理或使用方法

阅读 traceback 时先看最后的异常类型与消息，再沿调用链找到自己代码的位置。检查输入值、类型、分支条件和状态变化，可以用断点暂停执行并逐行观察。

`breakpoint()` 在默认环境下调出 pdb 调试器，常用命令包括 `n` 执行下一行、`s` 进入函数、`p expression` 查看表达式、`c` 继续、`q` 退出。生产或长期运行程序更适合使用 `logging` 记录分级日志，并保留异常堆栈。

### 15.3 示例

```python
import logging

logger = logging.getLogger(__name__)


def average(numbers):
    if not numbers:
        raise ValueError("numbers 不能为空")
    return sum(numbers) / len(numbers)


def main():
    scores = [80, 90, 100]
    breakpoint()  # 调试完成后应移除或避免在主要生产流程中无条件触发
    print(f"平均分：{average(scores):.1f}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    try:
        main()
    except Exception:
        logger.exception("计算平均分失败")
        raise
```

常见异常与排查方向：

| 异常或现象 | 常见原因 | 正确排查方向 |
|---|---|---|
| `SyntaxError` | 括号、冒号或语法结构错误 | 查看报错行及其前一行，检查字符串和括号是否闭合 |
| `NameError` | 名字未定义、拼写错误或作用域不对 | 检查定义与使用顺序、大小写和变量作用域 |
| `TypeError` | 类型不支持操作、参数数量不符 | 打印或断点查看实际类型与函数签名 |
| `ValueError` | 类型可接受但值不合法 | 校验数值范围、文本格式和业务约束 |
| `IndexError` | 序列下标越界 | 检查长度、空列表和切片边界 |
| `KeyError` | 字典键不存在 | 检查键来源，按需要使用 `get()` 或成员判断 |
| `AttributeError` | 对象没有该属性或实际为 `None` | 查看对象实际类型和初始化流程 |
| `ModuleNotFoundError` | 环境缺依赖、路径错误或本地文件遮蔽 | 核对当前解释器、安装环境和模块搜索路径 |
| 无限循环 | 状态未更新或退出条件不可达 | 记录条件变量，构造最小输入并验证退出路径 |

### 15.4 易错点和注意事项

- 不要只保留“程序报错了”这句话，应保存完整异常、输入、版本和复现步骤；敏感数据需要脱敏。
- 修复后应增加能先失败、修复后通过的测试，防止相同问题回归。
- 日志不是越多越好，应记录关键状态和上下文，避免密码、令牌等敏感信息。
- 不要通过扩大 `except` 范围让错误消失；应先确定根因，再决定恢复、降级还是终止。

### 15.5 常见问题

**问：`print` 调试和断点调试如何选择？**  
答：简单状态可临时打印；调用链复杂、状态变化多时使用断点更高效。长期服务需要结构化、分级日志，而不是依赖控制台临时输出。

**问：程序不报错但结果不对怎么办？**  
答：这通常是逻辑错误。为关键函数准备已知输入和预期结果，逐步比较中间值，并用单元测试固定正确行为。

**问：怎样制作最小复现？**  
答：保留仍能触发问题的最少代码和输入，逐步移除无关文件、网络和数据库依赖。最小复现能减少干扰，也便于他人定位问题。
