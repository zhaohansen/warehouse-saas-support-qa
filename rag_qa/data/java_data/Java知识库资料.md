# Java：知识库资料

> 目标读者：掌握一门编程语言基础、正在学习 Java 的本科生。示例以 Java 17 的常用语法为基线；涉及 Java 8、Java 9 或更新版本差异时会明确说明。示例均为单个类或围绕同一主题的最小完整程序。

## 目录

1. 基本数据类型、引用类型与字符串
2. 类、对象与封装
3. 继承、重写与多态
4. 接口与抽象类
5. 异常处理
6. 集合框架：List、Set、Map
7. 泛型与类型擦除
8. 反射与注解
9. 线程、线程池与并发基础
10. 锁、原子类与死锁
11. JVM 内存区域与垃圾回收
12. 类加载机制
13. 常见设计原则与易错点

## 1. 基本数据类型、引用类型与字符串

### 1.1 概念

Java 的类型分为基本类型和引用类型。基本类型共有八种：`byte`、`short`、`int`、`long`、`float`、`double`、`char`、`boolean`，变量直接保存数值；引用类型变量保存对象引用，对象本身位于堆中，数组、类实例、接口引用和字符串都属于引用类型。

每种基本类型都有对应包装类，如 `int` 对应 `Integer`。包装类让基本值能够参与泛型和集合等只接受引用类型的场景。`String` 是引用类型且不可变，字符串拼接、截取等操作通常产生新对象。

### 1.2 原理或使用方法

自动装箱把基本值转换为包装对象，自动拆箱执行相反过程。包装对象可能为 `null`，拆箱时若直接参与运算会抛出 `NullPointerException`。比较包装对象内容应使用 `equals()`，不要用 `==`；`==` 对引用类型比较的是是否指向同一对象。

`String` 使用 `equals()` 比较内容。编译期确定的字符串字面量会进入字符串常量池，可能被复用；运行时 `new String("java")` 会创建新的对象。频繁拼接字符串时，循环内可考虑 `StringBuilder`，它是可变的字符序列，但并非所有场景都必须替换 `String`。

### 1.3 示例

```java
public class TypeAndStringDemo {
    public static void main(String[] args) {
        int count = 7;
        Integer boxed = 7;
        long total = 10_000_000_000L;

        String a = "java";
        String b = "java";
        String c = new String("java");

        System.out.println(a == b);        // true，字面量可能复用同一对象
        System.out.println(a == c);        // false，c 是另一个对象
        System.out.println(a.equals(c));   // true，内容相同

        StringBuilder builder = new StringBuilder();
        for (int i = 0; i < 3; i++) {
            builder.append(i).append(',');
        }
        System.out.println(builder);       // 0,1,2,
    }
}
```

### 1.4 易错点和注意事项

- `Integer a = 1000; Integer b = 1000; a == b` 通常为 `false`。包装类缓存范围与实现有关，不能把 `==` 当作值比较。
- 自动拆箱前要确认包装对象不为 `null`，尤其从 Map、数据库结果或方法返回值取得对象时。
- `float` 和 `double` 采用二进制浮点表示，金额计算不应默认依赖它们；可使用最小货币单位整数或 `BigDecimal`。
- Java 的 `char` 是 UTF-16 代码单元，不能完整表示所有 Unicode 字符；复杂文本处理应使用代码点或字符串 API。

### 1.5 常见问题

**问：基本类型和包装类型怎么选？**  
答：局部数值计算优先基本类型；需要放入泛型集合、允许 `null` 或使用方法提供的对象接口时使用包装类型。

**问：为什么字符串被设计为不可变？**  
答：不可变使字符串可以安全共享、作为 Map 键时哈希稳定，也便于常量池复用和并发读取；代价是修改会创建新对象。

**问：`String`、`StringBuilder`、`StringBuffer` 有何区别？**  
答：`String` 不可变；`StringBuilder` 可变且不是线程安全的，通常用于单线程拼接；`StringBuffer` 提供同步方法，但并发场景是否选择它仍需结合整体设计。

## 2. 类、对象与封装

### 2.1 概念

类描述对象共有的字段和行为，对象是类创建出的实例。字段保存对象状态，方法定义可执行行为，构造器负责对象初始化。封装是把字段设为 `private`，通过公开方法维护合法状态，而不是把所有字段直接暴露给外部。

Java 对象通过 `new` 创建，引用变量指向堆中的实例。一个类可以定义多个构造器，也可使用 `this(...)` 在一个构造器中调用另一个构造器。

### 2.2 原理或使用方法

`private` 字段不能被类外直接访问，`getter` 和 `setter` 可以在读写时校验。封装的目的不是机械生成所有 getter/setter，而是让对象始终满足业务不变量。若某个字段一经构造就不应变化，可以声明为 `final` 并只提供读取方法。

`static` 成员属于类而不是单个实例。静态字段被所有实例共享，适合常量或真正的全局类级状态；不应把每个对象独立的数据错误地声明为静态。

### 2.3 示例

```java
public class BankAccount {
    private final String owner;
    private long balanceCents;

    public BankAccount(String owner, long initialCents) {
        if (owner == null || owner.isBlank()) {
            throw new IllegalArgumentException("owner 不能为空");
        }
        if (initialCents < 0) {
            throw new IllegalArgumentException("初始余额不能为负数");
        }
        this.owner = owner;
        this.balanceCents = initialCents;
    }

    public void deposit(long cents) {
        if (cents <= 0) {
            throw new IllegalArgumentException("存款金额必须大于 0");
        }
        balanceCents += cents;
    }

    public long getBalanceCents() {
        return balanceCents;
    }

    public static void main(String[] args) {
        BankAccount account = new BankAccount("小林", 10_000);
        account.deposit(5_000);
        System.out.println(account.getBalanceCents()); // 15000
    }
}
```

### 2.4 易错点和注意事项

- 构造器中未初始化对象的关键字段，可能让对象在非法状态下被使用。
- `this.owner = owner` 中 `this` 表示当前对象，省略后可能变成字段自赋值，实际没有改变对象状态。
- Java 没有 C++ 那种析构器，不能依赖 `finalize()` 做关键资源清理；文件、连接应使用 `try-with-resources` 或显式关闭。
- `equals()` 与 `hashCode()` 往往需要一起重写，否则对象放入 `HashSet` 或作为 `HashMap` 键时行为可能与预期不符。

### 2.5 常见问题

**问：类和对象有什么区别？**  
答：类是定义字段和行为的模板，对象是按该模板创建的具体实例。一个类可以创建多个相互独立的对象。

**问：字段都写成 public 不是更方便吗？**  
答：直接公开字段会失去校验和替换实现的能力，外部代码可能把对象改成非法状态。封装能集中维护约束。

**问：`final` 对象的内容一定不能改变吗？**  
答：`final` 引用不能改指另一个对象，但被引用对象的内部字段仍可能变化。需要不可变对象时，还要保证字段不可变且没有泄露可变内部结构。

## 3. 继承、重写与多态

### 3.1 概念

继承使用 `extends` 表达“是一种”关系，子类获得父类可访问的字段和方法，并可重写父类方法。重写要求方法签名兼容，并用 `@Override` 让编译器检查。多态是父类型引用指向子类对象，运行时根据实际对象调用对应重写方法。

Java 类只支持单继承，但一个类可以实现多个接口。构造子类对象时会先执行父类构造器，再执行子类构造器。

### 3.2 原理或使用方法

实例方法默认采用动态分派：调用 `animal.speak()` 时，执行的是实际对象类型中重写后的实现。字段访问、静态方法和私有方法不按这种实例多态方式分派，因此不要把“父类引用调用”理解为对所有成员都动态选择。

`super` 可调用父类构造器或父类被重写的方法。父类构造器中的可重写方法调用需要谨慎，因为此时子类字段可能尚未完成初始化。

### 3.3 示例

```java
class Animal {
    protected final String name;

    Animal(String name) {
        this.name = name;
    }

    public String speak() {
        return name + " 发出声音";
    }
}

class Dog extends Animal {
    Dog(String name) {
        super(name);
    }

    @Override
    public String speak() {
        return name + " 汪汪叫";
    }
}

public class PolymorphismDemo {
    public static void main(String[] args) {
        Animal animal = new Dog("旺财");
        System.out.println(animal.speak()); // 旺财 汪汪叫
    }
}
```

### 3.4 易错点和注意事项

- 重写方法不能降低可见性，返回类型需兼容，抛出的受检异常不能比父类方法更宽。
- 父类引用只能访问父类声明中可见的成员；要调用子类特有方法必须先确认类型或重新设计接口。
- 继承会形成强耦合，父类行为变化可能影响所有子类。只有稳定且真实的“是一种”关系才适合继承。
- 在 `equals()` 中随意使用 `instanceof` 和可变字段，可能破坏对称性、传递性或哈希稳定性。

### 3.5 常见问题

**问：重载和重写有什么区别？**  
答：重载发生在同一类中，方法名相同而参数列表不同，编译期选择；重写发生在父子类之间，方法签名兼容，运行时按实际对象分派。

**问：父类引用指向子类对象后，能访问子类新字段吗？**  
答：不能直接访问。引用类型决定编译期可访问的成员，运行时对象类型决定重写方法调用。

**问：什么时候应该使用继承？**  
答：子类确实可以替代父类且需要复用稳定行为时。只是复用部分代码时，组合通常更容易维护。

## 4. 接口与抽象类

### 4.1 概念

接口定义一组能力或契约，类通过 `implements` 实现接口。抽象类用 `abstract` 声明，可以包含抽象方法和具体实现，也可以持有字段和构造器，但不能直接实例化。

接口适合跨不相关类型表达共同能力；抽象类适合在一组关系紧密的类之间共享部分实现和状态。

### 4.2 原理或使用方法

接口中的字段默认是 `public static final`，抽象方法默认是 `public abstract`。Java 8 起接口可有 `default` 方法和静态方法；Java 9 起可有私有方法供接口内部复用。实现类必须实现全部抽象方法，除非自身也是抽象类。

一个类只能继承一个父类，但能实现多个接口。接口多态让调用方依赖契约而不是具体实现。默认方法发生冲突时，实现类必须显式覆盖或指定使用哪一个接口实现。

### 4.3 示例

```java
interface Payment {
    boolean pay(long amountCents);

    default String describe() {
        return "支付方式：" + name();
    }

    String name();
}

class WalletPayment implements Payment {
    @Override
    public boolean pay(long amountCents) {
        if (amountCents <= 0) {
            return false;
        }
        System.out.println("钱包支付 " + amountCents + " 分");
        return true;
    }

    @Override
    public String name() {
        return "电子钱包";
    }
}

public class PaymentDemo {
    public static void main(String[] args) {
        Payment payment = new WalletPayment();
        System.out.println(payment.describe()); // 支付方式：电子钱包
        System.out.println(payment.pay(500));   // true
    }
}
```

### 4.4 易错点和注意事项

- 接口不是“没有字段的抽象类”的同义词。接口表达能力契约，抽象类可表达带有共享状态的基类。
- 接口默认方法不要滥用。大量默认实现会让契约变复杂，也可能产生实现冲突。
- 抽象类构造器中调用抽象方法有初始化顺序风险，应避免依赖尚未初始化的子类状态。
- 接口设计应稳定且职责集中；把所有方法塞进一个接口会迫使实现类提供无关行为。

### 4.5 常见问题

**问：接口和抽象类如何选择？**  
答：需要多实现、只定义能力时优先接口；需要共享字段、构造流程和部分实现且类型关系紧密时可用抽象类。两者也可配合使用。

**问：接口可以有构造器吗？**  
答：不能。接口不创建实例，构造器属于类。接口实现类的构造器可在初始化时设置状态。

**问：默认方法是否破坏兼容性？**  
答：它允许给已有接口追加方法而不强制所有实现立即修改，但若多个父接口提供冲突默认方法，实现类必须显式解决。

## 5. 异常处理

### 5.1 概念

Java 异常分为受检异常和运行时异常。受检异常继承 `Exception` 但不继承 `RuntimeException`，调用者必须捕获或在方法签名中用 `throws` 声明；运行时异常通常表示编程错误或非法状态，不强制声明。`Error` 表示严重系统问题，一般不应由普通业务代码捕获。

`try`、`catch`、`finally` 控制异常处理流程，`try-with-resources` 可自动关闭实现 `AutoCloseable` 的资源。

### 5.2 原理或使用方法

捕获异常应针对能够处理的具体类型，并从具体到宽泛排列。需要保留原始原因时，应把原异常作为新异常的 cause 传入。不要只打印异常后继续返回错误结果，也不要使用空的 `catch` 块吞掉问题。

`finally` 常用于清理，但 `try-with-resources` 更可靠，它会按资源创建顺序的逆序关闭，并正确处理关闭阶段异常与主体异常的关系。

### 5.3 示例

```java
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;

public class ExceptionDemo {
    static int parsePort(String raw) {
        try {
            int port = Integer.parseInt(raw);
            if (port < 1 || port > 65535) {
                throw new IllegalArgumentException("端口超出范围");
            }
            return port;
        } catch (NumberFormatException ex) {
            throw new IllegalArgumentException("端口必须是整数：" + raw, ex);
        }
    }

    static String firstLine(String path) throws IOException {
        try (BufferedReader reader = new BufferedReader(new FileReader(path))) {
            return reader.readLine();
        }
    }

    public static void main(String[] args) {
        System.out.println(parsePort("8080")); // 8080
        try {
            parsePort("abc");
        } catch (IllegalArgumentException ex) {
            System.out.println(ex.getMessage());
        }
    }
}
```

### 5.4 易错点和注意事项

- 不要捕获 `Exception` 后静默忽略。若无法恢复，应记录上下文并向上抛出或转换为明确的业务异常。
- `finally` 中不要写会覆盖返回结果或再次抛出异常的复杂逻辑；清理动作应尽量简单可靠。
- 受检异常是方法契约的一部分，不应为了让调用方少写代码而随意包装成运行时异常。
- 异常消息不应包含密码、密钥等敏感信息，日志应保留异常堆栈而不是只记录消息。

### 5.5 常见问题

**问：运行时异常还需要捕获吗？**  
答：可以捕获，但通常应在边界统一处理并修正根因。对于参数错误，优先在方法入口校验并抛出清晰的 `IllegalArgumentException`。

**问：`try-with-resources` 相比 `finally` 有什么优势？**  
答：它能自动关闭多个资源并处理关闭异常，减少漏关和覆盖原异常的风险，代码也更集中。

**问：什么时候自定义异常？**  
答：需要区分业务错误、向上层传递上下文或统一处理时。自定义异常应提供有意义的类型和消息，不要为每个小错误建立无差别异常层级。

## 6. 集合框架：List、Set、Map

### 6.1 概念

集合框架用于组织一组对象。`List` 有序、可重复，常用实现有 `ArrayList`（随机访问较快）和 `LinkedList`（两端操作方便，但随机访问和遍历缓存局部性通常不如数组结构）。`Set` 不允许重复元素，常用 `HashSet`、`LinkedHashSet` 和 `TreeSet`。`Map` 保存键值映射，常用 `HashMap`、`LinkedHashMap`、`TreeMap`。

接口类型应优先出现在变量和方法参数中，例如 `List<String>`，便于替换实现。具体选型要结合顺序、排序、并发和性能需求，不能只背“某个集合更快”。

### 6.2 原理或使用方法

`ArrayList` 基于动态数组，尾部追加通常具有摊销 O(1) 成本，中间插入删除通常 O(n)。`HashSet` 与 `HashMap` 基于哈希，平均查找 O(1)，最坏情况受哈希冲突影响；`TreeSet` 与 `TreeMap` 基于有序结构，查找和更新通常 O(log n)。

自定义对象作为哈希键时，必须正确实现 `equals()` 与 `hashCode()`，并保证相等对象具有相同哈希值。`HashMap` 不保证迭代顺序；需要插入顺序用 `LinkedHashMap`，需要按键排序用 `TreeMap`。

### 6.3 示例

```java
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class CollectionDemo {
    public static void main(String[] args) {
        List<String> skills = new ArrayList<>();
        skills.add("Java");
        skills.add("SQL");
        skills.add("Java");
        skills.sort(Comparator.naturalOrder());
        System.out.println(skills); // [Java, Java, SQL]

        Map<String, Integer> counts = new HashMap<>();
        for (String skill : skills) {
            counts.merge(skill, 1, Integer::sum);
        }
        System.out.println(counts); // {Java=2, SQL=1}

        for (Map.Entry<String, Integer> entry : counts.entrySet()) {
            System.out.println(entry.getKey() + ": " + entry.getValue());
        }
    }
}
```

`counts.merge(key, 1, Integer::sum)` 表示键不存在时放入 1，存在时把旧值与 1 相加，避免手写判空。

### 6.4 易错点和注意事项

- 遍历集合时直接调用集合的 `remove()` 可能导致 `ConcurrentModificationException`；可显式使用 `Iterator.remove()`，或通过 `removeIf()` 等受支持方式修改。
- `Arrays.asList()` 返回的列表长度固定，不能随意 `add()` 或 `remove()`；需要可变列表时创建新的 `ArrayList`。
- `List.of()`、`Map.of()` 等工厂方法创建不可变集合，向其中添加元素会抛异常；`Map.of()` 还不接受重复键和 `null`。
- 集合中保存可变对象时，修改参与 `equals/hashCode` 的字段会使哈希定位失效。

### 6.5 常见问题

**问：`ArrayList` 和 `LinkedList` 如何选择？**  
答：需要按索引访问、遍历和尾部追加时通常 `ArrayList` 更合适；确实频繁在两端插入删除且有测量依据时再考虑 `LinkedList`。

**问：`HashMap` 的键可以为 `null` 吗？**  
答：`HashMap` 允许一个 `null` 键和多个 `null` 值；`Hashtable` 不允许。业务代码是否允许 `null` 应由自身约束决定，`ConcurrentHashMap` 也不允许 `null` 键或值。

**问：如何在遍历 Map 时删除元素？**  
答：可使用 `Iterator` 的 `remove()`，或调用 `map.entrySet().removeIf(...)`。不要在用增强 for 遍历 `entrySet()` 时直接 `map.remove()`。

## 7. 泛型与类型擦除

### 7.1 概念

泛型让类、接口和方法在定义时使用类型参数，在编译期获得类型检查，减少强制转换。例如 `List<String>` 表示只应容纳字符串的列表。

Java 泛型主要通过类型擦除实现：编译后类型参数通常被替换为上界或 `Object`，运行时一般拿不到具体的 `List<String>` 类型信息。因此不能创建 `new T()`、`new T[]`，也不能对泛型类型使用 `instanceof`。

### 7.2 原理或使用方法

上界 `<T extends Number>` 限定 T 必须为 Number 或其子类。通配符 `? extends Number` 表示“某种 Number 子类型”的只读集合，适合读取；`? super Integer` 表示可接收 Integer 或其父类型的写入目标。常用原则是生产者用 `extends`、消费者用 `super`，但仍应结合接口设计理解。

基本类型不能直接作为类型参数，必须使用包装类型；这会让集合存储产生装箱和拆箱成本。

### 7.3 示例

```java
import java.util.List;

public class GenericDemo {
    static double sum(List<? extends Number> values) {
        double total = 0;
        for (Number value : values) {
            total += value.doubleValue();
        }
        return total;
    }

    static <T extends Comparable<T>> T max(List<T> values) {
        if (values.isEmpty()) {
            throw new IllegalArgumentException("列表不能为空");
        }
        T result = values.get(0);
        for (T value : values) {
            if (value.compareTo(result) > 0) {
                result = value;
            }
        }
        return result;
    }

    public static void main(String[] args) {
        System.out.println(sum(List.of(1, 2.5, 3))); // 6.5
        System.out.println(max(List.of("Java", "SQL", "AI"))); // SQL
    }
}
```

### 7.4 易错点和注意事项

- 不要用原始类型 `List` 代替 `List<String>`，否则会失去编译期类型检查。
- `List<Object>` 不能接收 `List<String>`，二者不因元素继承关系而互为子类型；需要通配符表达兼容性。
- 泛型类型参数在运行时被擦除，不能依赖它做多态重载，例如仅返回类型不同的 `foo(List<String>)` 与 `foo(List<Integer>)` 不能共存。
- 可变参数配合泛型会产生泛型数组警告，框架代码中如需使用应确保类型安全并理解 `@SafeVarargs` 的适用条件。

### 7.5 常见问题

**问：为什么不能 `new T()`？**  
答：构造对象需要运行时类型信息，而类型参数被擦除。可传入 `Class<T>` 或工厂对象，由调用方提供创建能力。

**问：`List<?>` 和 `List<Object>` 一样吗？**  
答：不一样。`List<Object>` 明确元素是 Object；`List<?>` 表示未知元素类型，通常只能读取为 Object，不能随意添加非 null 元素。

**问：泛型会带来运行时开销吗？**  
答：类型检查主要发生在编译期，运行时通常没有专门的具体泛型类型；但使用包装类型集合可能产生装箱、拆箱和对象分配成本。

## 8. 反射与注解

### 8.1 概念

反射允许程序在运行时检查类、方法、字段、构造器等信息，并可创建实例、调用方法或访问字段。它常用于框架、序列化、依赖注入和测试工具，但会绕过部分编译期检查。

注解是为代码添加元数据的语法机制。注解本身不自动产生行为，需要处理器、反射或框架读取并执行相应逻辑。元注解 `@Retention` 决定注解保留到源码、类文件还是运行时，`@Target` 限定可标注的位置。

### 8.2 原理或使用方法

`Class<?>` 对象是反射入口，可通过 `类名.class`、`obj.getClass()` 或 `Class.forName()` 获取。`getDeclaredFields()` 能取得本类声明的字段，包括私有字段；访问前可能需要 `setAccessible(true)`。Java 模块系统下，跨模块访问非公开成员可能被拒绝。

反射调用应处理 `ClassNotFoundException`、`NoSuchMethodException`、`IllegalAccessException`、`InvocationTargetException` 等异常。框架中可封装这些差异，业务代码不宜到处散落反射。

### 8.3 示例

```java
import java.lang.annotation.ElementType;
import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;
import java.lang.annotation.Target;
import java.lang.reflect.Field;

@Retention(RetentionPolicy.RUNTIME)
@Target(ElementType.FIELD)
@interface Label {
    String value();
}

class Course {
    @Label("课程名称")
    private String name = "Java 程序设计";
}

public class ReflectionDemo {
    public static void main(String[] args) throws Exception {
        Course course = new Course();
        Field field = Course.class.getDeclaredField("name");
        field.setAccessible(true);

        Label label = field.getAnnotation(Label.class);
        System.out.println(label.value() + "：" + field.get(course));
        // 课程名称：Java 程序设计
    }
}
```

### 8.4 易错点和注意事项

- 反射绕过编译期类型检查，字段名或方法名写错往往到运行时才暴露；应集中封装并补充测试。
- 反射访问私有成员会破坏封装，也可能受模块访问限制，不应作为普通业务调用的首选。
- 注解未设置 `RetentionPolicy.RUNTIME` 时，运行时反射读取不到；自定义注解要同时检查保留策略和目标位置。
- 反射和动态代理有性能与可维护性成本，框架场景可接受，但不应无测量地用于高频热点路径。

### 8.5 常见问题

**问：注解能直接改变方法行为吗？**  
答：不能。注解只提供元数据，必须由编译器插件、反射代码或框架解释后才会产生行为。

**问：`getFields()` 和 `getDeclaredFields()` 有什么区别？**  
答：前者返回类及其父类中可访问的公开字段；后者返回当前类声明的所有字段，包括私有字段，但不自动包含父类字段。

**问：反射是否完全不受限制？**  
答：不是。模块系统、安全管理器（在仍受支持的版本中）和访问规则都可能限制反射，访问失败应作为正常错误路径处理。

## 9. 线程、线程池与并发基础

### 9.1 概念

线程是进程内的执行单元，多个线程共享进程堆内存，但有各自的栈和程序计数器。创建线程可通过继承 `Thread` 或实现 `Runnable`，更推荐把任务与线程管理分离。

线程池复用工作线程并管理任务队列，避免为每个请求创建线程带来的开销。`ExecutorService` 是常用抽象，`Callable` 任务可以返回结果，`Future` 表示异步计算结果。

### 9.2 原理或使用方法

线程池通常包含核心线程数、最大线程数、任务队列、线程工厂和拒绝策略。任务提交后，池根据配置决定新建线程、排队或拒绝。队列无界且任务生产速度长期高于消费速度时，可能耗尽内存。

任务结束后应关闭线程池。`shutdown()` 不再接收新任务但等待已提交任务完成；`shutdownNow()` 尝试中断正在执行的任务并返回未开始任务。是否可中断取决于任务是否响应中断。

### 9.3 示例

```java
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

public class ThreadPoolDemo {
    public static void main(String[] args) throws Exception {
        ExecutorService pool = Executors.newFixedThreadPool(2);
        try {
            Future<Integer> future = pool.submit(() -> {
                int total = 0;
                for (int i = 1; i <= 100; i++) {
                    total += i;
                }
                return total;
            });
            System.out.println(future.get()); // 5050
        } finally {
            pool.shutdown();
        }
    }
}
```

### 9.4 易错点和注意事项

- 不要在每次请求中创建新线程池却不关闭，线程和队列会持续占用资源。
- 共享可变状态仍需要同步。线程池只负责调度，不会自动保证业务数据线程安全。
- 捕获 `Future.get()` 抛出的 `ExecutionException` 并检查 cause，否则会丢失任务内部真实异常。
- 生产环境通常应显式配置线程池参数和拒绝策略，而不是长期依赖无界队列或默认工厂方法。

### 9.5 常见问题

**问：`Runnable` 和 `Callable` 有什么区别？**  
答：`Runnable.run()` 不返回结果且不能声明受检异常；`Callable.call()` 可返回结果并声明异常，适合提交给线程池获取 `Future`。

**问：线程越多吞吐量越高吗？**  
答：不是。线程过多会增加上下文切换、内存和竞争成本，还可能压垮下游服务；应结合任务类型和实测确定规模。

**问：如何正确停止线程池？**  
答：先停止提交新任务，再调用 `shutdown()` 等待完成；需要快速停止时可 `shutdownNow()` 并配合任务的中断处理，不能假设它一定立即终止所有任务。

## 10. 锁、原子类与死锁

### 10.1 概念

竞态条件发生在多个线程访问共享可变状态且执行结果依赖调度顺序时。锁用于把一组操作变成互斥临界区。`synchronized` 是 Java 内置锁，可修饰实例方法、静态方法或代码块；`ReentrantLock` 提供显式加锁、可中断等待、超时和公平性等能力。

原子类如 `AtomicInteger` 通过原子操作更新单个变量，适合简单计数，但不能自动保证多个变量之间的复合操作一致。

### 10.2 原理或使用方法

`synchronized` 锁对象很重要：实例方法通常锁定当前实例，静态方法锁定类对象。多个线程必须竞争同一把锁才能实现互斥；锁不同对象不会互斥。

死锁常见条件包括互斥、持有并等待、不可抢占和循环等待。工程上可通过固定锁顺序、缩小临界区、设置超时和减少嵌套锁降低风险。`ReentrantLock.tryLock()` 可避免无限等待，但必须在成功获取后于 `finally` 中解锁。

### 10.3 示例

```java
import java.util.concurrent.atomic.AtomicInteger;

class Counter {
    private int value;
    private final AtomicInteger attempts = new AtomicInteger();

    public synchronized void increment() {
        value++;
        attempts.incrementAndGet();
    }

    public synchronized int value() {
        return value;
    }

    public int attempts() {
        return attempts.get();
    }
}

public class LockDemo {
    public static void main(String[] args) throws InterruptedException {
        Counter counter = new Counter();
        Thread t1 = new Thread(() -> {
            for (int i = 0; i < 1000; i++) counter.increment();
        });
        Thread t2 = new Thread(() -> {
            for (int i = 0; i < 1000; i++) counter.increment();
        });
        t1.start();
        t2.start();
        t1.join();
        t2.join();
        System.out.println(counter.value());    // 2000
        System.out.println(counter.attempts()); // 2000
    }
}
```

### 10.4 易错点和注意事项

- 只给写操作加锁、读取时不加锁，可能读到不一致的中间状态；应让同一组不变量在统一同步策略下访问。
- 锁对象必须对所有相关线程一致，使用 `synchronized (new Object())` 作为局部锁没有互斥意义。
- 嵌套加锁顺序不一致容易死锁；不要在持锁时执行不受控的网络调用或长时间任务。
- `volatile` 保证可见性和一定的有序性，但不保证复合操作原子性，`count++` 仍不是线程安全操作。

### 10.5 常见问题

**问：`synchronized` 和 `ReentrantLock` 如何选择？**  
答：简单互斥优先 `synchronized`，代码简洁且自动释放；需要超时、可中断、公平锁或多个条件变量时才考虑 `ReentrantLock`。

**问：原子类能替代所有锁吗？**  
答：不能。它适合单变量原子更新；涉及多个字段、外部资源或“检查后再更新”的复合业务不变量时仍需锁或事务性设计。

**问：死锁发生后如何排查？**  
答：可用 `jstack` 查看线程栈和锁等待关系，确认循环等待。修复应从统一锁顺序、减少锁粒度或超时重试入手，而不是简单增加线程数。

## 11. JVM 内存区域与垃圾回收

### 11.1 概念

JVM 运行时数据区常分为堆、虚拟机栈、本地方法栈、程序计数器和元空间等。堆存放对象实例，通常由所有线程共享；每个线程有独立的栈和程序计数器。Java 8 起，类元数据主要放在本地内存中的元空间，永久代不再是常见 HotSpot 配置。

垃圾回收自动回收不可达对象。判断对象是否可达通常从 GC Roots 出发做可达性分析，而不是简单看引用计数；因此循环引用本身不会阻止回收。

### 11.2 原理或使用方法

堆常按分代思路划分新生代和老年代。新生代对象大多生命周期短，常采用复制式回收；存活较久的对象晋升老年代。具体收集器、分区和参数因 JDK 版本与 JVM 实现而异，不能把某一种回收流程当成所有 JVM 的固定行为。

常见收集器包括 Serial、Parallel、G1、ZGC 等，适用于不同堆规模与停顿目标。G1 以区域为单位管理堆，ZGC 关注低停顿，但是否合适仍取决于版本、堆大小、吞吐量目标和实测。

### 11.3 示例

```java
import java.util.ArrayList;
import java.util.List;

public class MemoryDemo {
    public static void main(String[] args) {
        List<byte[]> retained = new ArrayList<>();
        for (int i = 0; i < 10; i++) {
            retained.add(new byte[1024 * 1024]); // 每项约 1 MiB
        }
        System.out.println("retained arrays: " + retained.size());
        retained.clear(); // 清除强引用后，这些数组才可能被回收
    }
}
```

该示例用于说明强引用和对象生命周期，不应把 `System.gc()` 当作常规内存管理手段。实际堆占用还受对象头、对齐和 JVM 实现影响。

### 11.4 易错点和注意事项

- 内存泄漏不等于“对象永远无法回收”这么简单，常见原因是集合持续增长、监听器未注销、线程池或缓存持有对象。
- `OutOfMemoryError: Java heap space` 表示堆空间不足；`Metaspace` 不足、栈溢出和本地内存不足是不同问题，不能统一通过加大 `-Xmx` 解决。
- 频繁 Full GC 应结合 GC 日志、堆转储和分配速率分析，不能仅凭一次监控截图断定原因。
- 调优前要明确吞吐量、延迟和内存预算目标；盲目增大堆可能延长单次回收停顿。

### 11.5 常见问题

**问：Java 对象一定分配在堆上吗？**  
答：逻辑上对象主要由堆管理，但 JIT 编译器等实现可能做逃逸分析和标量替换等优化。讨论内存模型时应区分语言规范与具体 JVM 实现。

**问：有了垃圾回收还需要关心内存吗？**  
答：需要。回收器只能回收不可达对象，持续被集合、缓存或线程引用的对象仍会占用内存。

**问：G1 是否一定比 Parallel 快？**  
答：不能这样断言。收集器选择取决于延迟目标、吞吐量、堆大小和负载特征，应通过代表性压测与 GC 日志比较。

## 12. 类加载机制

### 12.1 概念

类从字节码到可使用通常经历加载、链接和初始化等阶段。加载阶段查找并读取类文件，链接包括验证、准备和解析，初始化阶段执行静态字段赋值和静态初始化块。

类加载器负责定位和定义类。常见层次包括启动类加载器、平台类加载器和应用类加载器。双亲委派模型让加载请求先向上委托，以减少核心类被随意替换的风险；自定义类加载器在插件、隔离和热部署中会打破或扩展这一模型。

### 12.2 原理或使用方法

同一个类文件被不同类加载器加载，可能形成两个在 JVM 中不相同的类，即使全限定名相同。这解释了为什么某些框架中的类型转换异常会提示“同一类但由不同加载器加载”。

类初始化通常延迟到首次主动使用时，例如创建实例、访问静态字段或调用静态方法。仅声明数组引用不一定触发初始化。静态初始化块只执行一次，应避免耗时或可能失败的关键逻辑。

### 12.3 示例

```java
class Config {
    static {
        System.out.println("Config 初始化");
    }

    static final String NAME = "demo".toUpperCase();
}

public class ClassLoadingDemo {
    public static void main(String[] args) {
        System.out.println("main 开始");
        System.out.println(Config.NAME);
        System.out.println(Config.NAME);
    }
}
```

典型输出中“Config 初始化”只出现一次。若静态字段是编译期常量，编译器可能把值内联，实际触发时机需结合常量表达式判断。

### 12.4 易错点和注意事项

- `ClassNotFoundException` 常见于显式按名称加载但类路径缺失；`NoClassDefFoundError` 常表示编译时存在、运行时依赖或初始化失败，排查方向不同。
- 类加载器持有类和静态状态，插件或热部署若未释放加载器，可能导致内存泄漏。
- 不要假设所有环境都严格遵循同一种双亲委派实现；容器和框架常有自定义加载策略。
- 静态初始化块抛出的异常会被包装为 `ExceptionInInitializerError`，根因仍需查看 cause。

### 12.5 常见问题

**问：双亲委派的作用是什么？**  
答：优先让上层加载器查找类，避免应用自定义同名核心类造成冲突，并提升类加载的一致性和安全性。

**问：为什么两个全限定名相同的类不能互相赋值？**  
答：JVM 中类的身份由全限定名和 defining class loader 共同决定，不同加载器定义的同名类不是同一个运行时类型。

**问：静态变量什么时候初始化？**  
答：通常在类首次主动使用时进行初始化，且只执行一次。编译期常量内联等优化会影响具体观察到的时机。

## 13. 常见设计原则与易错点

### 13.1 概念

面向对象设计原则常用于控制模块耦合、职责划分和扩展成本。常见表述包括单一职责、开闭原则、里氏替换、接口隔离和依赖倒置。它们不是强制语法规则，也不是机械套用的模板，需要结合业务变化点权衡。

好的设计通常表现为职责清晰、依赖方向合理、对象状态可维护、错误边界明确，并有测试保护。设计模式是经验总结，不应为了“用了模式”而增加无必要抽象。

### 13.2 原理或使用方法

优先组合而非继承，可以让对象通过注入协作者替换行为。依赖接口而不是具体实现，有利于测试时使用替代实现。接口应围绕调用方需要保持小而稳定，避免一个巨大接口迫使实现类承担无关方法。

不可变对象减少共享状态变化，便于并发使用；但也要考虑复制和内存成本。异常、日志、资源关闭和空值处理应在边界统一约定，避免每个调用点重复判断。

### 13.3 示例

```java
interface Notifier {
    void send(String message);
}

class ConsoleNotifier implements Notifier {
    @Override
    public void send(String message) {
        System.out.println("[通知] " + message);
    }
}

class OrderService {
    private final Notifier notifier;

    OrderService(Notifier notifier) {
        this.notifier = notifier;
    }

    void complete(String orderId) {
        if (orderId == null || orderId.isBlank()) {
            throw new IllegalArgumentException("订单号不能为空");
        }
        // 省略业务处理，示例只展示依赖注入和接口协作
        notifier.send("订单已完成：" + orderId);
    }
}

public class DesignDemo {
    public static void main(String[] args) {
        new OrderService(new ConsoleNotifier()).complete("A1001");
    }
}
```

### 13.4 易错点和注意事项

- 不要把所有类都做成单例。单例引入全局状态，可能增加测试隔离和并发成本。
- 重写 `equals()` 时同时重写 `hashCode()`；用于排序或 `TreeMap` 键时还要保证比较逻辑与相等语义一致。
- 日期时间优先使用 `java.time` 包，避免旧日期类可变性和时区处理带来的混淆。
- 不要过早抽象。只有出现稳定变化点或真实复用需求时，抽象才有助于降低维护成本。

### 13.5 常见问题

**问：设计原则必须全部遵守吗？**  
答：不需要也不能机械遵守。原则用于帮助权衡，过度拆分会产生大量无意义类和间接层，应结合规模、变化频率与团队维护成本。

**问：为什么重写 `equals()` 还要重写 `hashCode()`？**  
答：哈希集合要求相等对象具有相同哈希值。只重写 `equals()` 可能让两个相等对象落入不同哈希桶，导致查找和去重异常。

**问：如何减少 Java 项目中的空指针异常？**  
答：在边界校验输入、避免返回无意义的 `null`、使用空集合或 `Optional` 表达“可能没有值”，并明确哪些字段允许为空；不要用 `Optional` 包装所有字段。
