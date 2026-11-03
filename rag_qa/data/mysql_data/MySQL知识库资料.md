# MySQL：知识库资料

> 目标读者：具备基本数据库概念、希望理解 MySQL 使用与 InnoDB 原理的本科生。SQL 示例以 MySQL 8.0 的常用语法为基线；涉及 5.7 与 8.0 差异、存储引擎或隔离级别时会单独说明。示例中的表名和数据仅用于教学，执行前应在测试库确认字符集、权限和业务约束。

## 目录

1. 数据库基本概念与 SQL 分类
2. 表结构、数据类型与约束
3. 查询、排序、分组与连接
4. InnoDB 存储引擎
5. B+ 树索引与联合索引
6. 索引失效与执行计划
7. 事务、ACID 与隔离级别
8. MVCC 多版本并发控制
9. 锁与死锁
10. 慢查询与 SQL 优化
11. 常见设计问题与优化思路

## 1. 数据库基本概念与 SQL 分类

### 1.1 概念

数据库按一定结构组织数据，数据库管理系统负责存储、查询、并发控制、权限和恢复。MySQL 是关系型数据库管理系统，数据组织为库、表、行和列；表由字段定义结构，行表示具体记录。

SQL 是关系数据库的查询和操作语言，常按用途分为：DDL（`CREATE`、`ALTER`、`DROP`）、DML（`INSERT`、`UPDATE`、`DELETE`）、DQL（`SELECT`）、DCL（`GRANT`、`REVOKE`）和事务控制语句（`COMMIT`、`ROLLBACK`）。分类是理解工具，不是所有数据库都严格使用同一术语。

### 1.2 原理或使用方法

数据库通过模式约束保证结构一致，通过事务和锁处理并发，通过日志与备份支持恢复。应用不应把数据库仅当作文件容器，连接、事务边界、索引和 SQL 写法都会影响正确性与性能。

SQL 的书写顺序与实际逻辑处理顺序不同。常见查询中，`FROM/JOIN` 确定数据源，`WHERE` 过滤行，`GROUP BY` 分组，`HAVING` 过滤组，`SELECT` 计算投影，`ORDER BY` 排序，最后 `LIMIT` 截取。理解该顺序有助于判断别名和聚合函数能用在哪里。

### 1.3 示例

```sql
CREATE DATABASE IF NOT EXISTS school
  CHARACTER SET utf8mb4
  COLLATE utf8mb4_0900_ai_ci;

USE school;

CREATE TABLE student (
  id BIGINT UNSIGNED PRIMARY KEY AUTO_INCREMENT,
  name VARCHAR(64) NOT NULL,
  score DECIMAL(5,2) NOT NULL,
  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
);

INSERT INTO student (name, score) VALUES
  ('小林', 88.50),
  ('小周', 92.00);

SELECT id, name, score
FROM student
WHERE score >= 90
ORDER BY score DESC;
```

`utf8mb4` 可存储常见 Unicode 字符，包括部分需要四字节编码的字符。`utf8mb4_0900_ai_ci` 是 MySQL 8.0 的排序规则；若迁移到较旧版本，需要选择该版本支持的规则。

### 1.4 易错点和注意事项

- 不要把 `utf8` 与 `utf8mb4` 混为一谈。MySQL 中历史 `utf8` 通常是三字节实现，不能覆盖全部 Unicode 字符。
- DDL、DML 和查询都应该在明确环境中执行；`DROP`、无 `WHERE` 的 `DELETE` 可能造成严重数据损失。
- SQL 大小写和标识符规则受平台配置影响，跨平台项目应统一命名规范，避免依赖大小写差异。
- 连接断开、事务回滚和客户端自动提交设置会影响写入结果，不能只看一条 SQL 是否返回成功。

### 1.5 常见问题

**问：MySQL 和 SQL 是一回事吗？**  
答：不是。SQL 是语言，MySQL 是支持 SQL 的具体数据库产品，并有自己的数据类型、存储引擎和语法扩展。

**问：为什么查询里 `SELECT` 写在前面，却说它后面才执行？**  
答：书写顺序服务于可读性，逻辑处理顺序用于理解语义。数据库优化器还会在保持结果等价的前提下调整执行方式。

**问：开发时可以直接改生产库表结构吗？**  
答：不应随意操作。应先评估锁表、数据量、回滚和备份，在测试环境验证迁移脚本，并按变更流程执行。

## 2. 表结构、数据类型与约束

### 2.1 概念

表结构由字段名、数据类型、是否允许空值、默认值和约束组成。常用数据类型包括整数、定点小数 `DECIMAL`、浮点 `FLOAT/DOUBLE`、日期时间、字符串和二进制类型。约束用于表达数据规则，如主键、唯一键、非空、外键和检查约束。

主键唯一标识一行，不能为 `NULL`；唯一索引允许在 MySQL 中存在多个 `NULL`（具体行为由索引和版本规则决定）。外键用于维护表间引用完整性，但是否启用取决于业务架构和性能取舍。

### 2.2 原理或使用方法

金额等要求十进制精度的数据通常使用 `DECIMAL`，避免二进制浮点误差。可变长字符串用 `VARCHAR`，固定长度编码可用 `CHAR`，但不应为了微小空间收益牺牲可维护性。时间类型要区分“日期”“时间点”和“时间戳”，并统一时区处理策略。

InnoDB 表通常应有明确主键。自增整数主键较短且有序，有利于聚簇索引；业务编号若很长且随机，作为主键会增加索引体积和页分裂风险，可另建唯一索引。

### 2.3 示例

```sql
CREATE TABLE course (
  id BIGINT UNSIGNED PRIMARY KEY AUTO_INCREMENT,
  code VARCHAR(32) NOT NULL,
  title VARCHAR(128) NOT NULL,
  credits DECIMAL(3,1) NOT NULL,
  status ENUM('draft', 'published', 'archived') NOT NULL DEFAULT 'draft',
  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  CONSTRAINT uk_course_code UNIQUE (code),
  CONSTRAINT ck_course_credits CHECK (credits > 0 AND credits <= 20)
);

CREATE TABLE enrollment (
  student_id BIGINT UNSIGNED NOT NULL,
  course_id BIGINT UNSIGNED NOT NULL,
  enrolled_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (student_id, course_id),
  CONSTRAINT fk_enrollment_student
    FOREIGN KEY (student_id) REFERENCES student(id),
  CONSTRAINT fk_enrollment_course
    FOREIGN KEY (course_id) REFERENCES course(id)
);
```

`enrollment` 使用联合主键防止同一学生重复选同一门课。外键要求被引用列有合适索引；示例中的 `student(id)` 和 `course(id)` 是主键，满足条件。

### 2.4 易错点和注意事项

- `CHECK` 约束在 MySQL 8.0.16 及以后按标准执行；更早版本可能解析但忽略，不能据此认为旧版本已保证规则。
- 不要用字符串保存所有数字和日期，这会增加比较、排序和校验成本；也不要过早使用 `JSON` 替代稳定关系结构。
- 外键会增加写入时的约束检查，跨服务或分库场景常由应用层维护一致性；这不是“外键一定差”，而是架构权衡。
- 修改大表结构可能触发表重建或长时间锁，应先确认版本支持的在线 DDL 能力和业务低峰窗口。

### 2.5 常见问题

**问：主键用自增 ID 还是 UUID？**  
答：自增 ID 更短且顺序写入友好；UUID 便于分布式生成但索引更大、随机写入更分散。选择应结合分库、合并和安全需求。

**问：`VARCHAR(255)` 和 `VARCHAR(1000)` 在存储上一样吗？**  
答：实际占用主要取决于内容长度和长度前缀等规则，但声明上限会影响行格式、索引长度限制和内存估算，不应无脑设得极大。

**问：唯一索引能保证字段永不为空吗？**  
答：不能。唯一索引约束唯一性，非空约束应使用 `NOT NULL`。多个 `NULL` 的唯一性行为需要结合 MySQL 版本和索引定义验证。

## 3. 查询、排序、分组与连接

### 3.1 概念

`SELECT` 从表中投影列，`WHERE` 过滤行，`ORDER BY` 排序，`GROUP BY` 把行按分组键聚合，`HAVING` 过滤聚合结果，`LIMIT` 限制返回行数。连接把多张表按关联条件组合，常见有内连接、左连接和右连接。

聚合函数包括 `COUNT`、`SUM`、`AVG`、`MAX`、`MIN`。`COUNT(*)` 统计行数，`COUNT(column)` 不统计该列为 `NULL` 的行，两者语义不同。

### 3.2 原理或使用方法

内连接只保留满足连接条件的组合；左连接保留左表全部行，右表无匹配时右表列为 `NULL`。若在 `WHERE` 中把右表列限定为某个非空值，左连接可能退化成类似内连接的效果，条件应放在 `ON` 或 `WHERE` 中需按语义选择。

MySQL 8.0 默认启用 `ONLY_FULL_GROUP_BY` 时，`SELECT` 中非聚合列通常必须出现在 `GROUP BY` 中，或由函数依赖保证唯一。这样可避免返回不确定的分组代表值。

### 3.3 示例

```sql
SELECT c.id, c.title, COUNT(e.student_id) AS student_count
FROM course AS c
LEFT JOIN enrollment AS e ON e.course_id = c.id
WHERE c.status = 'published'
GROUP BY c.id, c.title
HAVING COUNT(e.student_id) >= 1
ORDER BY student_count DESC, c.id ASC
LIMIT 10;
```

按课程分组统计选课人数。`LEFT JOIN` 保留没有选课记录的已发布课程，但 `HAVING` 又要求至少一名学生，因此结果只包含有选课记录的课程。

```sql
SELECT s.name, c.title
FROM student AS s
JOIN enrollment AS e ON e.student_id = s.id
JOIN course AS c ON c.id = e.course_id
ORDER BY s.name, c.title;
```

### 3.4 易错点和注意事项

- `WHERE score != NULL` 不会得到预期结果，因为与 `NULL` 的比较结果是未知值。应写 `score IS NOT NULL`。
- `NOT IN` 的子查询若返回 `NULL`，整体结果可能为空；应排除子查询中的 `NULL` 或改用 `NOT EXISTS`。
- 分组后选取非聚合列要明确语义，不要依赖关闭 `ONLY_FULL_GROUP_BY` 后的不确定结果。
- 深分页 `LIMIT 100000, 20` 可能扫描大量行；可按稳定排序键使用“上一页最后一个键”的游标分页，并确保排序键有索引。

### 3.5 常见问题

**问：`LEFT JOIN` 和 `INNER JOIN` 有什么区别？**  
答：内连接只返回匹配行；左连接保留左表所有行，右表无匹配时补 `NULL`。

**问：`HAVING` 能不能代替 `WHERE`？**  
答：语法上有时可以，但语义和性能不同。能在分组前过滤的行条件应放 `WHERE`，聚合结果过滤放 `HAVING`。

**问：`COUNT(*)` 一定比 `COUNT(1)` 慢吗？**  
答：没有普遍结论。优化器会对常见写法做处理，实际差异应通过执行计划和版本测试判断，不应把固定口诀当成事实。

## 4. InnoDB 存储引擎

### 4.1 概念

InnoDB 是 MySQL 常用的事务型存储引擎，支持事务、行级锁、外键、崩溃恢复和 MVCC。表数据按主键组织为聚簇索引，二级索引保存索引列和主键值。Buffer Pool 在内存中缓存数据页和索引页，减少磁盘访问。

InnoDB 还使用 redo log、undo log 和双写机制等支持持久性、回滚与崩溃恢复。具体日志结构和刷盘策略受版本与参数影响，不应把某一个流程简化成“写一次磁盘就绝对安全”。

### 4.2 原理或使用方法

聚簇索引的叶子节点存放完整行数据，因此按主键查询通常可以直接取得整行。二级索引叶子节点保存主键值，查询非覆盖列时可能需要回到聚簇索引查找，称为回表。覆盖索引让查询所需列都在索引中，可以减少回表。

InnoDB 以页为基本读写单位，Buffer Pool 命中可避免磁盘 I/O。表数据量和访问模式影响缓存命中率；增加缓存并非无限有效，仍需优化 SQL 和索引。

### 4.3 示例

```sql
CREATE TABLE article (
  id BIGINT UNSIGNED PRIMARY KEY AUTO_INCREMENT,
  author_id BIGINT UNSIGNED NOT NULL,
  title VARCHAR(200) NOT NULL,
  status TINYINT NOT NULL DEFAULT 0,
  created_at DATETIME NOT NULL,
  KEY idx_author_status_time (author_id, status, created_at)
) ENGINE=InnoDB
  DEFAULT CHARSET=utf8mb4
  COLLATE=utf8mb4_0900_ai_ci;

SELECT id, title, created_at
FROM article
WHERE author_id = 10 AND status = 1
ORDER BY created_at DESC
LIMIT 20;
```

该查询可能利用联合索引定位作者和状态，并利用索引中的 `created_at` 顺序减少额外排序。是否真正使用索引以及是否回表，需要查看 `EXPLAIN` 和实际数据分布。

### 4.4 易错点和注意事项

- InnoDB 的主键选择会影响所有二级索引，因为二级索引需要存储主键值；过长主键会增加索引体积。
- 事务未提交会占用 undo、锁和相关资源，长事务可能阻碍 purge 并增加恢复时间。
- 不要在没有测量的情况下随意修改刷盘参数追求性能，这会改变持久性权衡。
- `ENGINE=MyISAM` 不支持事务和行级锁，不应因历史习惯继续用于需要一致性和并发的业务表。

### 4.5 常见问题

**问：为什么 InnoDB 表最好有主键？**  
答：InnoDB 按主键组织聚簇索引。没有显式主键时，系统会选择或生成内部行标识，业务查询和索引设计也更难控制。

**问：什么是回表？**  
答：通过二级索引找到主键后，再回到聚簇索引读取查询所需的其他列。覆盖索引可避免这一步，但会增加索引列和维护成本。

**问：InnoDB 一定比 MyISAM 慢吗？**  
答：不能这样断言。InnoDB 有事务和并发控制开销，但现代版本在多数通用场景更合适，实际性能取决于负载、索引和配置。

## 5. B+ 树索引与联合索引

### 5.1 概念

B+ 树是 MySQL 常用索引结构。非叶子节点保存键和子节点指针，叶子节点按键有序并通过链表连接，适合等值查询和范围扫描。相比二叉树，B+ 树节点可容纳更多键，树高较低，减少磁盘页访问次数。

联合索引是在多个列上建立的索引，列顺序很重要。最左前缀原则表示查询通常需要从联合索引最左列开始连续使用，才能充分利用索引；范围条件之后的后继列用于定位的能力会受限。

### 5.2 原理或使用方法

主键索引是聚簇索引，二级索引存储索引键和主键。联合索引 `(a, b, c)` 可支持按 `a`、`(a,b)`、`(a,b,c)` 的等值或部分范围查询，也常能支持按这些前缀排序。跳过 `a` 直接查 `b` 通常无法有效使用该索引。

索引并非越多越好。每个索引增加写入维护、存储和优化器选择成本。低选择性列单独建索引可能收益有限，应结合查询频率、数据分布和覆盖需求设计。

### 5.3 示例

```sql
CREATE INDEX idx_order_user_status_time
  ON orders (user_id, status, created_at);

-- 能使用联合索引的最左前缀
SELECT id, amount, created_at
FROM orders
WHERE user_id = 100 AND status = 1
ORDER BY created_at DESC
LIMIT 20;

-- 通常不能有效使用上述联合索引
SELECT id, amount
FROM orders
WHERE status = 1;
```

第一条查询从 `user_id` 开始并继续使用 `status`，`created_at` 可用于索引内排序。第二条缺少最左列 `user_id`，通常需要其他访问路径。

### 5.4 易错点和注意事项

- 最左前缀是理解工具，不是绝对规则；优化器会根据统计信息、范围和成本选择全表扫描或其他索引。
- 联合索引的列顺序应结合等值条件、范围条件和排序需求设计，不应只按字段出现顺序排列。
- 对长字符串建前缀索引要评估选择性；前缀过短可能大量重复，过长则索引空间增大。
- 索引列上使用函数、隐式类型转换或前导通配符可能阻止有效索引访问，应改写条件或使用合适函数索引。

### 5.5 常见问题

**问：为什么联合索引要讲最左前缀？**  
答：B+ 树按索引列顺序排序。缺少左侧列时，后续列在全局并非连续有序，难以直接缩小扫描范围。

**问：一个查询能使用多个二级索引吗？**  
答：可以出现索引合并等策略，但并非总是优于单个合适联合索引。应查看执行计划，不要假设索引越多越快。

**问：索引区分度越高越好吗？**  
答：通常高区分度更有利于定位少量行，但最终仍取决于查询模式、覆盖列和写入成本。低区分度列可作为联合索引的一部分。

## 6. 索引失效与执行计划

### 6.1 概念

“索引失效”通常是口语化说法，准确含义是优化器没有选择某个索引，或该索引无法有效缩小扫描范围。索引存在不代表每次查询都会使用；最终访问路径由语法、统计信息、数据分布和成本估算共同决定。

`EXPLAIN` 展示优化器选择的访问方式、候选键、实际使用的键、估算行数和额外操作。MySQL 8.0.18 起可使用 `EXPLAIN ANALYZE` 获取部分实际执行信息，但输出格式和可用性随版本变化。

### 6.2 原理或使用方法

常见导致无法有效使用索引的写法包括：对索引列使用函数、隐式类型转换、`LIKE '%keyword'` 前导通配符、`OR` 连接未索引列、联合索引跳过最左列。范围条件之后再按后续列精确过滤，可能只能在已扫描范围内逐行判断。

优化时先看真实 SQL、表结构、索引、参数和 `EXPLAIN`，再用代表性数据验证。不要仅凭“`type` 不是 `ref` 就加索引”，也不要看到 `Using filesort` 就认为一定很慢，排序数据量小或已在内存中时成本可能可接受。

### 6.3 示例

```sql
-- 假设 idx_created_at(created_at) 存在
-- 不推荐：对索引列做函数运算
SELECT id, title
FROM article
WHERE DATE(created_at) = '2026-11-11';

-- 可改写为范围条件，便于利用索引
SELECT id, title
FROM article
WHERE created_at >= '2026-11-11 00:00:00'
  AND created_at <  '2026-11-12 00:00:00';

EXPLAIN
SELECT id, title
FROM article
WHERE author_id = 10 AND status = 1
ORDER BY created_at DESC
LIMIT 20;
```

查看 `EXPLAIN` 时重点关注 `key`（实际使用的索引）、`rows`（估算扫描行数）、`filtered`（估算过滤比例）和 `Extra`（排序、临时表、覆盖索引等信息）。估算值不是实际执行次数，需要结合数据分布判断。

### 6.4 易错点和注意事项

- 隐式类型转换可能使索引列被当作另一类型处理，例如字符串列与数字比较；应让参数类型与列类型一致。
- `OR` 条件中部分列无索引时，优化器可能选择全表扫描；可评估改写为 `UNION ALL` 或补足索引，但要以执行计划为准。
- 统计信息过期会让优化器估算偏差，必要时可更新统计信息，但不应把 `ANALYZE TABLE` 当作日常无成本操作。
- 强制索引 `FORCE INDEX` 可能暂时绕过错误选择，但数据分布变化后会带来新风险，应谨慎并保留回退方案。

### 6.5 常见问题

**问：给 `WHERE` 中每个列都建索引就会快吗？**  
答：不会。索引增加写入和空间成本，优化器通常只选择一个较合适路径，过多索引还可能干扰选择。应按查询组合设计。

**问：`LIKE 'abc%'` 能用索引吗？**  
答：通常前导固定时可能利用索引做范围扫描；`LIKE '%abc'` 缺少确定前缀，普通 B+ 树索引难以有效定位。

**问：`EXPLAIN` 的 `rows` 为什么和实际差很多？**  
答：它是基于统计信息的估算，受数据分布、相关性和采样影响。可用 `EXPLAIN ANALYZE` 或实际压测核对。

## 7. 事务、ACID 与隔离级别

### 7.1 概念

事务是一组应作为整体执行的数据库操作，常用 `START TRANSACTION` 开始，`COMMIT` 提交，`ROLLBACK` 回滚。ACID 分别表示原子性、一致性、隔离性和持久性：事务要么全部生效要么全部撤销；约束和业务规则保持一致；并发事务之间按隔离级别相互影响；提交后的结果在故障恢复后仍应保留。

并发异常包括脏读（读到未提交数据）、不可重复读（同一事务两次读取同一行结果不同）和幻读（同一范围查询出现不同行集合）。隔离级别越高通常并发开销越大，需要在一致性和吞吐量之间权衡。

### 7.2 原理或使用方法

SQL 标准的隔离级别包括 `READ UNCOMMITTED`、`READ COMMITTED`、`REPEATABLE READ` 和 `SERIALIZABLE`。InnoDB 默认通常是 `REPEATABLE READ`，并使用 MVCC 和锁实现相应语义；这不等于所有数据库的默认值相同。

一致性还依赖应用正确划分事务边界。事务不应包含等待用户输入、远程调用等长时间操作；更新多行时应使用稳定顺序减少死锁。仅靠数据库隔离级别不能保证跨服务业务的一致性。

### 7.3 示例

```sql
START TRANSACTION;

UPDATE account
SET balance = balance - 100
WHERE id = 1 AND balance >= 100;

UPDATE account
SET balance = balance + 100
WHERE id = 2;

COMMIT;
```

示例假设余额列使用合适的数值类型并有非负业务约束。实际转账还应检查第一条更新影响行数，若为 0 应回滚，防止在余额不足时仍执行后续操作。

```sql
SET SESSION TRANSACTION ISOLATION LEVEL READ COMMITTED;
START TRANSACTION;
SELECT balance FROM account WHERE id = 1;
COMMIT;
```

### 7.4 易错点和注意事项

- 自动提交模式下，单条语句可能立即提交；需要多条语句原子执行时必须显式开启事务。
- 不要用“先查询再更新”替代原子条件更新。并发下两个事务可能都读到旧值，应使用条件更新、行锁或合适隔离策略。
- 隔离级别降低可能提高并发，但会引入更多可见性差异；修改前要确认业务是否依赖可重复读语义。
- 事务提交成功不代表备份已完成，也不代表外部消息已发送；跨系统一致性需补偿或事务消息等机制。

### 7.5 常见问题

**问：InnoDB 默认隔离级别是什么？**  
答：常见 MySQL 版本中 InnoDB 默认为 `REPEATABLE READ`，但部署配置可能修改，应以实际参数和版本文档为准。

**问：`REPEATABLE READ` 能完全避免幻读吗？**  
答：不能简单说完全避免。InnoDB 通过 MVCC 和临键锁等机制处理不同读场景；快照读与当前读行为不同，需结合具体语句分析。

**问：事务越长越好还是越短越好？**  
答：通常应尽量短，减少锁持有、undo 增长和阻塞。是否拆分还要保证业务原子性，不能为了缩短事务破坏一致性。

## 8. MVCC 多版本并发控制

### 8.1 概念

MVCC（多版本并发控制）让读操作通常不必等待写操作，通过保留行的历史版本提供一致性快照。InnoDB 在行中维护事务相关信息，并把旧版本放入 undo 日志；读取时根据可见性规则判断哪个版本对当前事务可见。

快照读是普通 `SELECT` 在一致性读场景下读取某个快照；当前读会读取最新可见版本并可能加锁，如 `SELECT ... FOR UPDATE`、`UPDATE`、`DELETE`。两种读的语义不同。

### 8.2 原理或使用方法

在 `REPEATABLE READ` 下，事务第一次一致性读会建立 ReadView，之后的普通查询通常复用该视图，因此能看到稳定快照。在 `READ COMMITTED` 下，每条语句可能创建新视图，所以同一事务两次查询可能看到不同已提交结果。

MVCC 降低读写冲突，但不是无锁并发。更新同一行、当前读和唯一性检查仍需要锁。长事务会让旧版本无法及时清理，占用 undo 空间并增加 purge 压力。

### 8.3 示例

```sql
-- 会话 A
START TRANSACTION;
SELECT balance FROM account WHERE id = 1; -- 假设读到 500

-- 会话 B 此时执行并提交
UPDATE account SET balance = 600 WHERE id = 1;
COMMIT;

-- 会话 A 再次普通查询，在 REPEATABLE READ 下通常仍读到 500
SELECT balance FROM account WHERE id = 1;

-- 当前读会读取最新已提交版本（并加锁）
SELECT balance FROM account WHERE id = 1 FOR UPDATE;
ROLLBACK;
```

实际结果取决于隔离级别、事务开始和首次读取时机、是否已有锁等待以及版本实现。示例用于区分快照读和当前读，不应作为所有配置下的固定输出。

### 8.4 易错点和注意事项

- 不要用 MVCC 快照读到的旧值做“检查后更新”，并发更新应使用原子条件和当前读。
- 长事务不仅占用连接，还会使 undo 无法及时回收；应监控事务持续时间。
- MVCC 主要服务于读一致性，不保证跨多个事务或外部系统的全局一致性。
- `SELECT ... FOR UPDATE` 会加锁，若使用不当会增加锁等待和死锁风险，不应为“更安全”而无条件使用。

### 8.5 常见问题

**问：MVCC 是否意味着读永远不阻塞？**  
答：普通一致性读通常不需要等待写锁，但当前读、锁冲突和 DDL 等场景仍可能阻塞。

**问：为什么同一事务两次查询结果不同？**  
答：可能处于 `READ COMMITTED`，也可能第二次是当前读，或隔离级别/连接配置与预期不同。应确认语句类型和会话参数。

**问：undo 日志越大越好吗？**  
答：不是。undo 支持回滚和 MVCC，但持续增长会占用空间并增加清理压力，长期未提交事务是常见原因。

## 9. 锁与死锁

### 9.1 概念

InnoDB 支持行级锁，包括共享锁和排他锁，并通过意向锁协调表级和行级锁。根据索引访问范围，还可能出现记录锁、间隙锁和临键锁。间隙锁用于阻止范围内插入，主要在特定隔离级别和访问路径下出现。

死锁是两个或多个事务互相等待对方持有的锁，形成循环等待。InnoDB 能检测部分死锁并回滚其中一个事务；锁等待超时则是等待超过配置时限。死锁不等同于普通锁等待。

### 9.2 原理或使用方法

更新或删除通常会获取排他锁，普通一致性读在 MVCC 下可能不加行锁，`SELECT ... FOR UPDATE` 获取排他当前读锁，`SELECT ... FOR SHARE` 获取共享当前读锁。锁的对象与索引访问路径相关；没有合适索引时，扫描范围扩大可能锁定更多记录或间隙。

降低死锁的常见方法：事务尽量短、按固定顺序访问多行、减少无索引更新、避免在事务中等待外部资源、对可重试死锁设置有限重试。捕获死锁后应重新执行整个事务，而不是只重试失败语句。

### 9.3 示例

```sql
-- 会话 A
START TRANSACTION;
UPDATE account SET balance = balance - 10 WHERE id = 1;
-- 暂不提交

-- 会话 B
START TRANSACTION;
UPDATE account SET balance = balance - 10 WHERE id = 2;
-- 暂不提交

-- 会话 A
UPDATE account SET balance = balance + 10 WHERE id = 2;

-- 会话 B
UPDATE account SET balance = balance + 10 WHERE id = 1;
-- 两个事务可能形成循环等待，其中一个被回滚
```

实际是否死锁取决于执行时机和锁范围。应用中应按相同顺序更新账户，例如总是先锁较小 ID，缩短持锁时间。

### 9.4 易错点和注意事项

- 没有索引的 `UPDATE` 可能扫描大量行并持有更多锁，既慢又更容易阻塞其他事务。
- 死锁错误不是简单“加锁失败”，应用应识别可重试错误并回滚后重做完整事务。
- 不要盲目提高 `innodb_lock_wait_timeout` 掩盖问题，等待更久可能只是延长故障。
- 间隙锁行为与隔离级别、索引和查询范围有关，不能把所有行锁都理解成只锁住已存在记录。

### 9.5 常见问题

**问：死锁一定是代码写错了吗？**  
答：不一定。并发访问顺序交错可能产生死锁，但设计可以通过统一顺序、缩短事务和重试降低概率。

**问：如何查看 InnoDB 死锁信息？**  
答：可查看 `SHOW ENGINE INNODB STATUS` 的最近死锁部分；MySQL 8.0 还可结合 `performance_schema.data_locks` 等表观察锁等待。

**问：共享锁和排他锁有什么区别？**  
答：共享锁之间通常可共存，排他锁与其他共享或排他锁冲突。具体兼容性还受锁类型和访问路径影响。

## 10. 慢查询与 SQL 优化

### 10.1 概念

慢查询是执行时间超过阈值的 SQL。定位慢查询需要开启慢查询日志或使用性能视图，收集 SQL 文本、执行时间、扫描行数、锁等待等上下文。单次慢不一定代表持续问题，应关注频率、数据量和并发影响。

SQL 优化通常从减少扫描行数、改善索引访问、避免不必要排序和临时表、降低返回数据量入手。数据库参数和硬件也影响性能，但不应在未定位 SQL 前盲目调参。

### 10.2 原理或使用方法

慢查询日志相关参数包括 `slow_query_log`、`long_query_time` 和 `log_queries_not_using_indexes` 等；生产环境开启前要评估日志量和隐私。`EXPLAIN` 看访问计划，`EXPLAIN ANALYZE` 在支持的版本中看实际执行时间和行数，但分析本身也有开销。

优化步骤可概括为：确认慢 SQL 与调用频率，查看执行计划和表结构，检查数据分布与索引，改写 SQL 或补索引，在接近生产的数据量下验证，再观察线上指标。只测空表或极小数据集没有代表性。

### 10.3 示例

```sql
-- 查看当前会话的慢查询相关配置
SHOW VARIABLES LIKE 'slow_query_log';
SHOW VARIABLES LIKE 'long_query_time';

-- 分析查询计划（MySQL 8.0.18+ 可用 EXPLAIN ANALYZE）
EXPLAIN ANALYZE
SELECT s.name, c.title
FROM student AS s
JOIN enrollment AS e ON e.student_id = s.id
JOIN course AS c ON c.id = e.course_id
WHERE c.status = 'published'
ORDER BY s.name
LIMIT 50;
```

若 `enrollment` 的连接列有合适索引、`course.status` 选择性较高，优化器可能选择不同连接顺序。真实优化应结合执行计划和数据分布，而不是只给每条 SQL 加索引。

### 10.4 易错点和注意事项

- 只优化单条 SQL 可能把压力转移到其他查询；索引增加也会影响写入，应整体评估。
- `SELECT *` 会增加网络传输和覆盖索引失败概率；明确返回所需列通常更可控。
- `ORDER BY`、`GROUP BY` 和 `DISTINCT` 可能引入排序或临时表，数据量大时要看 `Extra` 和执行统计。
- 慢查询日志中的 SQL 可能含业务数据，访问权限和脱敏策略要同步考虑。

### 10.5 常见问题

**问：给查询加 `LIMIT` 就一定快吗？**  
答：不一定。若排序或过滤需要扫描大量数据，`LIMIT` 只减少最终返回，不一定减少前期工作；深分页尤其明显。

**问：`EXPLAIN` 显示使用索引就一定高效吗？**  
答：不一定。可能扫描大量索引行或频繁回表。要结合 `rows`、过滤比例和实际执行时间判断。

**问：优化 SQL 和增加硬件哪个优先？**  
答：通常先定位并优化明显低效的 SQL 和索引，再评估硬件。硬件可以提升上限，但不能修复错误的数据访问方式。

## 11. 常见设计问题与优化思路

### 11.1 概念

数据库设计需要同时考虑数据一致性、查询模式、写入频率、增长速度和运维成本。规范化减少冗余和更新异常，反规范化可减少复杂连接或重复计算，但会增加一致性维护成本。

常见设计问题包括：字段类型不合理、缺少主键、索引与查询不匹配、把多值塞进逗号字符串、过度使用大字段、事务边界过长、分页方式和统计查询设计不当。

### 11.2 原理或使用方法

设计表时先明确实体、唯一标识、关系基数和访问路径。一对多通常通过子表外键表达；多对多使用关联表并设置联合唯一约束。需要在数据库中强制的不变量应通过约束、事务或原子更新实现，不能只依赖界面校验。

大表优化可从索引、查询改写、归档历史数据、读写分离、分区或分片逐步考虑。分区和分片会增加路由、跨片查询和运维复杂度，不应作为第一选择。缓存可降低热点读压力，但要处理一致性和失效。

### 11.3 示例

不推荐把多个标签存成逗号字符串：

```sql
-- 不推荐：难以建立约束，查询和更新都依赖字符串处理
CREATE TABLE article_bad (
  id BIGINT UNSIGNED PRIMARY KEY,
  tags VARCHAR(500)
);
```

推荐使用关联表：

```sql
CREATE TABLE tag (
  id BIGINT UNSIGNED PRIMARY KEY AUTO_INCREMENT,
  name VARCHAR(32) NOT NULL UNIQUE
);

CREATE TABLE article_tag (
  article_id BIGINT UNSIGNED NOT NULL,
  tag_id BIGINT UNSIGNED NOT NULL,
  PRIMARY KEY (article_id, tag_id),
  KEY idx_tag_article (tag_id, article_id),
  CONSTRAINT fk_article_tag_article
    FOREIGN KEY (article_id) REFERENCES article(id),
  CONSTRAINT fk_article_tag_tag
    FOREIGN KEY (tag_id) REFERENCES tag(id)
);
```

关联表便于按标签反查文章、保证引用完整性，也能利用索引；代价是连接查询和额外写入。若读多写少且有明确测量收益，才考虑受控冗余。

### 11.4 易错点和注意事项

- 不要为每个“可能查询”的列建索引。索引要对应真实高频查询，并计算写入和空间成本。
- 大字段和 JSON 不是垃圾箱。需要频繁过滤、排序或约束的数据应结构化；JSON 适合结构灵活但访问模式明确的附加信息。
- 分库分表会引入分布式事务、跨片分页和全局唯一 ID 问题，必须基于容量和增长预测，而不是追潮流。
- 统计类查询直接扫大表可能影响线上业务，可使用预聚合、离线汇总或读副本，但要明确延迟和一致性要求。

### 11.5 常见问题

**问：数据库设计一定要第三范式吗？**  
答：不一定。规范化是减少冗余和更新异常的重要基线，但高频查询可在明确一致性与维护成本后适度反规范化。

**问：什么时候考虑分库分表？**  
答：单库单表经过索引、归档、读写分离等优化仍无法满足容量或吞吐目标，且有清晰分片键和运维能力时再考虑。

**问：缓存能解决数据库性能问题吗？**  
答：能缓解部分热点读，但不能修复错误 SQL、缺失索引或写放大。还要处理缓存一致性、穿透和失效策略。
