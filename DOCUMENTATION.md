# 📚 商品销售管理系统 - 完整文档

## 📋 目录
1. [项目概述](#项目概述)
2. [系统架构](#系统架构)
3. [技术栈详解](#技术栈详解)
4. [功能模块](#功能模块)
5. [数据库设计](#数据库设计)
6. [API接口](#api接口)
7. [界面美化](#界面美化)
8. [部署指南](#部署指南)
9. [项目总结](#项目总结)

---

## 项目概述

本项目是一个基于 Python Flask + MySQL + JavaScript 技术栈开发的轻量级商品销售管理系统，专为中小商户设计。系统集成了订单管理、销售统计、智能问数、销量预测等核心功能，提供了完整的销售管理解决方案。

### ✨ 系统特色
- 🛒 **完整订单管理** - 订单创建、状态更新、库存自动扣减
- 📊 **多维度统计** - 时间维度、商品种类维度统计，支持数据导出
- 🤖 **智能问数** - 自然语言查询销售数据，支持关键词解析
- 🔮 **销量预测** - 基于历史数据的智能预测，提供需求等级评估
- 👥 **用户管理** - 多角色权限控制，管理员和普通用户分离
- 📝 **留言系统** - 客户留言提交和回复管理
- 🏷️ **商品分类** - 支持层级分类，便于商品组织

---

## 系统架构

### 架构图
```
┌─────────────────────────────────────────────────────────────┐
│                    前端层 (Frontend)                        │
├─────────────────────────────────────────────────────────────┤
│  HTML5 + CSS3 + JavaScript + Bootstrap 5 + Chart.js       │
│  ┌─────────────┬─────────────┬─────────────┬─────────────┐ │
│  │   登录页    │   仪表板    │   订单管理  │   销售统计  │ │
│  ├─────────────┼─────────────┼─────────────┼─────────────┤ │
│  │   用户管理  │   商品管理  │   智能问数  │   销量预测  │ │
│  └─────────────┴─────────────┴─────────────┴─────────────┘ │
└─────────────────────────────────────────────────────────────┘
                                │
                                │ HTTP/HTTPS
                                ▼
┌─────────────────────────────────────────────────────────────┐
│                    API层 (RESTful API)                      │
├─────────────────────────────────────────────────────────────┤
│  Flask 2.3.3 + Flask-CORS                                  │
│  ┌─────────────┬─────────────┬─────────────┬─────────────┐ │
│  │ 用户管理API │ 商品管理API │ 订单管理API │ 统计查询API │ │
│  ├─────────────┼─────────────┼─────────────┼─────────────┤ │
│  │ 智能问数API │ 销量预测API │ 留言管理API │ 文件上传API │ │
│  └─────────────┴─────────────┴─────────────┴─────────────┘ │
└─────────────────────────────────────────────────────────────┘
                                │
                                │ PyMySQL
                                ▼
┌─────────────────────────────────────────────────────────────┐
│                    数据层 (Database)                        │
├─────────────────────────────────────────────────────────────┤
│  MySQL 8.0                                                 │
│  ┌─────────────┬─────────────┬─────────────┬─────────────┐ │
│  │   用户表    │   商品表    │   订单表    │   留言表    │ │
│  ├─────────────┼─────────────┼─────────────┼─────────────┤ │
│  │   种类表    │ 订单详情表  │ 查询日志表  │ 预测结果表  │ │
│  └─────────────┴─────────────┴─────────────┴─────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

### 技术栈详解

#### 前端技术
- **HTML5**: 语义化标签，提供良好的页面结构
- **CSS3**: 响应式设计，支持多种设备
- **JavaScript**: 原生JS，无框架依赖，轻量级
- **Bootstrap 5**: UI组件库，提供美观的界面
- **Chart.js**: 图表库，用于数据可视化

#### 后端技术
- **Python 3.7+**: 主要开发语言
- **Flask 2.3.3**: 轻量级Web框架
- **PyMySQL**: MySQL数据库连接器
- **Flask-CORS**: 跨域请求支持
- **SiliconFlow API**: AI服务提供商，提供强大的自然语言处理能力
- **requests**: HTTP客户端，用于调用AI API
- **pandas**: 数据处理和分析
- **numpy**: 数值计算，用于预测算法

#### 数据库技术
- **MySQL 8.0**: 关系型数据库
- **InnoDB**: 存储引擎，支持事务
- **UTF8MB4**: 字符集，支持emoji等特殊字符

---

## 功能模块

### 1. 用户管理模块
- **登录验证**: MD5密码加密，Session状态管理
- **权限控制**: 管理员/普通用户角色分离
- **用户CRUD**: 完整的用户增删改查功能
- **状态管理**: 用户启用/禁用状态控制

### 2. 商品管理模块
- **商品信息**: 名称、价格、库存、描述管理
- **分类关联**: 与商品种类表关联
- **状态控制**: 上架/下架状态管理
- **库存管理**: 实时库存数量跟踪

### 3. 订单管理模块
- **订单创建**: 支持多商品订单创建
- **状态管理**: 待发货/已发货/已完成/已取消
- **库存扣减**: 下单时自动扣减商品库存
- **订单详情**: 完整的订单信息查看
- **唯一订单号**: 时间戳+随机数生成

### 4. 销售统计模块
- **多维度统计**: 时间维度、商品种类维度
- **图表可视化**: Chart.js实现销售趋势图、饼图
- **数据导出**: CSV格式数据导出功能
- **热销排行**: 商品销量排行榜
- **实时数据**: 从数据库实时查询统计

### 5. AI智能问数模块
- **AI服务集成**: 集成SiliconFlow API，提供强大的AI能力
- **自然语言解析**: 支持中文自然语言查询，AI自动理解用户意图
- **智能类型识别**: 自动识别查询类型（销售额/销量/订单数/客户数/商品种类）
- **实时数据查询**: 直接查询数据库获取实际数据，确保结果准确性
- **AI解释生成**: 基于查询结果生成专业的自然语言解释和分析
- **查询示例**: 提供常见问题示例，帮助用户快速上手
- **结果展示**: 结构化的查询结果展示，包含AI增强信息

### 6. 销量预测模块
- **历史数据分析**: 基于近6个月销售数据
- **移动平均算法**: 简单有效的时间序列预测
- **需求等级评估**: 高/中/低需求等级划分
- **预测可视化**: 图表展示预测结果
- **历史记录**: 预测结果保存和查看

### 7. 留言管理模块
- **客户留言**: 客户可提交留言和反馈
- **管理员回复**: 管理员可回复客户留言
- **状态管理**: 未回复/已回复状态跟踪
- **内容过滤**: 防止SQL注入和XSS攻击
- **统计展示**: 留言数量统计

---

## AI模块详解

### 📋 概述

本项目的AI模块集成了SiliconFlow API，提供强大的自然语言查询功能。用户可以使用自然语言查询销售数据，AI会自动解析查询意图，查询数据库获取实际数据，并生成专业的解释。

### 🏗️ 模块架构

#### 文件结构
```
├── ai_service.py          # AI服务核心模块
├── ai_module.py           # AI功能接口模块
└── templates/ai_query.html # 智能问数前端页面
```

#### 核心组件

##### 1. LLMClient类 (ai_service.py)
```python
class LLMClient:
    """通用LLM客户端，支持重试与严格JSON输出清洗"""
    
    def __init__(self, use_siliconflow: bool = True):
        # 直接使用SiliconFlow配置
        self.base_url = "https://api.siliconflow.cn/v1/chat/completions"
        self.api_key = "sk-vfxewwbfcgthjjabvksdrezxdvtwqjmpdglubfthmzinlren"
        self.model = "moonshotai/Kimi-K2-Instruct-0905"
        self.timeout = 60
        self.retry_count = 3
    
    def chat(self, messages, max_tokens=1024, temperature=0.3):
        """调用LLM API，支持重试机制"""
        # 实现带重试的API调用
    
    @staticmethod
    def extract_json_block(text):
        """从json代码块中提取JSON内容"""
        # 从文本中提取JSON块
```

##### 2. AIService类 (ai_service.py)
```python
class AIService:
    """封装智能问数与销量预测功能"""
    
    def __init__(self):
        self.client = LLMClient(use_siliconflow=True)
    
    def analyze_query(self, query_text):
        """分析用户查询，提取时间范围、商品种类和查询指标"""
        # 使用正则表达式匹配时间、指标等
    
    def build_query_sql(self, parsed_query, categories):
        """构建SQL查询语句"""
        # 根据解析结果构建安全的SQL查询
    
    def process_query_with_llm(self, query_text, csv_path, metric):
        """使用LLM处理查询并生成自然语言解释"""
        # 调用LLM生成专业的数据分析报告
```

##### 3. 智能问数接口 (ai_module.py)
```python
def intelligent_query(query_text: str, user_id: int):
    """处理智能问数请求"""
    # 1. 分析用户查询
    parsed_query = sales_ai_service.analyze_query(query_text)
    
    # 2. 获取数据库连接和商品种类
    categories = sales_ai_service.get_categories_from_db(cursor)
    
    # 3. 构建SQL并执行查询
    sql, params, metric = sales_ai_service.build_query_sql(parsed_query, categories)
    
    # 4. 导出数据到CSV
    csv_path = sales_ai_service.export_query_data_to_csv(...)
    
    # 5. 使用LLM处理查询
    llm_result = sales_ai_service.process_query_with_llm(...)
    
    # 6. 写入查询日志
    save_query_log(user_id, query_text, llm_result["data"])
    
    return result
```

### 🚀 快速开始

#### 1. 配置API密钥
```python
# 在 ai_service.py 中配置
SILICONFLOW_API_KEY = "your-api-key-here"
```

#### 2. 配置数据库
```bash
# 导入数据库结构
mysql -u root -p product_sales_system < database/schema.sql

# （可选）导入示例数据
mysql -u root -p product_sales_system < database/sample_data.sql
```

#### 3. 启动服务
```bash
python app.py
```

#### 4. 访问智能问数页面
打开浏览器访问：`http://localhost:5000/ai-query`

### 💡 使用示例

#### 支持的查询类型

##### 1. 销售额查询
```
用户输入: "销售额是多少"
AI识别: 销售额
数据库查询: SELECT SUM(total_amount) FROM orders WHERE status != 'cancelled'
返回结果: ¥123,456.78
AI解释: "根据您的查询，系统显示当前总销售额为¥123,456.78..."
```

##### 2. 销量查询
```
用户输入: "销量是多少"
AI识别: 销量
数据库查询: SELECT SUM(od.quantity) FROM order_detail od JOIN orders o ON od.order_id = o.order_id WHERE o.status != 'cancelled'
返回结果: 1,234
AI解释: "当前总销量为1,234件商品..."
```

##### 3. 订单数查询
```
用户输入: "有多少个订单"
AI识别: 订单数
数据库查询: SELECT COUNT(*) FROM orders WHERE status != 'cancelled'
返回结果: 56
AI解释: "系统显示当前有效订单总数为56个..."
```

##### 4. 客户数查询
```
用户输入: "有多少客户"
AI识别: 客户数
数据库查询: SELECT COUNT(DISTINCT user_id) FROM orders WHERE status != 'cancelled'
返回结果: 23
AI解释: "当前共有23位客户下单..."
```

##### 5. 商品种类查询
```
用户输入: "有哪些商品种类"
AI识别: 商品种类
数据库查询: SELECT category_name FROM category WHERE status = 1 LIMIT 10
返回结果: 手机, 电脑, 电子产品
AI解释: "系统显示当前有以下商品种类：手机、电脑、电子产品..."
```

### 🔧 API接口

#### 智能问数接口
```http
POST /api/ai/query
Content-Type: application/json

{
    "query_text": "销售额是多少"
}
```

#### 响应格式
```json
{
    "code": 1,
    "msg": "查询成功",
    "data": {
        "query_text": "销售额是多少",
        "info_type": "销售额",
        "data_value": "123456.78",
        "explanation": "AI生成的解释...",
        "ai_enhanced": true
    }
}
```

### ⚙️ 配置说明

#### SiliconFlow API配置
网址：https://www.siliconflow.cn/
```python
# ai_service.py 中的 LLMClient 类，直接集成配置
class LLMClient:
    def __init__(self, use_siliconflow: bool = True):
        if use_siliconflow:
            self.base_url = "https://api.siliconflow.cn/v1/chat/completions"
            self.api_key = "sk-vfxewwbfcgthjjabvksdrezxdvtwqjmpdglubfthmzinlren"  # 已配置
            self.model = "moonshotai/Kimi-K2-Instruct-0905"  # 当前使用模型
            self.timeout = 60
            self.retry_count = 3

# 支持的模型(可在SiliconFlow平台切换)
模型选择：
- "moonshotai/Kimi-K2-Instruct-0905"  # 当前使用，中文支持好
- "Qwen/QwQ-32B"                      # 推荐模型，性能最佳
- "Qwen/QwQ-14B"                      # 备选模型，平衡性能
- "Qwen/Qwen2.5-Coder-32B-Instruct"   # 编程专用模型
```

#### 无需环境变量配置
根据项目记忆，API配置已直接集成在ai_service.py中，无需额外的环境变量配置。

### 💪 增强功能

#### 智能时间识别
系统支持多种时间表达形式：
- **相对时间**: 今天、昨天、最近N天
- **绝对时间**: 本月、今年、具体年月
- **智能匹配**: 使用正则表达式自动提取时间条件

#### 商品种类智能匹配
```python
def match_category_by_query(self, query_text, categories):
    """根据查询文本匹配商品种类"""
    query_lower = query_text.lower()
    for category in categories:
        if category['category_name'].lower() in query_lower:
            return category
    return None
```

#### 安全SQL构建
系统使用参数化查询防止SQL注入：
```python
# 构建安全的WHERE子句
where_clauses = ["o.status IN ('shipped', 'completed')"]
params = []

if time_type == 'recent':
    days = time_data.get('days', 7)
    where_clauses.append("o.create_time >= DATE_SUB(NOW(), INTERVAL %s DAY)")
    params.append(days)

if category_id:
    where_clauses.append("g.category_id = %s")
    params.append(category_id)
```

#### 错误处理增强
根据经验记忆，系统增强了对None值的安全处理：
```python
# 安全的字段检查
time_data = parsed_query.get('time')
metric = parsed_query.get('metric')

if not time_data:
    return {"code": 0, "msg": "请补充时间范围", "data": {}}
elif not metric:
    return {"code": 0, "msg": "请明确查询指标", "data": {}}

# 安全的对象访问
category_data = parsed_query.get('category')
category_id = category_data.get('category_id') if category_data else None
```

### 🛠️ 自定义配置

#### 1. 添加新的查询类型
在 ai_service.py 的 analyze_query 方法中修改 metric_patterns：
```python
metric_patterns = {
    'sales': [r'销售额|销售金额|营业额|营收|收入'],
    'quantity': [r'销量|销售量|数量|件数|台数'],
    'orders': [r'订单数|订单量|订单|单数'],
    'customers': [r'客户数|用户数|客户|用户|买家数'],
    # 添加新的查询类型
    'avg_order': [r'平均订单|平均金额']
}
```

同时在 build_query_sql 方法中添加相应的SQL语句：
```python
elif metric == 'avg_order':
    select_clause = "AVG(o.total_amount) AS value"
    from_clause = "FROM orders o"
```

#### 2. 自定义AI提示词
在 generate_query_prompt 方法中修改提示词：
```python
prompt = f"""你是专业的销售数据分析师。请基于用户问题和CSV数据，生成精准的回答。

用户问题：{query_text}

CSV数据内容：
{csv_content}

请按照以下要求输出JSON格式：
{json_format}

请严格按照JSON格式输出，不要包含其他无关内容。"""
```

#### 3. 自定义解释模板
```python
# 在 generate_query_explanation 方法中修改提示词
system_prompt = """你是专业的销售数据分析师。请基于用户问题和查询结果，生成简短的解释。
要求：
1. 数据总结：点明核心数据
2. 业务分析：解释含义
3. 专业建议：简短建议
4. 自定义要求：添加您的特殊要求
字数在80~120字之间。"""
```

### 📊 销量预测功能

#### 预测流程
1. **导出历史数据**: 从数据库获取近6个月的销售数据
2. **数据预处理**: 按商品种类聚合，计算数据完整性
3. **AI分析**: 使用LLM分析趋势和周期规律
4. **生成预测**: 输出未来1个月各种类的销量预测
5. **增强处理**: 基于业务逻辑修正预测结果

#### 关键特性
```python
def sales_prediction():
    """执行销量预测"""
    # 1. 导出历史销售数据
    historical_data_path = sales_ai_service.export_historical_sales_data(cursor)
    
    # 2. 使用LLM处理预测
    llm_result = sales_ai_service.process_prediction_with_llm(historical_data_path)
    
    # 3. 增强预测结果
    enhanced_predictions = sales_ai_service.enhance_predictions(predictions, base_sales_dict)
    
    # 4. 保存预测结果
    save_prediction_results(enhanced_predictions)
```

#### 预测结果结构
```json
{
  "predictions": [
    {
      "category_id": 1,
      "category_name": "手机",
      "predicted_sales": 150,
      "demand_level": "high",
      "growth_rate": 12.5,
      "confidence": 0.85,
      "ai_enhanced": true
    }
  ]
}
```

#### 需求等级划分
- **high**: 预测值高于历史平均值
- **medium**: 预测值接近历史平均值  
- **low**: 预测值低于历史平均值

#### 置信度计算
```python
# 根据数据完整性计算置信度
if data_completeness >= 90:
    confidence = 0.9
else:
    confidence = max(0.5, 0.9 - (90 - data_completeness) * 0.01)
```

#### 常见问题

##### 1. API调用失败
```
错误: API调用失败: 401 - Unauthorized
解决: 检查API密钥是否正确配置
```

##### 2. 查询解析失败
```
错误: AI未识别查询类型
解决: 检查查询文本是否包含支持的关键词
```

##### 3. 数据库连接失败
```
错误: 数据库查询失败
解决: 检查数据库连接配置和表结构
```

#### 调试模式
```python
# 在 ai_service.py 中启用调试日志
import logging
logging.basicConfig(level=logging.DEBUG)

# 在关键位置添加日志
print(f"AI响应: {ai_response}")
print(f"解析结果: {parsed_result}")
print(f"数据库查询结果: {data_value}")
```

### 📊 性能优化

#### 1. API调用优化
- 设置合理的超时时间（60秒）
- 实现请求重试机制
- 缓存常用查询结果

#### 2. 数据库优化
- 为常用查询字段添加索引
- 限制查询结果数量
- 使用连接池管理数据库连接

#### 3. 前端优化
- 实现查询防抖，避免频繁请求
- 显示加载状态，提升用户体验
- 优化结果展示，提高可读性

### 🔒 安全考虑

#### 1. API密钥安全
- ✅ **直接集成**: 根据项目规范，API配置直接集成在ai_service.py中
- ✅ **客户端隐藏**: API密钥不暴露给前端
- ⚠️ **定期更新**: 建议定期更新SiliconFlow API密钥

#### 2. 输入验证
```python
# 查询内容验证
if not query_text or len(query_text.strip()) == 0:
    return {"code": 0, "msg": "查询内容不能为空"}

if len(query_text) > 500:  # 限制查询长度
    return {"code": 0, "msg": "查询内容过长"}

# SQL注入防护——使用参数化查询
cursor.execute(sql, params)  # 不直接拼接SQL字符串
```

#### 3. 错误处理
```python
# 安全的错误消息处理
def safe_format_error(error_msg: str) -> str:
    """安全地格式化错误消息，避免包含花括号的内容导致格式化错误"""
    return str(error_msg).replace('{', '{{').replace('}', '}}')

# 不暴露敏感错误信息
except Exception as e:
    return {"code": 0, "msg": f"处理查询失败: {safe_format_error(str(e))}", "data": {}}
```

### 📈 监控和日志

#### 1. 查询日志
```python
# 在 ai_module.py 中记录查询日志
cursor.execute(
    "INSERT INTO query_log (user_id, query_text, query_result) VALUES (%s, %s, %s)",
    (session.get('user_id'), query_text, str(ai_result))
)
```

#### 2. 性能监控
```python
import time

start_time = time.time()
result = ai_service.enhance_query_parsing(query_text)
end_time = time.time()

print(f"查询耗时: {end_time - start_time:.2f}秒")
```

#### 3. 错误监控
```python
try:
    result = ai_service.enhance_query_parsing(query_text)
except Exception as e:
    # 记录错误日志
    logging.error(f"AI查询失败: {str(e)}")
    # 发送告警通知
    send_alert(f"AI服务异常: {str(e)}")
```

### 🚀 扩展功能

#### 1. 多语言支持
```python
# 支持英文查询
ENGLISH_KEYWORDS = {
    "sales": "销售额",
    "quantity": "销量",
    "orders": "订单数",
    "customers": "客户数"
}
```

#### 2. 时间范围查询
```python
# 支持时间范围查询
def parse_time_range(query_text):
    if "今天" in query_text:
        return "DATE(create_time) = CURDATE()"
    elif "昨天" in query_text:
        return "DATE(create_time) = DATE_SUB(CURDATE(), INTERVAL 1 DAY)"
    # 更多时间范围...
```

#### 3. 商品种类过滤
```python
# 支持特定商品种类查询
def parse_category_filter(query_text):
    categories = get_all_categories()
    for category in categories:
        if category['category_name'] in query_text:
            return f"AND g.category_id = {category['category_id']}"
    return ""
```

### 📚 参考资料

- [SiliconFlow API文档](https://siliconflow.cn/docs)
- [Flask官方文档](https://flask.palletsprojects.com/)
- [PyMySQL文档](https://pymysql.readthedocs.io/)
- [MySQL官方文档](https://dev.mysql.com/doc/)

---

## 数据库设计

### 核心表结构
```sql
-- 用户表
user (user_id, username, password, role, real_name, phone, email, status, create_time)

-- 商品种类表
category (category_id, category_name, description, parent_id, sort_order, status, create_time)

-- 商品表
goods (goods_id, goods_name, category_id, price, stock, description, image_url, status, create_time)

-- 订单表
orders (order_id, user_id, total_amount, status, shipping_address, contact_phone, remark, create_time)

-- 订单详情表
order_detail (detail_id, order_id, goods_id, goods_name, price, quantity, subtotal, create_time)

-- 留言表
message (message_id, customer_name, contact_info, content, status, reply_content, reply_time, reply_user_id, create_time)

-- 查询日志表
query_log (log_id, user_id, query_text, query_result, query_time)

-- 销量预测表
sales_prediction (prediction_id, category_id, category_name, predicted_sales, demand_level, accuracy_rate, prediction_date, create_time)
```

### 数据库文件说明

#### 1. schema.sql - 数据库结构
- 包含所有表的创建语句
- 包含索引和外键约束
- 不包含任何示例数据
- 适合生产环境使用

#### 2. sample_data.sql - 示例数据（可选）
- 包含管理员账号（admin/admin123）
- 包含商品种类和商品信息
- 包含历史订单和销售数据
- 适合测试和学习使用

### 数据库初始化步骤

#### 1. 仅创建表结构（生产环境推荐）
```bash
# 创建数据库
mysql -u root -p -e "CREATE DATABASE product_sales_system DEFAULT CHARSET utf8mb4 COLLATE utf8mb4_unicode_ci;"

# 导入表结构
mysql -u root -p product_sales_system < database/schema.sql
```

#### 2. 创建表结构 + 示例数据（开发/测试环境推荐）
```bash
# 创建数据库
mysql -u root -p -e "CREATE DATABASE product_sales_system DEFAULT CHARSET utf8mb4 COLLATE utf8mb4_unicode_ci;"

# 导入表结构
mysql -u root -p product_sales_system < database/schema.sql

# 导入示例数据
mysql -u root -p product_sales_system < database/sample_data.sql
```

> **注意**: 示例数据包含完整的测试数据，包括管理员账号、商品信息、历史订单等，适合快速体验系统功能。

### 表关系图
```
user (1) ──── (N) orders
orders (1) ──── (N) order_detail
goods (1) ──── (N) order_detail
category (1) ──── (N) goods
user (1) ──── (N) query_log
user (1) ──── (N) message (reply_user_id)
category (1) ──── (N) sales_prediction
```

---

## API接口

### RESTful API规范
```
GET    /api/resource          # 获取资源列表
GET    /api/resource/{id}     # 获取单个资源
POST   /api/resource          # 创建资源
PUT    /api/resource/{id}     # 更新资源
DELETE /api/resource/{id}     # 删除资源
```

### 统一响应格式
```json
{
    "code": 1,           // 1=成功, 0=失败
    "msg": "操作成功",    // 提示信息
    "data": {            // 数据内容
        // 具体数据
    }
}
```

### 主要API端点
```
用户管理:
POST   /api/user/login
POST   /api/user/logout
GET    /api/user/list
POST   /api/user/add

商品管理:
GET    /api/goods/list
POST   /api/goods/add
POST   /api/goods/update

订单管理:
POST   /api/order/create
GET    /api/order/list
GET    /api/order/detail
POST   /api/order/update-status

销售统计:
GET    /api/statistics/sales
GET    /api/statistics/export
GET    /api/statistics/top-goods

智能问数:
POST   /api/ai/query
POST   /api/ai/prediction
GET    /api/ai/prediction-history

留言管理:
POST   /api/message/submit
GET    /api/message/list
POST   /api/message/reply
POST   /api/message/delete
```

---

## 界面美化

### 🎨 设计理念
- **现代化设计**: 采用最新的UI设计趋势
- **响应式布局**: 完美适配桌面端和移动端
- **优雅动画**: 流畅的过渡效果和交互反馈
- **直观易用**: 清晰的信息层次和操作流程

### 🎯 视觉升级

#### 1. 色彩系统
- **主色调**: 现代紫色 (#4f46e5)
- **辅助色**: 灰色系 (#6b7280)
- **状态色**: 成功绿、警告橙、危险红、信息蓝
- **背景色**: 浅灰色 (#f8fafc)

#### 2. 组件样式
- **卡片**: 圆角设计，阴影效果，悬停动画
- **按钮**: 渐变背景，悬停提升效果
- **表格**: 悬停高亮，圆角边框
- **表单**: 聚焦状态，圆角输入框
- **侧边栏**: 渐变背景，图标导航

#### 3. 动画效果
- **页面加载**: 淡入上升动画
- **悬停效果**: 卡片提升，按钮变色
- **过渡动画**: 平滑的状态切换
- **滚动条**: 自定义美化样式

### 📱 响应式设计

#### 桌面端 (>768px)
- 固定侧边栏导航
- 多列布局
- 完整功能展示

#### 移动端 (<768px)
- 可折叠侧边栏
- 单列布局
- 触摸友好设计

---

## 部署指南





### 环境准备

#### 1. Python环境安装

**Windows系统**
1. 下载Python 3.8+：https://www.python.org/downloads/
2. 安装时勾选"Add Python to PATH"
3. 验证安装：
```cmd
python --version
pip --version
```

**Linux系统**
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install python3 python3-pip python3-venv

# CentOS/RHEL
sudo yum install python3 python3-pip
```

#### 2. MySQL数据库安装

**Windows系统**
1. 下载MySQL Installer：https://dev.mysql.com/downloads/installer/
2. 选择"MySQL Server"和"MySQL Workbench"
3. 设置root密码并记住

**Linux系统**
```bash
# Ubuntu/Debian
sudo apt install mysql-server mysql-client

# CentOS/RHEL
sudo yum install mysql-server mysql
sudo systemctl start mysqld
sudo systemctl enable mysqld
```

### 使用UV进行项目管理

#### 1. 安装UV
```bash
# 安装UV包管理器
pip install uv
```

#### 2. 创建项目环境
```bash
# 创建虚拟环境
uv venv

# 激活虚拟环境
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate
```

#### 3. 安装依赖
```bash
# 使用UV安装依赖
uv pip install -r requirements.txt

# 或者使用pyproject.toml
uv sync
```

#### 4. 启动系统
```bash
# 使用UV运行
uv run python start.py

# 或者使用run.py
python run.py
```


---

## 项目总结

### 核心功能实现

#### ✅ 1. 用户管理模块
- **登录验证**: MD5密码加密，Session状态管理
- **权限控制**: 管理员/普通用户角色分离
- **用户CRUD**: 完整的用户增删改查功能
- **状态管理**: 用户启用/禁用状态控制

#### ✅ 2. 商品种类管理模块
- **种类管理**: 商品种类的增删改查
- **层级支持**: 支持父子级种类关联
- **商品统计**: 显示每个种类下的商品数量
- **唯一性验证**: 种类名称重复检查

#### ✅ 3. 商品管理模块
- **商品信息**: 名称、价格、库存、描述管理
- **分类关联**: 与商品种类表关联
- **状态控制**: 上架/下架状态管理
- **库存管理**: 实时库存数量跟踪

#### ✅ 4. 订单管理模块
- **订单创建**: 支持多商品订单创建
- **状态管理**: 待发货/已发货/已完成/已取消
- **库存扣减**: 下单时自动扣减商品库存
- **订单详情**: 完整的订单信息查看
- **唯一订单号**: 时间戳+随机数生成

#### ✅ 5. 销售统计模块
- **多维度统计**: 时间维度、商品种类维度
- **图表可视化**: Chart.js实现销售趋势图、饼图
- **数据导出**: CSV格式数据导出功能
- **热销排行**: 商品销量排行榜
- **实时数据**: 从数据库实时查询统计

#### ✅ 6. AI智能问数模块
- **AI服务集成**: 集成SiliconFlow API，提供强大的AI能力
- **自然语言解析**: 支持中文自然语言查询，AI自动理解用户意图
- **智能类型识别**: 自动识别查询类型（销售额/销量/订单数/客户数/商品种类）
- **实时数据查询**: 直接查询数据库获取实际数据，确保结果准确性
- **AI解释生成**: 基于查询结果生成专业的自然语言解释和分析
- **查询示例**: 提供常见问题示例，帮助用户快速上手
- **结果展示**: 结构化的查询结果展示，包含AI增强信息

#### ✅ 7. 销量预测模块
- **历史数据分析**: 基于近6个月销售数据
- **移动平均算法**: 简单有效的时间序列预测
- **需求等级评估**: 高/中/低需求等级划分
- **预测可视化**: 图表展示预测结果
- **历史记录**: 预测结果保存和查看

#### ✅ 8. 留言管理模块
- **客户留言**: 客户可提交留言和反馈
- **管理员回复**: 管理员可回复客户留言
- **状态管理**: 未回复/已回复状态跟踪
- **内容过滤**: 防止SQL注入和XSS攻击
- **统计展示**: 留言数量统计

### 技术实现亮点

#### 1. 轻量级架构设计
- **无复杂框架**: 前端使用原生JavaScript，无Vue/React依赖
- **模块化开发**: 后端按功能模块分离，便于维护
- **RESTful API**: 标准化的API接口设计
- **响应式设计**: Bootstrap 5实现移动端适配

#### 2. 数据库设计优化
- **规范化设计**: 符合第三范式的表结构设计
- **索引优化**: 关键字段建立索引提升查询性能
- **外键约束**: 保证数据完整性和一致性
- **字符集支持**: UTF8MB4支持emoji等特殊字符

#### 3. 安全机制完善
- **密码加密**: MD5加密存储用户密码
- **SQL注入防护**: 使用参数化查询
- **XSS防护**: 输入内容过滤和转义
- **权限验证**: 装饰器实现接口权限控制

#### 4. 用户体验优化
- **直观界面**: Bootstrap组件提供美观UI
- **实时反馈**: 操作结果即时提示
- **数据可视化**: Chart.js图表展示数据
- **响应式布局**: 适配不同屏幕尺寸

### 项目特色

#### 1. 轻量级设计
- 无复杂依赖
- 快速部署
- 易于维护

#### 2. 功能完整
- 覆盖销售管理全流程
- 智能化功能
- 数据可视化

#### 3. 用户友好
- 直观界面
- 操作简单
- 响应迅速

#### 4. 扩展性强
- 模块化设计
- 标准化接口
- 易于扩展

### 适用场景

#### 1. 中小商户
- 零售店铺
- 网店管理
- 小型企业

#### 2. 学习项目
- Python学习
- Web开发
- 数据库设计

#### 3. 原型开发
- 快速验证
- 功能演示
- 技术选型

### 未来发展方向

#### 1. 功能扩展
- 移动端APP
- 微信小程序
- 第三方集成

#### 2. 技术升级
- 微服务架构
- 容器化部署
- 云原生应用

#### 3. 性能优化
- 缓存机制
- 数据库优化
- 前端优化

### 总结

本项目成功实现了一个功能完整、技术先进、用户友好的商品销售管理系统。通过轻量级的技术栈选择和模块化的设计思路，为中小商户提供了一个实用的销售管理解决方案。项目代码结构清晰，文档完善，具有良好的可维护性和扩展性。

#### 主要成就
- ✅ 完成8个核心功能模块
- ✅ 集成SiliconFlow AI服务，实现智能问数功能
- ✅ 实现AI增强的自然语言查询和解释生成
- ✅ 提供完整的前后端解决方案
- ✅ 支持生产环境部署
- ✅ 提供详细的文档和部署指南

#### 技术价值
- 展示了Python Web开发的最佳实践
- 提供了完整的项目架构设计
- 实现了实用的AI功能应用，集成SiliconFlow API
- 建立了标准化的开发流程
- 展示了AI与传统业务系统的完美结合

这个项目不仅是一个实用的销售管理系统，更是一个优秀的技术学习案例，为开发者提供了宝贵的经验和参考。

---

## 📞 支持

如有问题，请提交 Issue 或联系开发者。

## 📝 许可证

MIT License

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

