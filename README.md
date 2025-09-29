# 🛒 商品销售管理系统

基于 Python Flask + MySQL 的现代化商品销售管理系统，专为中小商户设计。

## ✨ 系统特色

- 🛒 **完整订单管理** - 订单创建、状态更新、库存自动扣减
- 📊 **多维度统计** - 时间维度、商品种类维度统计，支持数据导出
- 🤖 **智能问数** - 自然语言查询销售数据，支持关键词解析
- 🔮 **销量预测** - 基于历史数据的智能预测，提供需求等级评估
- 👥 **用户管理** - 多角色权限控制，管理员和普通用户分离
- 📝 **留言系统** - 客户留言提交和回复管理
- 🏷️ **商品分类** - 支持层级分类，便于商品组织

## 🚀 快速开始

### 环境要求
- Python 3.8+
- MySQL 8.0+
- UV包管理器（推荐）

### 1. 安装UV（推荐）
```bash
# 安装UV包管理器
pip install uv
```

### 2. 克隆项目并安装依赖
```bash
# 克隆项目
git clone https://github.com/binggo1285573797/Product-Sales-Management-System.git
cd Product-Sales-Management-System

# 一键创建虚拟环境并安装依赖
uv sync
```

### 3. 激活虚拟环境
```bash
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate
```

### 4. 配置数据库
1. 创建数据库：
```sql
CREATE DATABASE product_sales_system DEFAULT CHARSET utf8mb4 COLLATE utf8mb4_unicode_ci;
```

2. 导入数据库结构：
```bash
mysql -u root -p product_sales_system < database/schema.sql
```

3. （可选）导入示例数据：
```bash
mysql -u root -p product_sales_system < database/sample_data.sql
```
> **注意**: 示例数据包含管理员账号（admin/admin123）、商品种类、商品信息和历史订单数据，适合测试和学习使用。

4. 修改数据库配置（编辑 `app.py`）：
```python
DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': 'your_password',  # 修改为您的MySQL密码
    'database': 'product_sales_system',
    'charset': 'utf8mb4'
}
```

### 5. 启动系统
```bash
# 方式一：使用UV运行（推荐）
uv run python start.py

# 方式二：直接运行
python start.py
```

访问 `http://localhost:5000` 即可使用系统。

### 6. 默认账号
- **管理员账号**: admin
- **密码**: admin123

## 📋 主要功能

### 🏠 仪表板
- 实时销售数据展示
- 销售趋势图表
- 商品种类销售占比
- 最近订单列表

### 📦 商品管理
- 商品信息管理（名称、价格、库存、描述）
- 商品分类管理（支持层级分类）
- 商品状态控制（上架/下架）

### 🛒 订单管理
- 订单创建（多商品支持）
- 订单状态管理（待发货/已发货/已完成/已取消）
- 自动库存扣减
- 订单详情查看

### 📊 销售统计
- 多维度统计展示
- 图表可视化
- 数据导出功能（CSV格式）
- 热销商品排行

### 🤖 智能功能
- **AI智能问数** - 集成SiliconFlow API，支持自然语言查询销售数据
- **智能解析** - AI自动识别查询类型（销售额/销量/订单数/客户数）
- **实时查询** - 直接查询数据库获取实际数据，提供准确结果
- **自然语言解释** - AI生成专业的查询结果解释和分析
- **时间范围识别** - 智能识别时间条件（今天/昨天/最近N天/本月/今年等）
- **商品种类匹配** - 自动匹配查询中的商品种类过滤条件
- **销量预测** - 基于历史数据的智能预测，提供需求等级评估
- **查询历史** - 保存用户查询记录，支持历史回顾

### 👥 用户管理
- 多角色权限控制
- 用户状态管理
- 留言系统

## 🛠️ 技术栈

- **后端**: Python 3.7+ + Flask 2.3.3
- **前端**: HTML5 + CSS3 + JavaScript + Bootstrap 5
- **数据库**: MySQL 8.0
- **图表**: Chart.js
- **AI服务**: SiliconFlow API (支持Qwen系列模型), requests
- **其他**: PyMySQL, pandas, numpy

## 📁 项目结构

```
product_sales_system/
├── app.py                 # 主应用文件
├── start.py              # 启动脚本
├── pyproject.toml         # UV项目配置（包含所有依赖）
├── order_module.py        # 订单管理模块
├── statistics_module.py   # 销售统计模块
├── ai_module.py          # AI功能模块（智能问数，销量预测）
├── ai_service.py         # AI服务模块（SiliconFlow API集成）
├── message_module.py     # 留言管理模块
├── database/
│   ├── schema.sql        # 数据库结构
│   └── sample_data.sql  # 示例数据（可选）
├── templates/            # HTML模板
│   ├── base.html         # 基础模板
│   ├── login.html        # 登录页面
│   ├── dashboard.html    # 仪表板
│   ├── orders.html       # 订单管理
│   ├── statistics.html   # 销售统计
│   ├── ai_query.html     # 智能问数
│   ├── prediction.html   # 销量预测
│   ├── categories.html   # 商品分类
│   ├── goods.html        # 商品管理
│   ├── users.html        # 用户管理
│   ├── messages.html     # 留言管理
│   └── index.html        # 首页
├── README.md             # 项目说明
└── DOCUMENTATION.md      # 完整文档
```

## 🚀 部署说明

### 开发环境
```bash
# 使用UV运行（推荐）
uv run python start.py

# 或直接运行
python start.py
```

### 生产环境
```bash
# 安装生产依赖
uv add gunicorn

# 启动服务
uv run gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

## 📝 许可证

MIT License

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📚 详细文档

查看 [DOCUMENTATION.md](./DOCUMENTATION.md) 获取完整的系统文档，包括：
- 系统架构详解
- 功能模块说明
- 数据库设计
- API接口文档
- 界面美化指南
- 部署指南
- 项目总结

## 📞 支持

如有问题，请提交 Issue 或联系开发者。

