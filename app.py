from flask import Flask, request, jsonify, session, render_template, redirect, url_for
from flask import Flask, render_template, request, session, redirect, url_for, jsonify
from flask_cors import CORS
import pymysql
import hashlib
import json
import os
from datetime import datetime, timedelta
from functools import wraps

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'
CORS(app)

# 数据库配置
DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': '123456',
    'database': 'product_sales_system',
    'charset': 'utf8mb4'
}

def get_db_connection():
    """获取数据库连接"""
    return pymysql.connect(**DB_CONFIG)

def login_required(f):
    """登录验证装饰器"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            # 如果是API请求，返回JSON
            if request.path.startswith('/api/'):
                return jsonify({"code": 0, "msg": "请先登录"})
            # 如果是页面请求，重定向到登录页
            else:
                return redirect(url_for('login_page'))
        return f(*args, **kwargs)
    return decorated_function

def admin_required(f):
    """管理员权限验证装饰器"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return jsonify({"code": 0, "msg": "请先登录"})
        if session.get('role') != 'admin':
            return jsonify({"code": 0, "msg": "需要管理员权限"})
        return f(*args, **kwargs)
    return decorated_function

# 统一API错误处理，避免返回HTML导致前端JSON解析失败
@app.errorhandler(500)
def handle_internal_server_error(e):
    if request.path.startswith('/api/'):
        return jsonify({"code": 0, "msg": "服务器内部错误", "detail": str(e)}), 500
    # 非API请求仍返回默认错误页面
    return render_template('base.html'), 500

@app.errorhandler(Exception)
def handle_unexpected_exception(e):
    if request.path.startswith('/api/'):
        return jsonify({"code": 0, "msg": "请求处理异常", "detail": str(e)}), 500
    # 让Flask默认处理或简单回退
    return render_template('base.html'), 500

# 用户管理模块
@app.route('/api/user/login', methods=['POST'])
def user_login():
    """用户登录"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"code": 0, "msg": "请求数据不能为空"})
        
        username = data.get('username')
        password = data.get('password')
        
        if not username or not password:
            return jsonify({"code": 0, "msg": "用户名和密码不能为空"})
        
        # MD5加密密码
        password_md5 = hashlib.md5(password.encode()).hexdigest()
        
        conn = get_db_connection()
        try:
            cursor = conn.cursor(pymysql.cursors.DictCursor)
            cursor.execute(
                "SELECT user_id, username, role, real_name FROM user WHERE username=%s AND password=%s AND status=1",
                (username, password_md5)
            )
            user = cursor.fetchone()
            
            if user:
                session['user_id'] = user['user_id']
                session['username'] = user['username']
                session['role'] = user['role']
                session['real_name'] = user['real_name']
                
                return jsonify({
                    "code": 1, 
                    "msg": "登录成功", 
                    "data": {
                        "user_id": user['user_id'],
                        "username": user['username'],
                        "role": user['role'],
                        "real_name": user['real_name']
                    }
                })
            else:
                return jsonify({"code": 0, "msg": "用户名或密码错误"})
        except Exception as e:
            return jsonify({"code": 0, "msg": f"登录失败：{str(e)}"})
        finally:
            conn.close()
    except Exception as e:
        return jsonify({"code": 0, "msg": f"处理请求失败：{str(e)}"})

@app.route('/api/user/logout', methods=['POST'])
def user_logout():
    """用户登出"""
    session.clear()
    return jsonify({"code": 1, "msg": "登出成功"})

@app.route('/api/user/list', methods=['GET'])
@admin_required
def user_list():
    """获取用户列表"""
    page = int(request.args.get('page', 1))
    page_size = int(request.args.get('page_size', 10))
    keyword = request.args.get('keyword', '')
    
    offset = (page - 1) * page_size
    
    conn = get_db_connection()
    try:
        cursor = conn.cursor(pymysql.cursors.DictCursor)
        
        # 构建查询条件
        where_clause = "WHERE 1=1"
        params = []
        
        if keyword:
            where_clause += " AND (username LIKE %s OR real_name LIKE %s)"
            params.extend([f'%{keyword}%', f'%{keyword}%'])
        
        # 查询总数
        cursor.execute(f"SELECT COUNT(*) as total FROM user {where_clause}", params)
        total = cursor.fetchone()['total']
        
        # 查询数据
        cursor.execute(
            f"SELECT user_id, username, role, real_name, phone, email, status, create_time FROM user {where_clause} ORDER BY create_time DESC LIMIT %s OFFSET %s",
            params + [page_size, offset]
        )
        users = cursor.fetchall()
        
        return jsonify({
            "code": 1,
            "msg": "获取成功",
            "data": {
                "list": users,
                "total": total,
                "page": page,
                "page_size": page_size
            }
        })
    except Exception as e:
        return jsonify({"code": 0, "msg": f"获取失败：{str(e)}"})
    finally:
        conn.close()

@app.route('/api/user/add', methods=['POST'])
@admin_required
def user_add():
    """添加用户"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"code": 0, "msg": "请求数据不能为空"})
        
        username = data.get('username')
        password = data.get('password')
        role = data.get('role', 'normal')
        real_name = data.get('real_name')
        phone = data.get('phone')
        email = data.get('email')
        
        if not username or not password:
            return jsonify({"code": 0, "msg": "用户名和密码不能为空"})
        
        # MD5加密密码
        password_md5 = hashlib.md5(password.encode()).hexdigest()
        
        conn = get_db_connection()
        try:
            cursor = conn.cursor()
            
            # 检查用户名是否已存在
            cursor.execute("SELECT user_id FROM user WHERE username=%s", (username,))
            if cursor.fetchone():
                return jsonify({"code": 0, "msg": "用户名已存在"})
            
            # 插入新用户
            cursor.execute(
                "INSERT INTO user (username, password, role, real_name, phone, email) VALUES (%s, %s, %s, %s, %s, %s)",
                (username, password_md5, role, real_name, phone, email)
            )
            conn.commit()
            
            return jsonify({"code": 1, "msg": "用户添加成功"})
        except Exception as e:
            conn.rollback()
            return jsonify({"code": 0, "msg": f"添加失败：{str(e)}"})
        finally:
            conn.close()
    except Exception as e:
        return jsonify({"code": 0, "msg": f"处理请求失败：{str(e)}"})

# LLM配置管理模块
@app.route('/api/llm/config', methods=['GET'])
@admin_required
def get_llm_config():
    """获取当前LLM配置"""
    try:
        # 直接返回当前使用的SiliconFlow配置
        current_config = {
            "api_url": "https://api.siliconflow.cn/v1/chat/completions",
            "api_key": "sk-vfxewwbfcgthjjabvksdrezxdvtwqjmpdglubfthmzinlren",
            "model": "moonshotai/Kimi-K2-Instruct-0905",
            "timeout": 60,
            "retry_count": 3
        }
        
        preset_configs = {
            "siliconflow": {
                "name": "SiliconFlow",
                "api_url": "https://api.siliconflow.cn/v1/chat/completions",
                "default_model": "deepseek-ai/DeepSeek-V2.5",
                "auth_header": "Bearer"
            },
            "openai": {
                "name": "OpenAI", 
                "api_url": "https://api.openai.com/v1/chat/completions",
                "default_model": "gpt-4o-mini",
                "auth_header": "Bearer"
            }
        }
        
        return jsonify({
            "code": 1,
            "msg": "获取成功",
            "data": {
                "current_config": current_config,
                "preset_configs": preset_configs
            }
        })
    except Exception as e:
        return jsonify({"code": 0, "msg": f"获取配置失败：{str(e)}"})

@app.route('/api/llm/config', methods=['POST'])
@admin_required
def save_llm_config():
    """保存LLM配置"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"code": 0, "msg": "请求数据不能为空"})
        
        # 对于当前版本，只返回成功信息，不实际保存配置
        # 因为配置已经直接集成在代码中
        return jsonify({"code": 1, "msg": "配置已集成，无需保存"})
            
    except Exception as e:
        return jsonify({"code": 0, "msg": f"保存配置失败：{str(e)}"})

@app.route('/api/llm/test', methods=['POST'])
@admin_required
def test_llm_connection():
    """测试LLM连接"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"code": 0, "msg": "请求数据不能为空"})
        
        from ai_service import sales_ai_service
        
        # 测试配置
        test_config = {
            "api_url": data.get('api_url', '').strip(),
            "api_key": data.get('api_key', '').strip(),
            "model": data.get('model', '').strip(),
            "timeout": data.get('timeout', 60),
            "retry_count": 1,  # 测试时只重试1次
            "auth_header": data.get('auth_header', 'Bearer')
        }
        
        # 验证必要字段
        if not test_config['api_url'] or not test_config['api_key'] or not test_config['model']:
            return jsonify({"code": 0, "msg": "API地址、密钥和模型名称不能为空"})
        
        result = sales_ai_service.test_connection(test_config)
        
        if result['success']:
            return jsonify({
                "code": 1,
                "msg": "连接测试成功",
                "data": {
                    "response": result.get('response', '')
                }
            })
        else:
            return jsonify({
                "code": 0,
                "msg": f"连接测试失败: {result['message']}"
            })
            
    except Exception as e:
        return jsonify({"code": 0, "msg": f"测试连接失败：{str(e)}"})

# 商品种类管理模块
@app.route('/api/category/list', methods=['GET'])
@login_required
def category_list():
    """获取商品种类列表"""
    conn = get_db_connection()
    try:
        cursor = conn.cursor(pymysql.cursors.DictCursor)
        cursor.execute(
            "SELECT c.*, COUNT(g.goods_id) as goods_count FROM category c LEFT JOIN goods g ON c.category_id = g.category_id AND g.status = 1 WHERE c.status = 1 GROUP BY c.category_id ORDER BY c.sort_order, c.create_time"
        )
        categories = cursor.fetchall()
        
        return jsonify({"code": 1, "msg": "获取成功", "data": categories})
    except Exception as e:
        return jsonify({"code": 0, "msg": f"获取失败：{str(e)}"})
    finally:
        conn.close()

@app.route('/api/category/add', methods=['POST'])
@admin_required
def category_add():
    """添加商品种类"""
    data = request.get_json()
    category_name = data.get('category_name')
    description = data.get('description', '')
    parent_id = int(data.get('parent_id', 0))
    
    if not category_name:
        return jsonify({"code": 0, "msg": "种类名称不能为空"})
    
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        
        # 检查种类名称是否已存在
        cursor.execute("SELECT category_id FROM category WHERE category_name=%s", (category_name,))
        if cursor.fetchone():
            return jsonify({"code": 0, "msg": "种类名称已存在"})
        
        # 插入新种类
        cursor.execute(
            "INSERT INTO category (category_name, description, parent_id) VALUES (%s, %s, %s)",
            (category_name, description, parent_id)
        )
        conn.commit()
        
        return jsonify({"code": 1, "msg": "种类添加成功"})
    except Exception as e:
        conn.rollback()
        return jsonify({"code": 0, "msg": f"添加失败：{str(e)}"})
    finally:
        conn.close()

# 商品管理模块
@app.route('/api/goods/list', methods=['GET'])
@login_required
def goods_list():
    """获取商品列表"""
    page = int(request.args.get('page', 1))
    page_size = int(request.args.get('page_size', 10))
    category_id = request.args.get('category_id', '')
    keyword = request.args.get('keyword', '')
    
    offset = (page - 1) * page_size
    
    conn = get_db_connection()
    try:
        cursor = conn.cursor(pymysql.cursors.DictCursor)
        
        # 构建查询条件
        where_clause = "WHERE g.status = 1"
        params = []
        
        if category_id:
            where_clause += " AND g.category_id = %s"
            params.append(category_id)
        
        if keyword:
            where_clause += " AND g.goods_name LIKE %s"
            params.append(f'%{keyword}%')
        
        # 查询总数
        cursor.execute(f"SELECT COUNT(*) as total FROM goods g {where_clause}", params)
        total = cursor.fetchone()['total']
        
        # 查询数据
        cursor.execute(
            f"SELECT g.*, c.category_name FROM goods g LEFT JOIN category c ON g.category_id = c.category_id {where_clause} ORDER BY g.create_time DESC LIMIT %s OFFSET %s",
            params + [page_size, offset]
        )
        goods_list = cursor.fetchall()
        
        return jsonify({
            "code": 1,
            "msg": "获取成功",
            "data": {
                "list": goods_list,
                "total": total,
                "page": page,
                "page_size": page_size
            }
        })
    except Exception as e:
        return jsonify({"code": 0, "msg": f"获取失败：{str(e)}"})
    finally:
        conn.close()

# 模块导入将在需要时动态导入，避免循环导入问题

# 订单管理路由
@app.route('/api/order/create', methods=['POST'])
@login_required
def api_create_order():
    from order_module import create_order
    return create_order()

@app.route('/api/order/list', methods=['GET'])
@login_required
def api_get_order_list():
    from order_module import get_order_list
    return get_order_list()

@app.route('/api/order/detail', methods=['GET'])
@login_required
def api_get_order_detail():
    from order_module import get_order_detail
    return get_order_detail()

@app.route('/api/order/update-status', methods=['POST'])
@admin_required
def api_update_order_status():
    from order_module import update_order_status
    return update_order_status()

# 销售统计路由
@app.route('/api/statistics/sales', methods=['GET'])
@login_required
def api_get_sales_statistics():
    from statistics_module import get_sales_statistics
    return get_sales_statistics()

@app.route('/api/statistics/export', methods=['GET'])
@login_required
def api_export_sales_data():
    from statistics_module import export_sales_data
    return export_sales_data()

@app.route('/api/statistics/top-goods', methods=['GET'])
@login_required
def api_get_top_selling_goods():
    from statistics_module import get_top_selling_goods
    return get_top_selling_goods()

# 智能问数路由（兼容 /api 与非 /api 路径，前端要求使用 /ai/*）
@app.route('/api/ai/query', methods=['POST'])
@app.route('/ai/query', methods=['POST'])
@login_required
def api_intelligent_query():
    from ai_module import intelligent_query
    from flask import jsonify
    
    try:
        data = request.get_json()
        if not data:
            return jsonify({"code": 0, "msg": "请求数据不能为空"})
        
        query_text = data.get('query_text', '').strip()
        if not query_text:
            return jsonify({"code": 0, "msg": "查询内容不能为空"})
        
        user_id = session.get('user_id', 0)
        
        result = intelligent_query(query_text, user_id)
        return jsonify(result)
    except Exception as e:
        return jsonify({"code": 0, "msg": f"处理请求失败: {str(e)}"})

@app.route('/api/ai/prediction', methods=['POST'])
@app.route('/ai/prediction', methods=['POST'])
@admin_required
def api_sales_prediction():
    from ai_module import sales_prediction
    from flask import jsonify
    
    result = sales_prediction()
    return jsonify(result)

@app.route('/api/ai/prediction-history', methods=['GET'])
@app.route('/ai/prediction-history', methods=['GET'])
@login_required
def api_get_prediction_history():
    from ai_module import get_prediction_history
    from flask import jsonify
    
    page = int(request.args.get('page', 1))
    page_size = int(request.args.get('page_size', 10))
    
    result = get_prediction_history(page, page_size)
    return jsonify(result)

@app.route('/api/ai/query-history', methods=['GET'])
@login_required
def api_get_query_history():
    from ai_module import get_query_history
    from flask import jsonify
    
    user_id = session.get('user_id', 0)
    page = int(request.args.get('page', 1))
    page_size = int(request.args.get('page_size', 10))
    
    result = get_query_history(user_id, page, page_size)
    return jsonify(result)

# 留言管理路由
@app.route('/api/message/submit', methods=['POST'])
def api_submit_message():
    from message_module import submit_message
    return submit_message()

@app.route('/api/message/list', methods=['GET'])
@admin_required
def api_get_message_list():
    from message_module import get_message_list
    return get_message_list()

@app.route('/api/message/reply', methods=['POST'])
@admin_required
def api_reply_message():
    from message_module import reply_message
    return reply_message()

@app.route('/api/message/delete', methods=['POST'])
@admin_required
def api_delete_message():
    from message_module import delete_message
    return delete_message()

@app.route('/api/message/stats', methods=['GET'])
@admin_required
def api_get_message_stats():
    from message_module import get_message_stats
    return get_message_stats()

# 商品管理路由
@app.route('/api/goods/add', methods=['POST'])
@admin_required
def goods_add():
    """添加商品"""
    data = request.get_json()
    goods_name = data.get('goods_name')
    category_id = data.get('category_id')
    price = data.get('price')
    stock = data.get('stock', 0)
    description = data.get('description', '')
    
    if not goods_name or not category_id or not price:
        return jsonify({"code": 0, "msg": "商品名称、种类和价格不能为空"})
    
    try:
        price = float(price)
        stock = int(stock)
    except ValueError:
        return jsonify({"code": 0, "msg": "价格和库存必须是数字"})
    
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        
        # 检查种类是否存在
        cursor.execute("SELECT category_id FROM category WHERE category_id=%s AND status=1", (category_id,))
        if not cursor.fetchone():
            return jsonify({"code": 0, "msg": "商品种类不存在"})
        
        # 插入新商品
        cursor.execute(
            "INSERT INTO goods (goods_name, category_id, price, stock, description) VALUES (%s, %s, %s, %s, %s)",
            (goods_name, category_id, price, stock, description)
        )
        conn.commit()
        
        return jsonify({"code": 1, "msg": "商品添加成功"})
    except Exception as e:
        conn.rollback()
        return jsonify({"code": 0, "msg": f"添加失败：{str(e)}"})
    finally:
        conn.close()

@app.route('/api/goods/update', methods=['POST'])
@admin_required
def goods_update():
    """更新商品"""
    data = request.get_json()
    goods_id = data.get('goods_id')
    goods_name = data.get('goods_name')
    category_id = data.get('category_id')
    price = data.get('price')
    stock = data.get('stock')
    description = data.get('description', '')
    status = data.get('status', 1)
    
    if not goods_id:
        return jsonify({"code": 0, "msg": "商品ID不能为空"})
    
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        
        # 检查商品是否存在
        cursor.execute("SELECT goods_id FROM goods WHERE goods_id=%s", (goods_id,))
        if not cursor.fetchone():
            return jsonify({"code": 0, "msg": "商品不存在"})
        
        # 构建更新语句
        update_fields = []
        params = []
        
        if goods_name:
            update_fields.append("goods_name = %s")
            params.append(goods_name)
        
        if category_id:
            update_fields.append("category_id = %s")
            params.append(category_id)
        
        if price:
            try:
                price = float(price)
                update_fields.append("price = %s")
                params.append(price)
            except ValueError:
                return jsonify({"code": 0, "msg": "价格必须是数字"})
        
        if stock is not None:
            try:
                stock = int(stock)
                update_fields.append("stock = %s")
                params.append(stock)
            except ValueError:
                return jsonify({"code": 0, "msg": "库存必须是整数"})
        
        if description is not None:
            update_fields.append("description = %s")
            params.append(description)
        
        if status is not None:
            update_fields.append("status = %s")
            params.append(status)
        
        if not update_fields:
            return jsonify({"code": 0, "msg": "没有需要更新的字段"})
        
        params.append(goods_id)
        sql = f"UPDATE goods SET {', '.join(update_fields)} WHERE goods_id = %s"
        
        cursor.execute(sql, params)
        conn.commit()
        
        return jsonify({"code": 1, "msg": "商品更新成功"})
    except Exception as e:
        conn.rollback()
        return jsonify({"code": 0, "msg": f"更新失败：{str(e)}"})
    finally:
        conn.close()

@app.route('/favicon.ico')
def favicon():
    """处理favicon.ico请求"""
    return '', 204  # 返回空内容，状态码204表示无内容

# 前端页面路由
@app.route('/')
def index():
    """首页"""
    return render_template('index.html')

@app.route('/login')
def login_page():
    """登录页"""
    return render_template('login.html')

@app.route('/logout')
def logout():
    """登出"""
    session.clear()
    return redirect(url_for('index'))

@app.route('/dashboard')
@login_required
def dashboard():
    """仪表板"""
    return render_template('dashboard.html')

@app.route('/users')
@login_required
def users_page():
    """用户管理页"""
    return render_template('users.html')

@app.route('/categories')
@login_required
def categories_page():
    """商品种类管理页"""
    return render_template('categories.html')

@app.route('/goods')
@login_required
def goods_page():
    """商品管理页"""
    return render_template('goods.html')

@app.route('/orders')
@login_required
def orders_page():
    """订单管理页"""
    return render_template('orders.html')

@app.route('/statistics')
@login_required
def statistics_page():
    """销售统计页"""
    return render_template('statistics.html')

@app.route('/ai-query')
@login_required
def ai_query_page():
    """智能问数页"""
    return render_template('ai_query.html')

@app.route('/llm-config')
@admin_required
def llm_config_page():
    """LLM配置管理页面"""
    return render_template('llm_config.html')

@app.route('/prediction')
@login_required
def prediction_page():
    """销量预测页"""
    return render_template('prediction.html')

@app.route('/test-api')
def test_api_page():
    """测试API页面"""
    return render_template('../test_api.html')

@app.route('/messages')
@login_required
def messages_page():
    """留言管理页"""
    return render_template('messages.html')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
