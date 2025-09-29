from flask import request, jsonify, session
import pymysql
# import pandas as pd
# import numpy as np
from datetime import datetime, timedelta
from app import get_db_connection, login_required

def get_sales_statistics():
    """获取销售统计数据"""
    period = request.args.get('period', '7days')  # 7days, month, today, custom
    start_date = request.args.get('start_date', '')
    end_date = request.args.get('end_date', '')
    
    conn = get_db_connection()
    try:
        cursor = conn.cursor(pymysql.cursors.DictCursor)
        
        # 调试：检查数据库中的基本数据
        cursor.execute("SELECT COUNT(*) as count FROM orders")
        total_orders = cursor.fetchone()['count']
        print(f"数据库总订单数: {total_orders}")
        
        cursor.execute("SELECT COUNT(*) as count FROM order_detail")
        total_details = cursor.fetchone()['count']
        print(f"数据库订单详情数: {total_details}")
        
        cursor.execute("SELECT COUNT(*) as count FROM goods")
        total_goods = cursor.fetchone()['count']
        print(f"数据库商品数: {total_goods}")
        
        # 检查今日数据
        today = datetime.now().strftime('%Y-%m-%d')
        cursor.execute("SELECT COUNT(*) as count FROM orders WHERE DATE(create_time) = %s", [today])
        today_orders = cursor.fetchone()['count']
        print(f"今日订单数: {today_orders}")
        
        cursor.execute("SELECT SUM(total_amount) as total FROM orders WHERE DATE(create_time) = %s AND status != 'cancelled'", [today])
        today_sales = cursor.fetchone()['total']
        print(f"今日销售额: {today_sales}")
        
        # 如果没有数据，返回空数据
        if total_orders == 0:
            return jsonify({
                "code": 1,
                "msg": "获取成功",
                "data": {
                    "overall_stats": {
                        'total_sales': 0.0,
                        'total_orders': 0,
                        'avg_order_value': 0.0,
                        'unique_customers': 0
                    },
                    "category_stats": [],
                    "daily_trends": [],
                    "period": {
                        "start_date": start_date,
                        "end_date": end_date
                    }
                }
            })
        
        # 根据时间范围构建查询条件
        if period == 'today':
            start_date = datetime.now().strftime('%Y-%m-%d')
            end_date = datetime.now().strftime('%Y-%m-%d')
            where_clause = "WHERE DATE(o.create_time) = %s AND o.status != 'cancelled'"
            params = [start_date]
        elif period == '7days':
            start_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
            end_date = datetime.now().strftime('%Y-%m-%d')
            where_clause = "WHERE DATE(o.create_time) BETWEEN %s AND %s AND o.status != 'cancelled'"
            params = [start_date, end_date]
        elif period == 'month':
            start_date = datetime.now().replace(day=1).strftime('%Y-%m-%d')
            end_date = datetime.now().strftime('%Y-%m-%d')
            where_clause = "WHERE DATE(o.create_time) BETWEEN %s AND %s AND o.status != 'cancelled'"
            params = [start_date, end_date]
        else:
            # 默认查询所有数据
            where_clause = "WHERE o.status != 'cancelled'"
            params = []
        
        print(f"查询条件: {where_clause}")
        print(f"查询参数: {params}")
        print(f"当前日期: {datetime.now().strftime('%Y-%m-%d')}")
        
        # 总体统计
        cursor.execute(f"""
            SELECT 
                COUNT(*) as total_orders,
                COALESCE(SUM(o.total_amount), 0) as total_sales,
                COALESCE(AVG(o.total_amount), 0) as avg_order_value,
                COUNT(DISTINCT o.user_id) as unique_customers
            FROM orders o 
            {where_clause}
        """, params)
        overall_stats = cursor.fetchone()
        
        # 确保统计数据不为None
        if overall_stats:
            overall_stats['total_sales'] = float(overall_stats['total_sales'] or 0)
            overall_stats['total_orders'] = int(overall_stats['total_orders'] or 0)
            overall_stats['avg_order_value'] = float(overall_stats['avg_order_value'] or 0)
            overall_stats['unique_customers'] = int(overall_stats['unique_customers'] or 0)
        else:
            # 如果没有数据，返回默认值
            overall_stats = {
                'total_sales': 0.0,
                'total_orders': 0,
                'avg_order_value': 0.0,
                'unique_customers': 0
            }
        
        # 按商品种类统计
        cursor.execute(f"""
            SELECT 
                c.category_name,
                COALESCE(SUM(od.quantity), 0) as total_quantity,
                COALESCE(SUM(od.subtotal), 0) as total_sales,
                COUNT(DISTINCT od.order_id) as order_count
            FROM order_detail od
            LEFT JOIN goods g ON od.goods_id = g.goods_id
            LEFT JOIN category c ON g.category_id = c.category_id
            LEFT JOIN orders o ON od.order_id = o.order_id
            {where_clause}
            GROUP BY c.category_id, c.category_name
            ORDER BY total_sales DESC
            LIMIT 10
        """, params)
        category_stats = cursor.fetchall()
        
        # 每日销售趋势
        cursor.execute(f"""
            SELECT 
                DATE(o.create_time) as sale_date,
                COUNT(*) as daily_orders,
                COALESCE(SUM(o.total_amount), 0) as daily_sales
            FROM orders o 
            {where_clause}
            GROUP BY DATE(o.create_time)
            ORDER BY sale_date
        """, params)
        daily_trends = cursor.fetchall()
        
        return jsonify({
            "code": 1,
            "msg": "获取成功",
            "data": {
                "overall_stats": overall_stats,
                "category_stats": category_stats,
                "daily_trends": daily_trends,
                "period": {
                    "start_date": start_date,
                    "end_date": end_date
                }
            }
        })
    except Exception as e:
        return jsonify({"code": 0, "msg": f"获取失败：{str(e)}"})
    finally:
        conn.close()

def export_sales_data():
    """导出销售数据"""
    period = request.args.get('period', '7days')
    start_date = request.args.get('start_date', '')
    end_date = request.args.get('end_date', '')
    export_type = request.args.get('type', 'orders')  # orders, categories
    
    conn = get_db_connection()
    try:
        cursor = conn.cursor(pymysql.cursors.DictCursor)
        
        # 根据时间范围构建查询条件
        if period == '7days':
            start_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
            end_date = datetime.now().strftime('%Y-%m-%d')
        elif period == 'month':
            start_date = datetime.now().replace(day=1).strftime('%Y-%m-%d')
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        where_clause = "WHERE DATE(o.create_time) BETWEEN %s AND %s AND o.status != 'cancelled'"
        params = [start_date, end_date]
        
        if export_type == 'orders':
            # 导出订单数据
            cursor.execute(f"""
                SELECT 
                    o.order_id,
                    o.create_time,
                    u.username,
                    u.real_name,
                    o.total_amount,
                    o.status,
                    o.shipping_address,
                    o.contact_phone
                FROM orders o
                LEFT JOIN user u ON o.user_id = u.user_id
                {where_clause}
                ORDER BY o.create_time DESC
            """, params)
            data = cursor.fetchall()
            
            # 生成CSV内容
            if not data:
                csv_content = ""
            else:
                # 获取列名
                columns = list(data[0].keys())
                csv_content = ",".join(columns) + "\n"
                
                # 添加数据行
                for row in data:
                    csv_content += ",".join(str(row[col]) for col in columns) + "\n"
            
        elif export_type == 'categories':
            # 导出商品种类销售数据
            cursor.execute(f"""
                SELECT 
                    c.category_name,
                    COALESCE(SUM(od.quantity), 0) as total_quantity,
                    COALESCE(SUM(od.subtotal), 0) as total_sales,
                    COUNT(DISTINCT od.order_id) as order_count,
                    COALESCE(AVG(od.price), 0) as avg_price
                FROM order_detail od
                LEFT JOIN goods g ON od.goods_id = g.goods_id
                LEFT JOIN category c ON g.category_id = c.category_id
                LEFT JOIN orders o ON od.order_id = o.order_id
                {where_clause}
                GROUP BY c.category_id, c.category_name
                ORDER BY total_sales DESC
            """, params)
            data = cursor.fetchall()
            
            # 生成CSV内容
            if not data:
                csv_content = ""
            else:
                # 获取列名
                columns = list(data[0].keys())
                csv_content = ",".join(columns) + "\n"
                
                # 添加数据行
                for row in data:
                    csv_content += ",".join(str(row[col]) for col in columns) + "\n"
        
        return jsonify({
            "code": 1,
            "msg": "导出成功",
            "data": {
                "csv_content": csv_content,
                "filename": f"sales_data_{period}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            }
        })
        
    except Exception as e:
        return jsonify({"code": 0, "msg": f"导出失败：{str(e)}"})
    finally:
        conn.close()

def get_top_selling_goods():
    """获取热销商品排行"""
    limit = int(request.args.get('limit', 10))
    period = request.args.get('period', '7days')
    start_date = request.args.get('start_date', '')
    end_date = request.args.get('end_date', '')
    
    conn = get_db_connection()
    try:
        cursor = conn.cursor(pymysql.cursors.DictCursor)
        
        # 根据时间范围构建查询条件
        if period == '7days':
            start_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
            end_date = datetime.now().strftime('%Y-%m-%d')
        elif period == 'month':
            start_date = datetime.now().replace(day=1).strftime('%Y-%m-%d')
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        where_clause = "WHERE DATE(o.create_time) BETWEEN %s AND %s AND o.status != 'cancelled'"
        params = [start_date, end_date]
        
        cursor.execute(f"""
            SELECT 
                g.goods_name,
                c.category_name,
                COALESCE(SUM(od.quantity), 0) as total_quantity,
                COALESCE(SUM(od.subtotal), 0) as total_sales,
                COUNT(DISTINCT od.order_id) as order_count,
                COALESCE(AVG(od.price), 0) as avg_price
            FROM order_detail od
            LEFT JOIN goods g ON od.goods_id = g.goods_id
            LEFT JOIN category c ON g.category_id = c.category_id
            LEFT JOIN orders o ON od.order_id = o.order_id
            {where_clause}
            GROUP BY g.goods_id, g.goods_name, c.category_name
            ORDER BY total_quantity DESC
            LIMIT %s
        """, params + [limit])
        
        top_goods = cursor.fetchall()
        
        return jsonify({
            "code": 1,
            "msg": "获取成功",
            "data": top_goods
        })
    except Exception as e:
        return jsonify({"code": 0, "msg": f"获取失败：{str(e)}"})
    finally:
        conn.close()
