#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI模块 - 智能问数和销量预测功能实现
"""

import json
import os
import logging
from datetime import datetime
from decimal import Decimal

import pymysql

from ai_service import sales_ai_service, DecimalEncoder
from app import get_db_connection

def ensure_dirs():
    """确保必要的目录存在"""
    base_dirs = [
        "data",
        os.path.join("data", "ai_query", "temp"),
        os.path.join("data", "ai_forecast", "temp"),
    ]
    for d in base_dirs:
        os.makedirs(d, exist_ok=True)

def get_data_paths():
    """获取数据路径配置"""
    return {
        "ai_query_temp": os.path.join("data", "ai_query", "temp"),
        "ai_forecast_temp": os.path.join("data", "ai_forecast", "temp")
    }

def safe_format_error(error_msg: str) -> str:
    """安全地格式化错误消息，避免包含花括号的内容导致格式化错误"""
    return str(error_msg).replace('{', '{{').replace('}', '}}')

# ===== 智能问数模块 =====
def intelligent_query(query_text: str, user_id: int) -> dict:
    """处理智能问数请求
    
    Args:
        query_text: 用户的自然语言问题
        user_id: 用户ID
        
    Returns:
        dict: 包含code、msg和data的响应结果
    """
    try:
        # 1. 分析用户查询
        parsed_query = sales_ai_service.analyze_query(query_text)
        if not parsed_query.get('success', False):
            # 检查失败原因
            time_data = parsed_query.get('time')
            metric = parsed_query.get('metric')
            
            if not time_data:
                return {"code": 0, "msg": "请补充时间范围", "data": {}}
            elif not metric:
                return {"code": 0, "msg": "请明确查询指标（如销售额、销量、订单数等）", "data": {}}
            else:
                return {"code": 0, "msg": "无法解析查询内容，请尝试重新表述", "data": {}}
        
        # 2. 获取数据库连接和商品种类
        conn = get_db_connection()
        try:
            cursor = conn.cursor(pymysql.cursors.DictCursor)
            
            # 获取所有商品种类
            categories = sales_ai_service.get_categories_from_db(cursor)
            
            # 匹配用户查询中的商品种类
            matched_category = sales_ai_service.match_category_by_query(query_text, categories)
            if matched_category:
                parsed_query['category'] = matched_category
        except Exception as e:
            return {"code": 0, "msg": f"数据库查询失败: {safe_format_error(str(e))}", "data": {}}
        finally:
            conn.close()
        
        # 3. 构建SQL并执行查询
        sql, params, metric = sales_ai_service.build_query_sql(parsed_query, categories)
        
        conn = get_db_connection()
        try:
            cursor = conn.cursor(pymysql.cursors.DictCursor)
            cursor.execute(sql, params)
            result = cursor.fetchone()
            value = result['value'] if result and result['value'] is not None else 0
        except Exception as e:
            return {"code": 0, "msg": f"数据查询失败: {safe_format_error(str(e))}", "data": {}}
        finally:
            conn.close()
        
        # 4. 导出数据到CSV
        csv_path = sales_ai_service.export_query_data_to_csv(query_text, parsed_query, value, user_id)
        
        # 5. 使用LLM处理查询
        llm_result = sales_ai_service.process_query_with_llm(query_text, csv_path, metric)
        if not llm_result["success"]:
            return {"code": 0, "msg": "API调用失败", "data": {}}
        
        # 6. 写入查询日志
        save_query_log(user_id, query_text, llm_result["data"])
        
        # 7. 返回结果
        return {"code": 1, "msg": "查询成功", "data": llm_result["data"]}
        
    except Exception as e:
        # 安全处理异常消息，避免格式化错误
        error_msg = str(e).replace('{', '{{').replace('}', '}}')
        return {"code": 0, "msg": f"处理查询失败: {error_msg}", "data": {}}

def get_query_history(user_id: int, page: int = 1, page_size: int = 10) -> dict:
    """获取用户的查询历史记录
    
    Args:
        user_id: 用户ID
        page: 页码
        page_size: 每页记录数
        
    Returns:
        dict: 包含code、msg和data的响应结果
    """
    try:
        conn = get_db_connection()
        try:
            cursor = conn.cursor(pymysql.cursors.DictCursor)
            
            # 计算偏移量
            offset = (page - 1) * page_size
            
            # 查询总记录数
            cursor.execute("SELECT COUNT(*) as total FROM query_log WHERE user_id = %s", (user_id,))
            total = cursor.fetchone()['total']
            
            # 查询当前页数据
            cursor.execute("""
                SELECT query_id, query_text, query_result, query_time 
                FROM query_log 
                WHERE user_id = %s 
                ORDER BY query_time DESC 
                LIMIT %s OFFSET %s
            """, (user_id, page_size, offset))
            history = cursor.fetchall()
            
            # 处理JSON字段
            for item in history:
                if item['query_result']:
                    try:
                        item['query_result'] = json.loads(item['query_result'])
                    except:
                        item['query_result'] = {}
                
                # 格式化时间
                if isinstance(item['query_time'], datetime):
                    item['query_time'] = item['query_time'].strftime('%Y-%m-%d %H:%M:%S')
            
            return {
                "code": 1,
                "msg": "获取成功",
                "data": {
                    "total": total,
                    "page": page,
                    "page_size": page_size,
                    "total_pages": (total + page_size - 1) // page_size,
                    "history": history
                }
            }
        except Exception as e:
            return {"code": 0, "msg": f"查询历史记录失败: {safe_format_error(str(e))}", "data": {}}
        finally:
            conn.close()
    except Exception as e:
        return {"code": 0, "msg": f"数据库连接失败: {safe_format_error(str(e))}", "data": {}}

def save_query_log(user_id: int, query_text: str, query_result: dict) -> bool:
    """保存查询日志
    
    Args:
        user_id: 用户ID
        query_text: 查询文本
        query_result: 查询结果
        
    Returns:
        bool: 是否保存成功
    """
    try:
        conn = get_db_connection()
        try:
            cursor = conn.cursor()
            
            # 将查询结果转换为JSON字符串
            result_json = json.dumps(query_result, ensure_ascii=False, cls=DecimalEncoder)
            
            # 插入日志
            cursor.execute(
                "INSERT INTO query_log (user_id, query_text, query_result, query_time) VALUES (%s, %s, %s, NOW())",
                (user_id, query_text, result_json)
            )
            conn.commit()
            return True
        except Exception as e:
            conn.rollback()
            print(f"保存查询日志失败: {safe_format_error(str(e))}")
            return False
        finally:
            conn.close()
    except Exception as e:
        print(f"数据库连接失败: {safe_format_error(str(e))}")
        return False

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("ai_prediction.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("ai_prediction")

# ===== 销量预测模块 =====
def sales_prediction() -> dict:
    """执行销量预测
    
    Returns:
        dict: 包含code、msg和data的响应结果
    """
    try:
        logger.info("开始执行销量预测")
        
        # 1. 导出历史销售数据
        logger.info("正在导出历史销售数据")
        conn = get_db_connection()
        try:
            cursor = conn.cursor(pymysql.cursors.DictCursor)
            historical_data_path = sales_ai_service.export_historical_sales_data(cursor)
            logger.info(f"历史数据导出成功: {historical_data_path}")
        except Exception as e:
            logger.error(f"导出历史数据失败: {safe_format_error(str(e))}")
            return {"code": 0, "msg": f"导出历史数据失败: {safe_format_error(str(e))}", "data": {}}
        finally:
            conn.close()
        
        # 2. 检查历史数据是否存在及内容
        if not os.path.exists(historical_data_path):
            logger.error(f"历史数据文件不存在: {historical_data_path}")
            return {"code": 0, "msg": "历史数据文件不存在", "data": {}}
        
        # 检查历史数据内容
        try:
            with open(historical_data_path, 'r', encoding='utf-8') as f:
                historical_data = json.load(f)
                if not historical_data:
                    logger.warning("历史数据为空")
                else:
                    logger.info(f"历史数据包含 {len(historical_data)} 条记录")
        except Exception as e:
            logger.error(f"读取历史数据失败: {safe_format_error(str(e))}")
            return {"code": 0, "msg": f"读取历史数据失败: {safe_format_error(str(e))}", "data": {}}
        
        # 3. 使用LLM进行预测
        logger.info("正在调用LLM进行销量预测")
        try:
            # 使用SiliconFlow客户端进行预测
            logger.info("尝试使用SiliconFlow客户端")
            llm_result = sales_ai_service.process_prediction_with_llm(historical_data_path)
            if not llm_result["success"]:
                error_msg = llm_result.get("error", "未知错误")
                logger.error(f"AI预测失败: {error_msg}")
                return {"code": 0, "msg": f"AI预测失败: {error_msg}", "data": {}}
        except Exception as e:
            logger.error(f"预测过程中发生异常: {safe_format_error(str(e))}")
            return {"code": 0, "msg": f"预测过程异常: {safe_format_error(str(e))}", "data": {}}
        
        logger.info("LLM预测调用成功")
        
        # 4. 获取预测数据
        prediction_data = llm_result["data"]
        predictions = prediction_data.get("predictions", [])
        
        if not predictions:
            logger.warning("未生成预测结果")
            return {"code": 0, "msg": "未生成预测结果", "data": {}}
        
        logger.info(f"生成了 {len(predictions)} 条预测结果")
        
        # 5. 验证并增强预测结果
        # 重新连接数据库以获取基期数据
        conn_for_base = get_db_connection()
        try:
            cursor_for_base = conn_for_base.cursor(pymysql.cursors.DictCursor)
            enhanced_predictions = sales_ai_service.validate_and_enhance_predictions(predictions, cursor_for_base)
        finally:
            conn_for_base.close()
        
        # 6. 保存预测结果到数据库
        current_date = datetime.now().strftime('%Y-%m-%d')
        save_prediction_results(enhanced_predictions, current_date)
        
        # 7. 返回结果
        return {
            "code": 1,
            "msg": "预测成功",
            "data": {
                "predictions": enhanced_predictions,
                "prediction_date": current_date
            }
        }
        
    except Exception as e:
        return {"code": 0, "msg": f"预测过程失败: {safe_format_error(str(e))}", "data": {}}

def get_prediction_history(page: int = 1, page_size: int = 10) -> dict:
    """获取销量预测历史记录
    
    Args:
        page: 页码
        page_size: 每页记录数
        
    Returns:
        dict: 包含code、msg和data的响应结果
    """
    try:
        logger.info(f"开始获取预测历史记录，页码: {page}, 每页数量: {page_size}")
        conn = get_db_connection()
        try:
            cursor = conn.cursor(pymysql.cursors.DictCursor)
            
            # 计算偏移量
            offset = (page - 1) * page_size
            
            # 查询总记录数
            cursor.execute("SELECT COUNT(DISTINCT prediction_date) as total FROM sales_prediction")
            total = cursor.fetchone()['total']
            logger.info(f"预测历史总记录数: {total}")
            
            # 查询当前页数据（按预测日期分组，获取最新的预测结果）
            # 使用派生表替代LIMIT & IN子查询语法，以兼容各MySQL版本
            cursor.execute("""
                SELECT 
                    sp.prediction_id,
                    sp.category_id,
                    sp.category_name,
                    sp.predicted_sales,
                    sp.demand_level,
                    sp.accuracy_rate,
                    sp.prediction_date,
                    sp.create_time,
                    c.category_name as actual_category_name
                FROM sales_prediction sp
                LEFT JOIN category c ON sp.category_id = c.category_id
                INNER JOIN (
                    SELECT DISTINCT prediction_date 
                    FROM sales_prediction 
                    ORDER BY prediction_date DESC 
                    LIMIT %s OFFSET %s
                ) AS p ON sp.prediction_date = p.prediction_date
                ORDER BY sp.prediction_date DESC, sp.category_name
            """, (page_size, offset))
            history = cursor.fetchall()
            
            # 格式化时间并重新计算增长率
            # 获取基期销量数据用于重新计算增长率
            base_sales_dict = sales_ai_service.get_base_period_sales(cursor)
            
            for item in history:
                if isinstance(item['prediction_date'], datetime):
                    item['prediction_date'] = item['prediction_date'].strftime('%Y-%m-%d')
                if isinstance(item['create_time'], datetime):
                    item['create_time'] = item['create_time'].strftime('%Y-%m-%d %H:%M:%S')
                
                # 安全处理accuracy_rate字段，可能是Decimal类型
                accuracy_rate = item.get('accuracy_rate', 50)
                if isinstance(accuracy_rate, Decimal):
                    accuracy_rate = float(accuracy_rate)
                elif accuracy_rate is None:
                    accuracy_rate = 50
                else:
                    try:
                        accuracy_rate = float(accuracy_rate)
                    except (ValueError, TypeError):
                        accuracy_rate = 50
                
                # 使用新的增长率计算方法：(AI预测量-基期)/基期（不乘以100）
                category_id = item.get('category_id', 0)
                predicted_sales = float(item.get('predicted_sales', 0))
                base_sales = base_sales_dict.get(category_id, 0.0)
                
                if base_sales > 0:
                    growth_rate = (predicted_sales - base_sales) / base_sales
                    item['growth_rate'] = round(growth_rate, 3)  # 保留3位小数，不乘以100
                else:
                    # 如果基期销量为0，使用accuracy_rate/1000作为备用方案（对应原来的/10/100）
                    item['growth_rate'] = round(accuracy_rate / 1000, 3)
                
                # 补充置信度字段
                item['confidence'] = round(accuracy_rate / 100, 2)
                
                # 补充ai_enhanced字段
                item['ai_enhanced'] = True
                
                # 使用实际的种类名称
                if item.get('actual_category_name'):
                    item['category_name'] = item['actual_category_name']
                
                # 移除不需要的字段
                if 'actual_category_name' in item:
                    del item['actual_category_name']
            
            # 按预测日期分组（为了前端兼容性，同时返回扁平数据）
            grouped_history = {}
            for item in history:
                date = item['prediction_date']
                if date not in grouped_history:
                    grouped_history[date] = []
                grouped_history[date].append(item)
            
            # 为了更好的前端渲染，直接返回扁平的历史记录
            # 前端可以根据需要进行分组处理
            logger.info(f"成功获取 {len(history)} 条预测历史记录")
            return {
                "code": 1,
                "msg": "获取成功",
                "data": {
                    "total": total,
                    "page": page,
                    "page_size": page_size,
                    "total_pages": (total + page_size - 1) // page_size,
                    "history": history  # 直接返回扁平化的历史记录
                }
            }
        except Exception as e:
            logger.error(f"查询预测历史失败: {safe_format_error(str(e))}")
            return {"code": 0, "msg": f"查询预测历史失败: {safe_format_error(str(e))}", "data": {}}
        finally:
            conn.close()
    except Exception as e:
        logger.error(f"数据库连接失败: {safe_format_error(str(e))}")
        return {"code": 0, "msg": f"数据库连接失败: {safe_format_error(str(e))}", "data": {}}

def save_prediction_results(predictions: list, prediction_date: str) -> bool:
    """保存预测结果到数据库
    
    Args:
        predictions: 预测结果列表
        prediction_date: 预测日期
        
    Returns:
        bool: 是否保存成功
    """
    try:
        logger.info(f"开始保存预测结果到数据库，预测日期: {prediction_date}")
        conn = get_db_connection()
        try:
            cursor = conn.cursor()
            
            # 开启事务
            conn.begin()
            
            # 先删除当天的预测结果（如果有）
            cursor.execute(
                "DELETE FROM sales_prediction WHERE prediction_date = %s",
                (prediction_date,)
            )
            
            # 插入新的预测结果
            count = 0
            for pred in predictions:
                # accuracy_rate为置信度*100
                accuracy_rate = float(pred.get('confidence', 0.5)) * 100
                
                # 获取种类名称，确保不为空
                category_name = pred.get('category_name', '未知')
                if not category_name or category_name.strip() == '':
                    category_name = '未知'
                    logger.warning(f"种类ID {pred.get('category_id', 0)} 缺少种类名称，使用默认值")
                
                cursor.execute(
                    """
                    INSERT INTO sales_prediction (
                        category_id,
                        category_name,
                        predicted_sales,
                        demand_level,
                        accuracy_rate,
                        prediction_date,
                        create_time
                    ) VALUES (%s, %s, %s, %s, %s, %s, NOW())
                    """,
                    (
                        pred.get('category_id', 0),
                        category_name,
                        pred.get('predicted_sales', 0),
                        pred.get('demand_level', 'medium'),
                        round(accuracy_rate, 2),
                        prediction_date
                    )
                )
                count += 1
            
            # 提交事务
            conn.commit()
            logger.info(f"成功保存 {count} 条预测结果到数据库")
            return True
        except Exception as e:
            conn.rollback()
            logger.error(f"保存预测结果失败: {safe_format_error(str(e))}")
            return False
        finally:
            conn.close()
    except Exception as e:
        logger.error(f"数据库连接失败: {safe_format_error(str(e))}")
        return False

# ===== 数据库查询辅助函数 =====
def get_sales_data_by_time_range(start_date: str, end_date: str, category_id: int = None) -> list:
    """获取指定时间范围内的销售数据
    
    Args:
        start_date: 开始日期 (YYYY-MM-DD)
        end_date: 结束日期 (YYYY-MM-DD)
        category_id: 商品种类ID（可选）
        
    Returns:
        list: 销售数据列表
    """
    try:
        conn = get_db_connection()
        try:
            cursor = conn.cursor(pymysql.cursors.DictCursor)
            
            # 构建SQL查询
            sql = """
                SELECT 
                    DATE(o.create_time) as sale_date,
                    g.category_id,
                    c.category_name,
                    COUNT(DISTINCT o.order_id) as order_count,
                    SUM(od.quantity) as total_quantity,
                    SUM(od.subtotal) as total_sales
                FROM orders o
                LEFT JOIN order_detail od ON o.order_id = od.order_id
                LEFT JOIN goods g ON od.goods_id = g.goods_id
                LEFT JOIN category c ON g.category_id = c.category_id
                WHERE o.status IN ('shipped', 'completed')
                AND DATE(o.create_time) BETWEEN %s AND %s
            """
            
            params = [start_date, end_date]
            
            # 添加商品种类条件
            if category_id:
                sql += " AND g.category_id = %s"
                params.append(category_id)
            
            # 添加分组和排序
            sql += " GROUP BY DATE(o.create_time), g.category_id, c.category_name"
            sql += " ORDER BY sale_date, g.category_id"
            
            cursor.execute(sql, params)
            return cursor.fetchall()
        finally:
            conn.close()
    except Exception as e:
        print(f"获取销售数据失败: {safe_format_error(str(e))}")
        return []

def get_category_sales_summary(category_id: int) -> dict:
    """获取指定商品种类的销售汇总
    
    Args:
        category_id: 商品种类ID
        
    Returns:
        dict: 销售汇总数据
    """
    try:
        conn = get_db_connection()
        try:
            cursor = conn.cursor(pymysql.cursors.DictCursor)
            
            sql = """
                SELECT 
                    SUM(od.quantity) as total_quantity,
                    SUM(od.subtotal) as total_sales,
                    COUNT(DISTINCT o.order_id) as order_count
                FROM order_detail od
                LEFT JOIN goods g ON od.goods_id = g.goods_id
                LEFT JOIN orders o ON od.order_id = o.order_id
                WHERE o.status IN ('shipped', 'completed')
                AND g.category_id = %s
            """
            
            cursor.execute(sql, (category_id,))
            result = cursor.fetchone()
            
            # 处理空结果
            if not result:
                return {"total_quantity": 0, "total_sales": 0, "order_count": 0}
            
            return result
        finally:
            conn.close()
    except Exception as e:
        print(f"获取分类销售汇总失败: {safe_format_error(str(e))}")
        return {"total_quantity": 0, "total_sales": 0, "order_count": 0}
