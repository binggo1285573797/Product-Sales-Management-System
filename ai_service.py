#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI服务模块 - LLM封装与数据获取
"""

import json
import time
import re
import os
import csv
import logging
from typing import Any, Dict, List, Optional, Union
from datetime import datetime, timedelta
from decimal import Decimal

import requests
import pymysql

def safe_format_error(error_msg: str) -> str:
    """安全地格式化错误消息，避免包含花括号的内容导致格式化错误"""
    return str(error_msg).replace('{', '{{').replace('}', '}}')

# Custom JSON encoder for Decimal and datetime
class DecimalEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Decimal):
            return float(obj)
        elif isinstance(obj, datetime):
            return obj.isoformat()
        return super(DecimalEncoder, self).default(obj)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("ai_service.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("ai_service")

class LLMClient:
    """通用LLM客户端，支持重试与严格JSON输出清洗"""

    def __init__(self, use_siliconflow: bool = True) -> None:
        # 直接使用SiliconFlow配置
        if use_siliconflow:
            self.base_url = "https://api.siliconflow.cn/v1/chat/completions"
            self.api_key = "sk-vfxewwbfcgthjjabvksdrezxdvtwqjmpdglubfthmzinlren"
            self.model = "moonshotai/Kimi-K2-Instruct-0905"
            self.timeout = 60
            self.retry_count = 3
            logger.info(f"初始化SiliconFlow客户端，模型: {self.model}")
        else:
            # 备用配置
            self.base_url = "https://api.openai.com/v1/chat/completions"
            self.api_key = ""
            self.model = "gpt-4o-mini"
            self.timeout = 60
            self.retry_count = 3
            logger.info(f"初始化备用客户端，模型: {self.model}")
        
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def chat(self, messages: List[Dict[str, str]], max_tokens: int = 1024, temperature: float = 0.3) -> Dict[str, Any]:
        """调用LLM API，支持重试机制"""
        logger.info(f"开始调用LLM API，模型: {self.model}，最大tokens: {max_tokens}")
        last_error: Optional[str] = None
        for attempt in range(1, self.retry_count + 1):
            try:
                logger.info(f"API调用第 {attempt} 次尝试")
                payload = {
                    "model": self.model,
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                }
                resp = requests.post(self.base_url, headers=self.headers, json=payload, timeout=self.timeout)
                if resp.status_code == 200:
                    data = resp.json()
                    content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                    logger.info(f"API调用成功，返回内容长度: {len(content)} 字符")
                    return {"success": True, "content": content}
                last_error = f"HTTP {resp.status_code}: {resp.text}"
                logger.warning(f"API调用失败，状态码: {resp.status_code}")
            except Exception as e:  # noqa: BLE001
                last_error = str(e)
                logger.warning(f"API调用异常: {safe_format_error(str(e))}")
            time.sleep(1.5)  # 固定延迟1.5秒后重试
        logger.error(f"所有重试均失败: {last_error or '未知错误'}")
        return {"success": False, "error": f"API调用失败: {last_error or '未知错误'}"}

    @staticmethod
    def extract_json_block(text: str) -> str:
        """从文本中提取JSON块"""
        t = text.strip()
        if "```json" in t:
            t = t.split("```json", 1)[1]
            t = t.split("```", 1)[0]
            return t.strip()
        if "```" in t:
            t = t.split("```", 1)[1]
            t = t.split("```", 1)[0]
            return t.strip()
        # 尝试从第一个{开始到最后一个}结束
        start = t.find("{")
        end = t.rfind("}")
        if start != -1 and end != -1 and end > start:
            return t[start : end + 1]
        return t

class AIService:
    """AI服务类：封装问数解释与预测调用"""

    def __init__(self) -> None:
        # 直接使用SiliconFlow配置
        self.client = LLMClient(use_siliconflow=True)
    
    def ensure_dirs(self):
        """确保必要的目录存在"""
        base_dirs = [
            "data",
            os.path.join("data", "ai_query", "temp"),
            os.path.join("data", "ai_forecast", "temp"),
        ]
        for d in base_dirs:
            os.makedirs(d, exist_ok=True)
    
    def get_data_paths(self):
        """获取数据路径配置"""
        return {
            "ai_query_temp": os.path.join("data", "ai_query", "temp"),
            "ai_forecast_temp": os.path.join("data", "ai_forecast", "temp")
        }
    
    def test_connection(self, config: dict = None) -> Dict[str, Any]:
        """测试大模型API连接"""
        try:
            # 如果提供了配置，创建临时客户端测试
            if config:
                # 创建一个临时的LLMClient实例进行测试
                test_client = LLMClient(use_siliconflow=False)  # 不使用默认配置
                # 手动设置测试配置
                test_client.base_url = config.get("api_url")
                test_client.api_key = config.get("api_key")
                test_client.model = config.get("model")
                test_client.timeout = config.get("timeout", 30)
                test_client.retry_count = config.get("retry_count", 1)
                
                auth_header = config.get("auth_header", "Bearer")
                test_client.headers = {
                    "Authorization": f"{auth_header} {test_client.api_key}",
                    "Content-Type": "application/json",
                }
            else:
                test_client = self.client
            
            # 发送简单的测试消息
            test_messages = [
                {"role": "user", "content": "你好，请回复'连接成功'四个字"}
            ]
            
            result = test_client.chat(test_messages, max_tokens=10, temperature=0)
            
            if result["success"]:
                return {
                    "success": True,
                    "message": "API连接成功",
                    "response": result["content"]
                }
            else:
                return {
                    "success": False,
                    "message": f"API连接失败: {result['error']}"
                }
        except Exception as e:
            return {
                "success": False,
                "message": f"测试连接异常: {str(e)}"
            }

    # ===== 智能问数相关功能 =====
    def analyze_query(self, query_text: str) -> Dict[str, Any]:
        """分析用户查询，提取时间范围、商品种类和查询指标"""
        result = {
            "time": None,
            "category": None,
            "metric": None,
            "success": False
        }

        # 解析时间范围
        time_patterns = {
            'today': [r'今天|今日|当天'],
            'yesterday': [r'昨天|昨日|前一天'],
            'recent': [r'最近(\d+)天'],
            'this_month': [r'本月|这个月|当月'],
            'this_year': [r'今年|本年|当年'],
            'month': [r'(\d{4})年(\d{1,2})月'],
        }

        query_lower = query_text.lower()
        
        for time_type, patterns in time_patterns.items():
            for pattern in patterns:
                match = re.search(pattern, query_lower)
                if match:
                    if time_type == 'today':
                        result['time'] = {'type': 'today'}
                    elif time_type == 'yesterday':
                        result['time'] = {'type': 'yesterday'}
                    elif time_type == 'recent':
                        days = int(match.group(1))
                        result['time'] = {'type': 'recent', 'days': days}
                    elif time_type == 'this_month':
                        result['time'] = {'type': 'this_month'}
                    elif time_type == 'this_year':
                        result['time'] = {'type': 'this_year'}
                    elif time_type == 'month':
                        year = int(match.group(1))
                        month = int(match.group(2))
                        result['time'] = {'type': 'month', 'year': year, 'month': month}
                    break
            if result['time']:
                break

        # 解析查询指标
        metric_patterns = {
            'sales': [r'销售额|销售金额|营业额|营收|收入'],
            'quantity': [r'销量|销售量|数量|件数|台数'],
            'orders': [r'订单数|订单量|订单|单数'],
            'customers': [r'客户数|用户数|客户|用户|买家数'],
        }

        for metric_type, patterns in metric_patterns.items():
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    result['metric'] = metric_type
                    break
            if result['metric']:
                break

        # 解析商品种类（需要从数据库获取种类名称）
        # 注意：这里简化处理，实际应用中应该从数据库获取所有商品种类并匹配
        # 在调用函数中会补充这部分信息

        result['success'] = bool(result['time'] and result['metric'])
        return result

    def build_query_sql(self, parsed_query: Dict[str, Any], categories: List[Dict[str, Any]]) -> tuple:
        """构建查询SQL和参数"""
        time_data = parsed_query.get('time')
        metric = parsed_query.get('metric')
        category_data = parsed_query.get('category')
        category_id = category_data.get('category_id') if category_data else None
        
        # 检查必要的参数
        if not time_data or not metric:
            raise ValueError("缺少必要的时间或指标参数")
        
        # 检查time_data的类型
        if not isinstance(time_data, dict) or 'type' not in time_data:
            raise ValueError("时间数据格式错误")

        # 构建WHERE子句
        where_clauses = ["o.status IN ('shipped', 'completed')"]  # 仅统计有效订单
        params = []

        # 处理时间条件
        time_type = time_data.get('type')
        if time_type == 'today':
            where_clauses.append("DATE(o.create_time) = CURDATE()")
        elif time_type == 'yesterday':
            where_clauses.append("DATE(o.create_time) = DATE_SUB(CURDATE(), INTERVAL 1 DAY)")
        elif time_type == 'recent':
            days = time_data.get('days', 7)  # 默认查询7天
            where_clauses.append("o.create_time >= DATE_SUB(NOW(), INTERVAL %s DAY)")
            params.append(days)
        elif time_type == 'this_month':
            where_clauses.append("DATE_FORMAT(o.create_time, '%%Y-%%m') = DATE_FORMAT(NOW(), '%%Y-%%m')")
        elif time_type == 'this_year':
            where_clauses.append("YEAR(o.create_time) = YEAR(NOW())")
        elif time_type == 'month':
            year = time_data.get('year', datetime.now().year)
            month = time_data.get('month', datetime.now().month)
            where_clauses.append("YEAR(o.create_time) = %s AND MONTH(o.create_time) = %s")
            params.append(year)
            params.append(month)
        else:
            # 未知时间类型，默认查询最近 7 天
            where_clauses.append("o.create_time >= DATE_SUB(NOW(), INTERVAL 7 DAY)")

        # 处理商品种类条件
        if category_id:
            where_clauses.append("g.category_id = %s")
            params.append(category_id)

        # 构建SELECT子句和FROM子句
        if metric == 'sales':
            select_clause = "SUM(od.subtotal) AS value"
            from_clause = "FROM order_detail od LEFT JOIN orders o ON od.order_id = o.order_id LEFT JOIN goods g ON od.goods_id = g.goods_id"
        elif metric == 'quantity':
            select_clause = "SUM(od.quantity) AS value"
            from_clause = "FROM order_detail od LEFT JOIN orders o ON od.order_id = o.order_id LEFT JOIN goods g ON od.goods_id = g.goods_id"
        elif metric == 'orders':
            select_clause = "COUNT(DISTINCT o.order_id) AS value"
            from_clause = "FROM orders o LEFT JOIN order_detail od ON o.order_id = od.order_id LEFT JOIN goods g ON od.goods_id = g.goods_id"
        elif metric == 'customers':
            select_clause = "COUNT(DISTINCT o.user_id) AS value"
            from_clause = "FROM orders o LEFT JOIN order_detail od ON o.order_id = od.order_id LEFT JOIN goods g ON od.goods_id = g.goods_id"
        else:
            select_clause = "0 AS value"
            from_clause = "FROM orders o"

        # 组合SQL
        sql = f"SELECT {select_clause} {from_clause} WHERE {' AND '.join(where_clauses)}"

        return sql, params, metric

    def export_query_data_to_csv(self, query_text: str, parsed_query: Dict[str, Any], value: Any, user_id: int) -> str:
        """导出查询数据到CSV文件"""
        self.ensure_dirs()
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        csv_path = os.path.join(self.get_data_paths()["ai_query_temp"], f"query_{user_id}_{timestamp}.csv")
        
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            # 写入表头
            writer.writerow(['query_text', 'info_type', 'data_value', 'time_range', 'category_id', 'category_name'])
            
            # 准备数据
            metric = parsed_query.get('metric', 'unknown')
            time_data = parsed_query.get('time')
            if time_data:
                time_desc = self.format_time_period(time_data)
            else:
                time_desc = "未知时间范围"
            category_data = parsed_query.get('category')
            cat = category_data if category_data else {}
            
            # 根据指标类型格式化数值
            if metric == 'sales':
                try:
                    data_value = round(float(value), 2)
                except:
                    data_value = 0.0
            else:
                try:
                    data_value = int(float(value))
                except:
                    data_value = 0
            
            writer.writerow([query_text, metric, data_value, time_desc, cat.get('category_id', ''), cat.get('category_name', '')])
        
        return csv_path

    def generate_query_prompt(self, query_text: str, csv_path: str, metric: str) -> str:
        """生成查询解释的Prompt"""
        # 读取CSV数据
        with open(csv_path, 'r', encoding='utf-8') as f:
            csv_content = f.read()
        
        # 定义JSON格式示例，避免在f-string中直接使用包含大括号的内容
        json_format = '''{
  "query_text": "用户的原始问题",
  "info_type": "查询的指标类型（销售额/销量/订单数/客户数）",
  "data_value": "数值结果（销售额保留2位小数，销量/订单数为整数）",
  "explanation": "对结果的解释和分析",
  "ai_enhanced": true
}'''
        
        prompt = f"""你是专业的销售数据分析师。请基于用户问题和CSV数据，生成精准的回答。

用户问题：{query_text}

CSV数据内容：
{csv_content}

请按照以下要求输出JSON格式：
{json_format}

请严格按照JSON格式输出，不要包含其他无关内容。"""
        
        return prompt

    def process_query_with_llm(self, query_text: str, csv_path: str, metric: str) -> Dict[str, Any]:
        """使用LLM处理查询"""
        prompt = self.generate_query_prompt(query_text, csv_path, metric)
        
        messages = [
            {"role": "system", "content": "你是专业的销售数据分析助理，只输出JSON格式的回答。"},
            {"role": "user", "content": prompt}
        ]
        
        # 使用SiliconFlow客户端
        result = self.client.chat(messages, max_tokens=500)
        
        if not result["success"]:
            return {"success": False, "error": result["error"]}
        
        try:
            json_content = self.client.extract_json_block(result["content"])
            parsed_json = json.loads(json_content)
            return {"success": True, "data": parsed_json}
        except Exception as e:
            return {"success": False, "error": f"解析AI响应失败: {safe_format_error(str(e))}"}

    # ===== 销量预测相关功能 =====
    def export_historical_sales_data(self, cursor: pymysql.cursors.Cursor) -> str:
        """导出历史销售数据用于预测"""
        self.ensure_dirs()
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        json_path = os.path.join(self.get_data_paths()["ai_forecast_temp"], f"forecast_history_{timestamp}.json")
        
        # 计算6个月前的日期
        six_months_ago = datetime.now() - timedelta(days=180)
        
        # 查询最近6个月的每日销量数据，按商品种类聚合
        sql = """
        SELECT 
            DATE(o.create_time) as sale_date,
            c.category_id,
            c.category_name,
            SUM(od.quantity) as daily_sales
        FROM order_detail od
        LEFT JOIN goods g ON od.goods_id = g.goods_id
        LEFT JOIN category c ON g.category_id = c.category_id
        LEFT JOIN orders o ON od.order_id = o.order_id
        WHERE o.status IN ('shipped', 'completed') 
        AND o.create_time >= %s
        GROUP BY DATE(o.create_time), c.category_id, c.category_name
        ORDER BY sale_date, c.category_id
        """
        
        cursor.execute(sql, (six_months_ago,))
        sales_data = cursor.fetchall()
        
        # 计算每个种类的数据完整性
        category_data = {}
        result_data = []
        
        for row in sales_data:
            category_id = row['category_id']
            if category_id not in category_data:
                category_data[category_id] = {'count': 0, 'start_date': row['sale_date']}
            category_data[category_id]['count'] += 1
            
            # 添加数据完整性字段
            data_completeness = "100%"
            # 简单估算：如果数据点少于180*0.9=162，则认为不完整
            if category_data[category_id]['count'] < 162:
                data_completeness = f"{int(category_data[category_id]['count']/180*100)}%"
            
            result_row = {
                'sale_date': row['sale_date'].strftime('%Y-%m-%d') if isinstance(row['sale_date'], datetime) else str(row['sale_date']),
                'category_id': row['category_id'],
                'category_name': row['category_name'],
                'daily_sales': int(row['daily_sales']),
                'data_completeness': data_completeness
            }
            result_data.append(result_row)
        
        # 保存为JSON文件
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(result_data, f, ensure_ascii=False, indent=2, cls=DecimalEncoder)
        
        return json_path

    def generate_prediction_prompt(self, historical_data_path: str) -> str:
        """生成销量预测的Prompt"""
        with open(historical_data_path, 'r', encoding='utf-8') as f:
            historical_data = json.load(f)
        
        # 按种类分组数据，只取前30条作为示例
        category_data = {}  
        for item in historical_data:
            category_id = item['category_id']
            if category_id not in category_data:
                category_data[category_id] = []
            category_data[category_id].append(item)
        
        # 准备数据示例
        data_examples = []
        for cat_id, items in category_data.items():
            data_examples.extend(items[:30])  # 每个种类取前30条
        
        # 构建JSON格式示例，避免在f-string中直接使用包含大括号的JSON
        json_format_example = '''
{
  "predictions": [
    {
      "category_id": 种类ID（整数）,
      "category_name": "种类名称",
      "predicted_sales": 预测未来1个月总销量（整数）,
      "demand_level": "需求等级（high/medium/low）",
      "growth_rate": 增长率（保留1位小数的百分比）,
      "confidence": 置信度（0-1之间的小数，根据数据完整性计算）,
      "ai_enhanced": true
    }
  ]
}
        '''.strip()
        
        prompt = f"""你是专业的销售数据分析师。请基于历史销售数据预测未来销量。

历史销售数据（前6个月，部分示例）：
{json.dumps(data_examples, ensure_ascii=False, indent=2)}

请按照以下要求输出JSON格式：
{json_format_example}

请注意：
1. 分析前6个月销量趋势、周期规律和异常值
2. 预测未来1个月每个种类的总销量
3. 需求等级：预测值高于历史均值为high，接近为medium，低于为low
4. 置信度：完整6个月数据为0.9，数据完整性每降低10%，置信度降低0.05
5. 严格按照JSON格式输出，不要包含其他无关内容。"""
        
        return prompt

    def process_prediction_with_llm(self, historical_data_path: str) -> Dict[str, Any]:
        """使用LLM处理销量预测"""
        logger.info(f"开始处理销量预测，历史数据路径: {historical_data_path}")
        try:
            prompt = self.generate_prediction_prompt(historical_data_path)
            logger.info(f"生成预测prompt，长度: {len(prompt)} 字符")
            
            messages = [
                {"role": "system", "content": "你是专业的销售预测分析师，只输出JSON格式的预测结果。"},
                {"role": "user", "content": prompt}
            ]
            
            # 使用SiliconFlow客户端进行预测
            logger.info("使用SiliconFlow客户端进行预测")
            result = self.client.chat(messages, max_tokens=2000)
            
            if not result["success"]:
                logger.error(f"LLM预测调用失败: {result['error']}")
                return {"success": False, "error": result["error"]}
            
            try:
                logger.info("开始解析LLM返回结果")
                json_content = self.client.extract_json_block(result["content"])
                parsed_json = json.loads(json_content)
                logger.info(f"成功解析预测结果，包含键: {list(parsed_json.keys())}")
                return {"success": True, "data": parsed_json}
            except json.JSONDecodeError as e:
                logger.error(f"JSON解析错误: {safe_format_error(str(e))}")
                logger.error(f"原始内容: {result['content'][:200]}...")  # 只记录前200个字符
                return {"success": False, "error": f"JSON解析失败: {safe_format_error(str(e))}"}  
            except Exception as e:
                logger.error(f"解析AI预测结果失败: {safe_format_error(str(e))}")
                return {"success": False, "error": f"解析AI预测结果失败: {safe_format_error(str(e))}"}
        except Exception as e:
            logger.error(f"处理预测过程中发生异常: {safe_format_error(str(e))}")
            return {"success": False, "error": f"处理预测失败: {safe_format_error(str(e))}"}

    def get_base_period_sales(self, cursor: pymysql.cursors.Cursor) -> Dict[int, float]:
        """获取各商品种类的基期销量数据（最近30天）"""
        try:
            # 计算30天前的日期作为基期
            thirty_days_ago = datetime.now() - timedelta(days=30)
            
            # 查询最近30天的销量数据，按商品种类聚合
            sql = """
            SELECT 
                c.category_id,
                c.category_name,
                COALESCE(SUM(od.quantity), 0) as base_sales
            FROM category c
            LEFT JOIN goods g ON c.category_id = g.category_id
            LEFT JOIN order_detail od ON g.goods_id = od.goods_id
            LEFT JOIN orders o ON od.order_id = o.order_id
            WHERE o.status IN ('shipped', 'completed') 
            AND o.create_time >= %s
            GROUP BY c.category_id, c.category_name
            """
            
            cursor.execute(sql, (thirty_days_ago,))
            base_data = cursor.fetchall()
            
            # 转换为字典格式：{category_id: base_sales}
            base_sales_dict = {}
            for row in base_data:
                category_id = row['category_id']
                base_sales = float(row['base_sales']) if row['base_sales'] else 0.0
                base_sales_dict[category_id] = base_sales
            
            logger.info(f"获取到 {len(base_sales_dict)} 个商品种类的基期销量数据")
            return base_sales_dict
            
        except Exception as e:
            logger.error(f"获取基期销量数据失败: {safe_format_error(str(e))}")
            return {}
    
    def calculate_growth_rate(self, predicted_sales: float, base_sales: float) -> float:
        """计算增长率：(AI预测量-基期)/基期"""
        try:
            if base_sales <= 0:
                # 如果基期销量为0或负数，使用默认增长率
                return 0.05  # 对应5%，
            
            growth_rate = (predicted_sales - base_sales) / base_sales
            return round(growth_rate, 3)  # 保留3位小数
            
        except Exception as e:
            logger.error(f"计算增长率失败: {safe_format_error(str(e))}")
            return 0.05
    
    def validate_and_enhance_predictions(self, predictions: List[Dict[str, Any]], cursor: pymysql.cursors.Cursor = None) -> List[Dict[str, Any]]:
        """验证并增强预测结果"""
        enhanced_predictions = []
        
        # 获取基期销量数据
        base_sales_dict = {}
        if cursor:
            base_sales_dict = self.get_base_period_sales(cursor)
        
        for pred in predictions:
            # 补全缺失字段
            category_id = pred.get('category_id', 0)
            try:
                category_id = int(category_id)
            except:
                category_id = 0
            
            confidence = pred.get('confidence', 0.5)
            try:
                confidence = float(confidence)
                # 确保置信度在0-1之间
                confidence = max(0, min(1, confidence))
            except:
                confidence = 0.5
            
            # 获取预测销量
            predicted_sales = pred.get('predicted_sales', 0)
            try:
                predicted_sales = float(predicted_sales)
            except:
                predicted_sales = 0.0
            
            # 计算新的增长率：(AI预测量-基期)/基期
            base_sales = base_sales_dict.get(category_id, 0.0)
            growth_rate = self.calculate_growth_rate(predicted_sales, base_sales)
            
            # 如果增长率计算失败或异常，使用备用方案
            if growth_rate == 0.05 and base_sales <= 0:
                # 基于置信度和需求等级生成合理的增长率（备用方案，不乘以100）
                demand_level = pred.get('demand_level', 'medium').lower()
                if demand_level == 'high':
                    growth_rate = confidence * 0.1 + 0.05  # 0.05-0.15 (5%-15%)
                elif demand_level == 'medium':
                    growth_rate = confidence * 0.08 + 0.02  # 0.02-0.10 (2%-10%)
                else:  # low
                    growth_rate = confidence * 0.05 + 0.01  # 0.01-0.06 (1%-6%)
                growth_rate = round(growth_rate, 3)
            
            demand_level = pred.get('demand_level', 'medium').lower()
            if demand_level not in ['high', 'medium', 'low']:
                demand_level = 'medium'
            
            enhanced_pred = {
                'category_id': category_id,
                'category_name': pred.get('category_name', '未知'),
                'predicted_sales': int(predicted_sales),
                'demand_level': demand_level,
                'growth_rate': growth_rate,
                'confidence': confidence,
                'ai_enhanced': True,
                'base_sales': base_sales  # 添加基期销量用于调试
            }
            
            enhanced_predictions.append(enhanced_pred)
        
        return enhanced_predictions

    # ===== 辅助函数 =====
    def format_time_period(self, time_data: Dict[str, Any]) -> str:
        """格式化时间范围描述"""
        if not time_data or not isinstance(time_data, dict):
            return "未知时间范围"
            
        if time_data.get('type') == 'today':
            return "今天"
        elif time_data.get('type') == 'yesterday':
            return "昨天"
        elif time_data.get('type') == 'recent':
            return f"最近{time_data.get('days', 7)}天"
        elif time_data.get('type') == 'this_month':
            return "本月"
        elif time_data.get('type') == 'this_year':
            return "今年"
        elif time_data.get('type') == 'month':
            return f"{time_data.get('year', datetime.now().year)}年{time_data.get('month')}月"
        return "未知时间范围"

    def get_categories_from_db(self, cursor: pymysql.cursors.Cursor) -> List[Dict[str, Any]]:
        """从数据库获取所有商品种类"""
        sql = "SELECT category_id, category_name FROM category WHERE status = 1"
        cursor.execute(sql)
        return cursor.fetchall()

    def match_category_by_query(self, query_text: str, categories: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """根据查询文本匹配商品种类"""
        query_lower = query_text.lower()
        for category in categories:
            if category['category_name'].lower() in query_lower:
                return category
        return None

# 创建全局AI服务实例
sales_ai_service = AIService()