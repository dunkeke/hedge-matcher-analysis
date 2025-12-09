import streamlit as st
import pandas as pd
import numpy as np
import io
import time
import warnings
from datetime import datetime, timedelta
from collections import deque
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots  # 添加这行
import re
import json

warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------
# 1. 核心匹配引擎 (完整集成)
# ---------------------------------------------------------

class HedgeMatchingEngine:
    """套保匹配引擎 - 完整版"""
    
    def __init__(self):
        self.df_paper = None
        self.df_physical = None
        self.df_paper_net = None
        self.df_relations = None
        self.df_physical_updated = None
        
    def clean_str(self, series):
        """清洗字符串"""
        return series.astype(str).str.strip().str.upper().replace('NAN', '')
    
    def standardize_month(self, series):
        """标准化月份格式"""
        s = series.astype(str).str.strip().str.upper()
        s = s.str.replace('-', ' ', regex=False).str.replace('/', ' ', regex=False)
        dates = pd.to_datetime(s, errors='coerce')
        result = dates.dt.strftime('%b %y').str.upper()
        mask_invalid = dates.isna()
        
        if mask_invalid.any():
            invalid = s[mask_invalid]
            def swap_if_match(val):
                m = re.match(r'^(\d{2})\s*([A-Z]{3})$', val)
                if m:
                    yr, mon = m.groups()
                    return f"{mon} {yr}"
                return val
            swapped = invalid.map(swap_if_match)
            swapped_dates = pd.to_datetime(swapped, errors='coerce')
            swapped_formatted = swapped_dates.dt.strftime('%b %y').str.upper()
            result.loc[mask_invalid & swapped_dates.notna()] = swapped_formatted.loc[swapped_dates.notna()]
            result.loc[mask_invalid & swapped_dates.isna()] = swapped.loc[swapped_dates.isna()]
        return result
    
    def calculate_net_positions(self, df_paper):
        """FIFO净仓计算"""
        st.info("🔄 执行纸货内部对冲 (FIFO Netting)...")
        progress_bar = st.progress(0)
        
        df_paper = df_paper.sort_values(by='Trade Date').reset_index(drop=True)
        df_paper['Group_Key'] = df_paper['Std_Commodity'] + "_" + df_paper['Month']
        records = df_paper.to_dict('records')
        groups = {}
        
        # 分组
        for i, row in enumerate(records):
            key = row['Group_Key']
            if key not in groups:
                groups[key] = []
            groups[key].append(i)
            if i % 100 == 0:
                progress_bar.progress(min(i / len(records) * 0.5, 0.5))
        
        # FIFO净额化
        group_count = 0
        total_groups = len(groups)
        for key, indices in groups.items():
            open_queue = deque()
            for idx in indices:
                row = records[idx]
                current_vol = row.get('Volume', 0)
                records[idx]['Net_Open_Vol'] = current_vol
                records[idx]['Closed_Vol'] = 0
                records[idx]['Close_Events'] = []
                
                if abs(current_vol) < 0.0001:
                    continue
                
                current_sign = 1 if current_vol > 0 else -1
                
                # 尝试与队列中的交易抵消
                while open_queue:
                    q_idx, q_vol, q_sign = open_queue[0]
                    if q_sign != current_sign:  # 方向相反才能抵消
                        offset = min(abs(current_vol), abs(q_vol))
                        current_vol -= (current_sign * offset)
                        q_vol -= (q_sign * offset)
                        
                        # 记录平仓事件
                        close_event = {
                            'Ref': str(records[idx].get('Recap No', '')),
                            'Date': records[idx].get('Trade Date'),
                            'Vol': offset,
                            'Price': records[idx].get('Price', 0)
                        }
                        records[q_idx]['Close_Events'].append(close_event)
                        records[q_idx]['Closed_Vol'] += offset
                        records[q_idx]['Net_Open_Vol'] = q_vol
                        records[idx]['Closed_Vol'] += offset
                        records[idx]['Net_Open_Vol'] = current_vol
                        
                        if abs(q_vol) < 0.0001:
                            open_queue.popleft()
                        else:
                            open_queue[0] = (q_idx, q_vol, q_sign)
                        
                        if abs(current_vol) < 0.0001:
                            break
                    else:
                        break
                
                # 剩余部分入队
                if abs(current_vol) > 0.0001:
                    open_queue.append((idx, current_vol, current_sign))
            
            group_count += 1
            progress_bar.progress(0.5 + (group_count / total_groups) * 0.5)
        
        progress_bar.progress(1.0)
        st.success(f"✅ 纸货内部对冲完成！共处理 {len(groups)} 个商品-月份组合")
        return pd.DataFrame(records)
    
    def match_hedges(self, df_physical, df_paper_net):
        """实货匹配"""
        st.info("🔄 开始实货匹配...")
        progress_bar = st.progress(0)
        
        hedge_relations = []
        active_paper = df_paper_net.copy()
        active_paper['Allocated_To_Phy'] = 0.0
        active_paper['_original_index'] = active_paper.index
        
        df_phy = df_physical.copy()
        df_phy['_orig_idx'] = df_phy.index
        
        # BRENT优先匹配
        if 'Pricing_Benchmark' in df_phy.columns:
            def bench_prio(x):
                x_str = str(x).upper()
                return 0 if 'BRENT' in x_str else 1
            df_phy['_priority'] = df_phy['Pricing_Benchmark'].apply(bench_prio)
            df_phy = df_phy.sort_values(by=['_priority', '_orig_idx']).reset_index(drop=True)
            df_phy = df_phy.drop(columns=['_priority'])
        else:
            df_phy = df_phy.reset_index(drop=True)
        
        total_cargos = len(df_phy)
        
        for idx, (_, cargo) in enumerate(df_phy.iterrows()):
            cargo_id = cargo.get('Cargo_ID')
            phy_vol = cargo.get('Unhedged_Volume', 0)
            
            if abs(phy_vol) < 0.0001:
                continue
            
            proxy = str(cargo.get('Hedge_Proxy', ''))
            target_month = cargo.get('Target_Contract_Month', None)
            phy_dir = cargo.get('Direction', 'Buy')
            desig_date = cargo.get('Designation_Date', pd.NaT)
            
            # 筛选候选交易
            candidates_df = active_paper[
                (active_paper['Std_Commodity'].str.contains(proxy, regex=False)) &
                (active_paper['Month'] == target_month)
            ].copy()
            
            if candidates_df.empty:
                continue
            
            # 时间排序：有指定日期按时间差，否则FIFO
            if pd.notna(desig_date) and not candidates_df['Trade Date'].isnull().all():
                candidates_df['Time_Lag_Days'] = (candidates_df['Trade Date'] - desig_date).dt.days
                candidates_df['Abs_Lag'] = candidates_df['Time_Lag_Days'].abs()
                candidates_df = candidates_df.sort_values(by=['Abs_Lag', 'Trade Date'])
            else:
                candidates_df['Time_Lag_Days'] = np.nan
                candidates_df = candidates_df.sort_values(by='Trade Date')
            
            # 分配匹配
            for _, ticket in candidates_df.iterrows():
                if abs(phy_vol) < 1:
                    break
                
                original_index = ticket['_original_index']
                curr_allocated = active_paper.at[original_index, 'Allocated_To_Phy']
                curr_total_vol = ticket.get('Volume', 0)
                avail = curr_total_vol - curr_allocated
                
                if abs(avail) < 0.0001:
                    continue
                
                alloc_amt_abs = abs(phy_vol) if abs(avail) >= abs(phy_vol) else abs(avail)
                alloc_amt = np.sign(avail) * alloc_amt_abs
                phy_vol -= alloc_amt_abs
                active_paper.at[original_index, 'Allocated_To_Phy'] += alloc_amt
                
                # 计算财务指标
                open_price = ticket.get('Price', 0)
                mtm_price = ticket.get('Mtm Price', open_price)  # 默认为开仓价
                total_pl_raw = ticket.get('Total P/L', 0)
                close_events = ticket.get('Close_Events', [])
                
                # 格式化平仓路径
                close_path_str = ""
                if close_events:
                    try:
                        sorted_events = sorted(close_events, key=lambda x: x['Date'] if pd.notna(x['Date']) else pd.Timestamp.min)
                        details = []
                        for e in sorted_events:
                            d_str = e['Date'].strftime('%Y-%m-%d') if pd.notna(e['Date']) else 'N/A'
                            p_str = f"@{e['Price']}" if pd.notna(e['Price']) else ""
                            details.append(f"[{d_str} Tkt#{e['Ref']} Vol:{e['Vol']:.0f} {p_str}]")
                        close_path_str = " -> ".join(details)
                    except:
                        close_path_str = str(close_events)
                
                # 计算分配比例
                ratio = abs(alloc_amt) / abs(curr_total_vol) if abs(curr_total_vol) > 0 else 0
                unrealized_mtm = (mtm_price - open_price) * alloc_amt
                allocated_total_pl = total_pl_raw * ratio
                
                hedge_relations.append({
                    'Cargo_ID': cargo_id,
                    'Proxy': proxy,
                    'Designation_Date': desig_date,
                    'Open_Date': ticket.get('Trade Date'),
                    'Time_Lag': ticket.get('Time_Lag_Days'),
                    'Ticket_ID': ticket.get('Recap No'),
                    'Month': ticket.get('Month'),
                    'Allocated_Vol': alloc_amt,
                    'Trade_Volume': ticket.get('Volume', 0),
                    'Trade_Net_Open': ticket.get('Net_Open_Vol', 0),
                    'Trade_Closed_Vol': ticket.get('Closed_Vol', 0),
                    'Open_Price': open_price,
                    'MTM_Price': mtm_price,
                    'Alloc_Unrealized_MTM': round(unrealized_mtm, 2),
                    'Alloc_Total_PL': round(allocated_total_pl, 2),
                    'Close_Path_Details': close_path_str,
                })
                
                # 更新实货未对冲量
                orig_idx = cargo.get('_orig_idx')
                if orig_idx in df_physical.index:
                    df_physical.at[orig_idx, 'Unhedged_Volume'] = phy_vol
            
            progress_bar.progress((idx + 1) / total_cargos)
        
        # 更新分配量
        cols_to_update = active_paper[['_original_index', 'Allocated_To_Phy']].set_index('_original_index')
        df_paper_net.update(cols_to_update)
        
        progress_bar.progress(1.0)
        df_relations = pd.DataFrame(hedge_relations)
        st.success(f"✅ 实货匹配完成！共生成 {len(df_relations)} 条匹配记录")
        
        return df_relations, df_physical
    
    def run_matching(self, df_paper_raw, df_physical_raw):
        """执行完整匹配流程"""
        # 数据预处理
        st.info("🔄 数据预处理中...")
        
        # 纸货预处理
        df_paper = df_paper_raw.copy()
        
        # 确保必要的列存在
        required_cols_paper = ['Trade Date', 'Volume', 'Commodity']
        for col in required_cols_paper:
            if col not in df_paper.columns:
                st.error(f"纸货数据缺少必要列: {col}")
                return None, None, None, None
        
        # 标准化处理
        df_paper['Trade Date'] = pd.to_datetime(df_paper['Trade Date'], errors='coerce')
        df_paper['Volume'] = pd.to_numeric(df_paper['Volume'], errors='coerce').fillna(0)
        df_paper['Std_Commodity'] = self.clean_str(df_paper['Commodity'])
        
        if 'Month' in df_paper.columns:
            df_paper['Month'] = self.standardize_month(df_paper['Month'])
        else:
            # 如果没有Month列，尝试从其他列推断或创建默认值
            df_paper['Month'] = df_paper['Trade Date'].dt.strftime('%b %y').str.upper()
        
        # 处理缺失字段
        if 'Recap No' not in df_paper.columns:
            df_paper['Recap No'] = [f"TKT-{i+1:04d}" for i in range(len(df_paper))]
        
        for col in ['Price', 'Mtm Price', 'Total P/L']:
            if col not in df_paper.columns:
                df_paper[col] = 0.0
        
        # 实货预处理
        df_physical = df_physical_raw.copy()
        
        # 标准化列名
        col_mapping = {
            'Target_Pricing_Month': 'Target_Contract_Month',
            'Month': 'Target_Contract_Month',
            'Hedge_Proxy': 'Hedge_Proxy',
            'Direction': 'Direction'
        }
        
        for old_col, new_col in col_mapping.items():
            if old_col in df_physical.columns and new_col not in df_physical.columns:
                df_physical[new_col] = df_physical[old_col]
        
        # 确保必要列
        if 'Volume' in df_physical.columns:
            df_physical['Volume'] = pd.to_numeric(df_physical['Volume'], errors='coerce').fillna(0)
            df_physical['Unhedged_Volume'] = df_physical['Volume']
        
        if 'Hedge_Proxy' in df_physical.columns:
            df_physical['Hedge_Proxy'] = self.clean_str(df_physical['Hedge_Proxy'])
        
        if 'Target_Contract_Month' in df_physical.columns:
            df_physical['Target_Contract_Month'] = self.standardize_month(df_physical['Target_Contract_Month'])
        
        # 指定日期
        date_cols = ['Designation_Date', 'Pricing_Start', 'Trade Date']
        for col in date_cols:
            if col in df_physical.columns:
                df_physical['Designation_Date'] = pd.to_datetime(df_physical[col], errors='coerce')
                break
        else:
            df_physical['Designation_Date'] = pd.NaT
        
        # 执行匹配
        self.df_paper_net = self.calculate_net_positions(df_paper)
        self.df_relations, self.df_physical_updated = self.match_hedges(df_physical, self.df_paper_net)
        
        return self.df_relations, self.df_physical_updated, self.df_paper_net, df_paper

# ---------------------------------------------------------
# 2. 分析模块 (基于真实匹配结果)
# ---------------------------------------------------------

class HedgeAnalysis:
    """套保分析模块"""
    
    def __init__(self, df_relations, df_physical, df_paper_net):
        self.df_relations = df_relations
        self.df_physical = df_physical
        self.df_paper_net = df_paper_net
        self.summary_stats = {}
        self.calculate_summary()
    
    def calculate_summary(self):
        """计算汇总统计"""
        if self.df_relations.empty:
            return
        
        try:
            # 匹配统计
            total_matched = abs(self.df_relations['Allocated_Vol']).sum() if 'Allocated_Vol' in self.df_relations.columns else 0
            total_physical = abs(self.df_physical['Volume']).sum() if 'Volume' in self.df_physical.columns else 0
            match_rate = (total_matched / total_physical * 100) if total_physical > 0 else 0
            
            # 财务统计
            total_pl = self.df_relations['Alloc_Total_PL'].sum() if 'Alloc_Total_PL' in self.df_relations.columns else 0
            total_unrealized = self.df_relations['Alloc_Unrealized_MTM'].sum() if 'Alloc_Unrealized_MTM' in self.df_relations.columns else 0
            
            # 数量统计
            matched_cargos = self.df_relations['Cargo_ID'].nunique() if 'Cargo_ID' in self.df_relations.columns else 0
            total_cargos = self.df_physical['Cargo_ID'].nunique() if 'Cargo_ID' in self.df_physical.columns else 0
            total_tickets = len(self.df_relations)
            
            # 时间统计
            if 'Time_Lag' in self.df_relations.columns:
                time_lag_abs = self.df_relations['Time_Lag'].abs()
                avg_time_lag = time_lag_abs.mean() if not time_lag_abs.isna().all() else 0
                std_time_lag = time_lag_abs.std() if not time_lag_abs.isna().all() else 0
            else:
                avg_time_lag = std_time_lag = 0
            
            self.summary_stats = {
                'total_matched': total_matched,
                'total_physical': total_physical,
                'match_rate': match_rate,
                'total_pl': total_pl,
                'total_unrealized': total_unrealized,
                'matched_cargos': matched_cargos,
                'total_cargos': total_cargos,
                'total_tickets': total_tickets,
                'avg_time_lag': avg_time_lag,
                'std_time_lag': std_time_lag
            }
        except Exception as e:
            st.warning(f"计算汇总统计时出错: {e}")
            self.summary_stats = {
                'total_matched': 0,
                'total_physical': 0,
                'match_rate': 0,
                'total_pl': 0,
                'total_unrealized': 0,
                'matched_cargos': 0,
                'total_cargos': 0,
                'total_tickets': 0,
                'avg_time_lag': 0,
                'std_time_lag': 0
            }
    
    def create_summary_metrics(self):
        """创建概览指标卡片"""
        stats = self.summary_stats
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("📊 匹配率", f"{stats['match_rate']:.1f}%", 
                     delta=f"{stats['total_matched']:,.0f}/{stats['total_physical']:,.0f}")
        
        with col2:
            coverage = (stats['matched_cargos'] / stats['total_cargos'] * 100) if stats['total_cargos'] > 0 else 0
            st.metric("📦 匹配覆盖率", f"{coverage:.1f}%",
                     delta=f"{stats['matched_cargos']}/{stats['total_cargos']}")
        
        with col3:
            st.metric("💰 总P/L", f"${stats['total_pl']:,.2f}",
                     delta=f"未实现: ${stats['total_unrealized']:,.2f}")
        
        with col4:
            st.metric("⏱️ 平均时间差", f"{stats['avg_time_lag']:.1f}天",
                     delta=f"±{stats['std_time_lag']:.1f}天")
    
    def create_match_volume_chart(self):
        """匹配量分布图表"""
        try:
            if self.df_relations.empty or 'Allocated_Vol' not in self.df_relations.columns:
                return None
            
            # 按Cargo_ID汇总
            cargo_summary = self.df_relations.copy()
            cargo_summary['Allocated_Vol_Abs'] = abs(cargo_summary['Allocated_Vol'])
            
            if 'Cargo_ID' not in cargo_summary.columns:
                return None
            
            cargo_group = cargo_summary.groupby('Cargo_ID')['Allocated_Vol_Abs'].sum().reset_index()
            
            # 按匹配量排序，取前20
            top_cargos = cargo_group.sort_values('Allocated_Vol_Abs', ascending=False).head(20)
            
            fig = px.bar(top_cargos, 
                         x='Cargo_ID', y='Allocated_Vol_Abs',
                         title='📈 各Cargo_ID匹配量TOP20',
                         labels={'Allocated_Vol_Abs': '匹配量', 'Cargo_ID': '实货编号'},
                         color='Allocated_Vol_Abs',
                         color_continuous_scale='Viridis')
            fig.update_layout(xaxis_tickangle=-45)
            return fig
        except Exception as e:
            st.warning(f"创建匹配量图表时出错: {e}")
            return None
    
    def create_pl_analysis_chart(self):
        """P/L分析图表"""
        try:
            if self.df_relations.empty or 'Alloc_Total_PL' not in self.df_relations.columns:
                return None
            
            # 使用更简单的图表，避免复杂子图
            fig = px.histogram(self.df_relations, 
                              x='Alloc_Total_PL',
                              nbins=30,
                              title='💰 P/L分布直方图',
                              labels={'Alloc_Total_PL': 'P/L值'})
            fig.add_vline(x=0, line_dash="dash", line_color="red")
            
            # 添加箱线图显示统计信息
            fig2 = px.box(self.df_relations, 
                         y='Alloc_Total_PL',
                         title='📊 P/L统计箱线图')
            
            return fig, fig2
        except Exception as e:
            st.warning(f"创建P/L图表时出错: {e}")
            return None, None
    
    def create_simple_pl_chart(self):
        """简化的P/L图表"""
        try:
            if self.df_relations.empty or 'Alloc_Total_PL' not in self.df_relations.columns:
                return None
            
            fig = go.Figure()
            
            # 添加直方图
            fig.add_trace(go.Histogram(
                x=self.df_relations['Alloc_Total_PL'],
                nbinsx=30,
                name='P/L分布',
                marker_color='skyblue'
            ))
            
            # 添加零线
            fig.add_vline(x=0, line_dash="dash", line_color="red")
            
            fig.update_layout(
                title='💰 P/L分布分析',
                xaxis_title='P/L值',
                yaxis_title='频数',
                showlegend=False
            )
            
            return fig
        except Exception as e:
            st.warning(f"创建简化P/L图表时出错: {e}")
            return None
    
    def create_time_analysis_chart(self):
        """时间分析图表"""
        try:
            if self.df_relations.empty or 'Time_Lag' not in self.df_relations.columns:
                return None
            
            time_lag_data = self.df_relations['Time_Lag'].dropna()
            if time_lag_data.empty:
                return None
            
            fig = px.histogram(time_lag_data,
                             nbinsx=30,
                             title='⏱️ 时间差分布',
                             labels={'value': '时间差(天)'})
            fig.add_vline(x=0, line_dash="dash", line_color="green",
                         annotation_text="完美匹配")
            
            return fig
        except Exception as e:
            st.warning(f"创建时间分析图表时出错: {e}")
            return None
    
    def create_price_analysis_chart(self):
        """价格分析图表"""
        try:
            if self.df_relations.empty:
                return None
            
            required_cols = ['Open_Price', 'MTM_Price', 'Allocated_Vol']
            missing_cols = [col for col in required_cols if col not in self.df_relations.columns]
            
            if missing_cols:
                st.info(f"缺少价格分析所需列: {missing_cols}")
                return None
            
            fig = px.scatter(self.df_relations, 
                            x='Open_Price', 
                            y='MTM_Price',
                            size=abs(self.df_relations['Allocated_Vol']),
                            color='Alloc_Total_PL' if 'Alloc_Total_PL' in self.df_relations.columns else None,
                            title='💹 开仓价 vs 当前价分析',
                            labels={'Open_Price': '开仓价', 'MTM_Price': '当前价'},
                            hover_data=['Cargo_ID', 'Ticket_ID', 'Allocated_Vol'] if 'Cargo_ID' in self.df_relations.columns else [])
            
            # 添加平价线
            min_price = min(self.df_relations['Open_Price'].min(), self.df_relations['MTM_Price'].min())
            max_price = max(self.df_relations['Open_Price'].max(), self.df_relations['MTM_Price'].max())
            
            fig.add_trace(go.Scatter(x=[min_price, max_price],
                                    y=[min_price, max_price],
                                    mode='lines',
                                    name='平价线',
                                    line=dict(color='red', dash='dash')))
            
            return fig
        except Exception as e:
            st.warning(f"创建价格分析图表时出错: {e}")
            return None
    
    def create_month_distribution_chart(self):
        """月份分布图表"""
        try:
            if self.df_relations.empty or 'Month' not in self.df_relations.columns:
                return None
            
            month_summary = self.df_relations.copy()
            month_summary['Allocated_Vol_Abs'] = abs(month_summary['Allocated_Vol'])
            month_group = month_summary.groupby('Month')['Allocated_Vol_Abs'].sum().reset_index()
            
            fig = px.bar(month_group.sort_values('Allocated_Vol_Abs', ascending=False),
                         x='Month', y='Allocated_Vol_Abs',
                         title='📅 各月份匹配量分布',
                         labels={'Allocated_Vol_Abs': '匹配量', 'Month': '合约月份'},
                         color='Allocated_Vol_Abs',
                         color_continuous_scale='Plasma')
            fig.update_layout(xaxis_tickangle=-45)
            
            return fig
        except Exception as e:
            st.warning(f"创建月份分布图表时出错: {e}")
            return None
    
    def create_match_detail_table(self, max_rows=50):
        """创建匹配明细表"""
        try:
            if self.df_relations.empty:
                return pd.DataFrame()
            
            # 选择要显示的列
            display_cols = []
            possible_cols = ['Cargo_ID', 'Ticket_ID', 'Month', 'Allocated_Vol',
                            'Open_Price', 'MTM_Price', 'Alloc_Total_PL',
                            'Alloc_Unrealized_MTM', 'Time_Lag', 'Proxy']
            
            for col in possible_cols:
                if col in self.df_relations.columns:
                    display_cols.append(col)
            
            if not display_cols:
                return pd.DataFrame()
            
            # 格式化数字
            formatted_df = self.df_relations[display_cols].copy()
            
            # 数字格式化函数
            def format_number(x):
                if isinstance(x, (int, float, np.integer, np.floating)):
                    return f"{x:,.2f}"
                return x
            
            # 格式化数值列
            num_cols = ['Allocated_Vol', 'Open_Price', 'MTM_Price', 
                       'Alloc_Total_PL', 'Alloc_Unrealized_MTM']
            for col in num_cols:
                if col in formatted_df.columns:
                    formatted_df[col] = formatted_df[col].apply(format_number)
            
            return formatted_df.head(max_rows)
        except Exception as e:
            st.warning(f"创建匹配明细表时出错: {e}")
            return pd.DataFrame()
    
    def create_risk_metrics(self):
        """风险指标计算"""
        try:
            if self.df_relations.empty or 'Alloc_Total_PL' not in self.df_relations.columns:
                return {}
            
            risk_metrics = {}
            pl_series = self.df_relations['Alloc_Total_PL']
            
            # VaR计算 (95%置信水平)
            if len(pl_series) > 1:
                var_95 = np.percentile(pl_series, 5)  # 95% VaR
                cvar_95 = pl_series[pl_series <= var_95].mean() if len(pl_series[pl_series <= var_95]) > 0 else 0
                risk_metrics['VaR_95'] = var_95
                risk_metrics['CVaR_95'] = cvar_95
                risk_metrics['PL_StdDev'] = pl_series.std()
                risk_metrics['PL_Max'] = pl_series.max()
                risk_metrics['PL_Min'] = pl_series.min()
                
                # 夏普比率 (假设无风险利率为0)
                avg_pl = pl_series.mean()
                std_pl = pl_series.std()
                risk_metrics['Sharpe_Ratio'] = avg_pl / std_pl if std_pl != 0 else 0
                
                # 最大回撤
                pl_cumulative = pl_series.cumsum()
                running_max = pl_cumulative.expanding().max()
                drawdown = (pl_cumulative - running_max) / running_max * 100
                risk_metrics['Max_Drawdown'] = drawdown.min() if not drawdown.empty else 0
            
            return risk_metrics
        except Exception as e:
            st.warning(f"计算风险指标时出错: {e}")
            return {}

# ---------------------------------------------------------
# 3. Streamlit 主应用
# ---------------------------------------------------------

def main():
    st.set_page_config(
        page_title="实纸货套保匹配分析系统",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # 自定义CSS
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #374151;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
        border-bottom: 2px solid #E5E7EB;
        padding-bottom: 0.5rem;
    }
    .success-box {
        background-color: #D1FAE5;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #10B981;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #DBEAFE;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #3B82F6;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #FEF3C7;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #F59E0B;
        margin: 1rem 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # 标题
    st.markdown('<h1 class="main-header">📈 实纸货套保匹配分析系统</h1>', unsafe_allow_html=True)
    st.markdown("### 专业套保匹配与风险分析工具 | 基于真实匹配数据")
    
    # 初始化session state
    if 'engine' not in st.session_state:
        st.session_state.engine = HedgeMatchingEngine()
    if 'analysis' not in st.session_state:
        st.session_state.analysis = None
    if 'matching_complete' not in st.session_state:
        st.session_state.matching_complete = False
    
    # 侧边栏
    with st.sidebar:
        st.markdown("### 📁 数据上传")
        
        paper_file = st.file_uploader(
            "纸货数据文件",
            type=["csv", "xlsx", "xls"],
            key="paper_uploader",
            help="支持CSV/Excel格式，需包含Trade Date, Volume, Commodity等字段"
        )
        
        physical_file = st.file_uploader(
            "实货数据文件",
            type=["csv", "xlsx", "xls"],
            key="physical_uploader",
            help="支持CSV/Excel格式，需包含Cargo_ID, Volume, Hedge_Proxy等字段"
        )
        
        st.markdown("---")
        st.markdown("### ⚙️ 分析设置")
        
        show_charts = st.checkbox("显示分析图表", value=True)
        show_risk = st.checkbox("显示风险指标", value=True)
        max_rows = st.slider("表格显示行数", 10, 200, 50)
        
        st.markdown("---")
        
        if st.button("🔄 重置所有数据", type="secondary"):
            st.session_state.engine = HedgeMatchingEngine()
            st.session_state.analysis = None
            st.session_state.matching_complete = False
            st.rerun()
    
    # 主内容区
    if paper_file is not None and physical_file is not None:
        # 读取数据
        try:
            # 读取纸货数据
            if paper_file.name.endswith(('.xlsx', '.xls')):
                df_paper_raw = pd.read_excel(paper_file)
            else:
                df_paper_raw = pd.read_csv(paper_file)
            
            # 读取实货数据
            if physical_file.name.endswith(('.xlsx', '.xls')):
                df_physical_raw = pd.read_excel(physical_file)
            else:
                df_physical_raw = pd.read_csv(physical_file)
            
            # 显示数据预览
            with st.expander("📋 原始数据预览", expanded=False):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"**纸货数据** ({len(df_paper_raw)}行, {len(df_paper_raw.columns)}列)")
                    st.dataframe(df_paper_raw.head(10), use_container_width=True)
                    st.caption(f"字段: {', '.join(df_paper_raw.columns.tolist()[:8])}...")
                
                with col2:
                    st.markdown(f"**实货数据** ({len(df_physical_raw)}行, {len(df_physical_raw.columns)}列)")
                    st.dataframe(df_physical_raw.head(10), use_container_width=True)
                    st.caption(f"字段: {', '.join(df_physical_raw.columns.tolist()[:8])}...")
            
            # 执行匹配按钮
            if st.button("🚀 执行套保匹配", type="primary", use_container_width=True):
                with st.spinner("正在执行套保匹配，请稍候..."):
                    try:
                        # 执行匹配
                        df_relations, df_physical_updated, df_paper_net, df_paper_processed = st.session_state.engine.run_matching(
                            df_paper_raw, df_physical_raw
                        )
                        
                        if df_relations is not None and not df_relations.empty:
                            # 创建分析模块
                            st.session_state.analysis = HedgeAnalysis(
                                df_relations, df_physical_updated, df_paper_net
                            )
                            st.session_state.matching_complete = True
                            
                            # 显示匹配成功信息
                            st.markdown('<div class="success-box">✅ 套保匹配成功完成！</div>', unsafe_allow_html=True)
                            
                            # 显示匹配过程数据
                            with st.expander("📊 匹配过程数据", expanded=False):
                                tab1, tab2, tab3 = st.tabs(["纸货净仓", "实货更新", "匹配关系"])
                                
                                with tab1:
                                    if df_paper_net is not None:
                                        st.dataframe(df_paper_net.head(20), use_container_width=True)
                                        st.caption(f"纸货净仓数据 ({len(df_paper_net)}行)")
                                    else:
                                        st.info("无纸货净仓数据")
                                
                                with tab2:
                                    if df_physical_updated is not None:
                                        st.dataframe(df_physical_updated.head(20), use_container_width=True)
                                        st.caption(f"更新后实货数据 ({len(df_physical_updated)}行)")
                                    else:
                                        st.info("无实货更新数据")
                                
                                with tab3:
                                    if df_relations is not None:
                                        st.dataframe(df_relations.head(20), use_container_width=True)
                                        st.caption(f"匹配关系数据 ({len(df_relations)}行)")
                                    else:
                                        st.info("无匹配关系数据")
                        else:
                            st.markdown('<div class="warning-box">⚠️ 匹配完成但未生成匹配记录，请检查数据格式和内容</div>', unsafe_allow_html=True)
                            
                    except Exception as e:
                        st.error(f"匹配过程中出现错误: {str(e)}")
                        st.exception(e)
        
        except Exception as e:
            st.error(f"数据读取错误: {str(e)}")
            st.info("请确保上传的文件格式正确，并包含必要的字段。")
    
    # 显示分析结果
    if st.session_state.matching_complete and st.session_state.analysis is not None:
        st.markdown("---")
        st.markdown('<h2 class="sub-header">📊 匹配分析结果</h2>', unsafe_allow_html=True)
        
        analysis = st.session_state.analysis
        
        # 检查是否有匹配数据
        if analysis.df_relations.empty:
            st.warning("⚠️ 匹配结果为空，无法进行分析")
            return
        
        # 1. 概览指标
        analysis.create_summary_metrics()
        
        # 2. 匹配明细表
        st.markdown('<h3 class="sub-header">📋 匹配明细表</h3>', unsafe_allow_html=True)
        detailed_table = analysis.create_match_detail_table(max_rows)
        
        if not detailed_table.empty:
            st.dataframe(detailed_table, use_container_width=True)
            st.caption(f"显示前 {len(detailed_table)} 条记录，共 {len(analysis.df_relations)} 条匹配记录")
        else:
            st.info("无匹配明细数据可显示")
        
        # 3. 分析图表
        if show_charts and not analysis.df_relations.empty:
            st.markdown('<h3 class="sub-header">📈 可视化分析</h3>', unsafe_allow_html=True)
            
            # 图表选项卡
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "📊 匹配量分析", "💰 P/L分析", 
                "⏱️ 时间分析", "💹 价格分析", "📅 月份分布"
            ])
            
            with tab1:
                fig1 = analysis.create_match_volume_chart()
                if fig1:
                    st.plotly_chart(fig1, use_container_width=True)
                else:
                    st.info("无匹配量数据可用于图表分析")
            
            with tab2:
                fig2 = analysis.create_simple_pl_chart()
                if fig2:
                    st.plotly_chart(fig2, use_container_width=True)
                    
                    # 显示P/L统计数据
                    if 'Alloc_Total_PL' in analysis.df_relations.columns:
                        pl_stats = analysis.df_relations['Alloc_Total_PL'].describe()
                        st.dataframe(pl_stats, use_container_width=True)
                else:
                    st.info("无P/L数据可用于图表分析")
            
            with tab3:
                fig3 = analysis.create_time_analysis_chart()
                if fig3:
                    st.plotly_chart(fig3, use_container_width=True)
                    
                    # 显示时间差统计数据
                    if 'Time_Lag' in analysis.df_relations.columns:
                        time_stats = analysis.df_relations['Time_Lag'].describe()
                        st.dataframe(time_stats, use_container_width=True)
                else:
                    st.info("无时间差数据可用于图表分析")
            
            with tab4:
                fig4 = analysis.create_price_analysis_chart()
                if fig4:
                    st.plotly_chart(fig4, use_container_width=True)
                    
                    # 显示价格统计数据
                    if 'Open_Price' in analysis.df_relations.columns and 'MTM_Price' in analysis.df_relations.columns:
                        price_stats = pd.DataFrame({
                            'Open_Price': analysis.df_relations['Open_Price'].describe(),
                            'MTM_Price': analysis.df_relations['MTM_Price'].describe()
                        }).T
                        st.dataframe(price_stats, use_container_width=True)
                else:
                    st.info("无价格数据可用于图表分析")
            
            with tab5:
                fig5 = analysis.create_month_distribution_chart()
                if fig5:
                    st.plotly_chart(fig5, use_container_width=True)
                    
                    # 显示月份统计数据
                    if 'Month' in analysis.df_relations.columns:
                        month_stats = analysis.df_relations['Month'].value_counts()
                        st.dataframe(month_stats, use_container_width=True)
                else:
                    st.info("无月份数据可用于图表分析")
        
        # 4. 风险指标
        if show_risk and not analysis.df_relations.empty:
            st.markdown('<h3 class="sub-header">⚠️ 风险指标分析</h3>', unsafe_allow_html=True)
            
            risk_metrics = analysis.create_risk_metrics()
            
            if risk_metrics:
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("VaR (95%)", f"${risk_metrics.get('VaR_95', 0):,.2f}")
                
                with col2:
                    st.metric("CVaR (95%)", f"${risk_metrics.get('CVaR_95', 0):,.2f}")
                
                with col3:
                    st.metric("夏普比率", f"{risk_metrics.get('Sharpe_Ratio', 0):.2f}")
                
                with col4:
                    st.metric("最大回撤", f"{risk_metrics.get('Max_Drawdown', 0):.1f}%")
                
                # 详细风险指标表格
                with st.expander("查看详细风险指标"):
                    risk_df = pd.DataFrame.from_dict(risk_metrics, orient='index', columns=['值'])
                    st.dataframe(risk_df.style.format("{:,.2f}"), use_container_width=True)
            else:
                st.info("无法计算风险指标，可能需要更多数据")
        
        # 5. 数据导出
        st.markdown("---")
        st.markdown('<h3 class="sub-header">💾 数据导出</h3>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # 导出匹配结果
            if not analysis.df_relations.empty:
                csv_data = analysis.df_relations.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 下载匹配结果",
                    data=csv_data,
                    file_name=f"hedge_matching_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
        
        with col2:
            # 导出分析报告
            report_data = {
                "匹配统计": analysis.summary_stats,
                "生成时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "数据量": {
                    "匹配记录数": len(analysis.df_relations),
                    "实货记录数": len(analysis.df_physical),
                    "纸货记录数": len(analysis.df_paper_net) if analysis.df_paper_net is not None else 0
                }
            }
            
            report_json = json.dumps(report_data, indent=2, default=str, ensure_ascii=False)
            st.download_button(
                label="📄 下载分析报告",
                data=report_json.encode('utf-8'),
                file_name=f"hedge_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True
            )
        
        with col3:
            # 导出所有数据
            @st.cache_data
            def convert_to_excel(df_dict):
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    for sheet_name, df in df_dict.items():
                        if df is not None and not df.empty:
                            df.to_excel(writer, sheet_name=sheet_name, index=False)
                return output.getvalue()
            
            if analysis.df_relations is not None:
                excel_data = convert_to_excel({
                    "匹配结果": analysis.df_relations,
                    "实货数据": analysis.df_physical,
                    "纸货净仓": analysis.df_paper_net
                })
                
                st.download_button(
                    label="📊 下载完整数据",
                    data=excel_data,
                    file_name=f"hedge_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )
    
    else:
        # 欢迎页面
        if not (paper_file and physical_file):
            st.markdown("---")
            st.markdown('<div class="info-box">👈 请在左侧上传纸货和实货数据文件开始分析</div>', unsafe_allow_html=True)
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown("""
                ### 🎯 系统工作流程
                
                1. **数据上传**
                   - 纸货交易数据 (包含交易日期、交易量、商品、价格等)
                   - 实货持仓数据 (包含Cargo_ID、交易量、套保代理、目标月份等)
                
                2. **智能匹配**
                   - FIFO内部对冲：先对纸货进行净额化
                   - 实货匹配：基于品种、月份、时间的智能匹配
                   - BRENT优先：优先匹配BRENT基准的交易
                
                3. **深度分析**
                   - 匹配率与覆盖率分析
                   - P/L与MTM分析
                   - 时间差与效率分析
                   - 风险指标计算 (VaR、夏普比率等)
                
                4. **数据导出**
                   - 匹配结果CSV
                   - 分析报告JSON
                   - 完整数据Excel
                """)
            
            with col2:
                st.markdown("""
                ### 📋 数据要求
                
                **纸货数据必需字段:**
                - `Trade Date`: 交易日期
                - `Volume`: 交易量 (正买负卖)
                - `Commodity`: 商品品种
                
                **实货数据必需字段:**
                - `Cargo_ID`: 实货编号
                - `Volume`: 交易量
                - `Hedge_Proxy`: 套保代理
                
                **可选字段:**
                - `Month`: 合约月份
                - `Price`: 交易价格
                - `Target_Contract_Month`: 目标月份
                - `Designation_Date`: 指定日期
                """)

if __name__ == "__main__":
    main()
