import pandas as pd
import numpy as np
import os
import time
import warnings
from datetime import datetime
from collections import deque

warnings.filterwarnings("ignore", category=UserWarning)

# ==============================================================================
# 基础工具 (Utils)
# ==============================================================================

def clean_str(series):
    """字符串清洗：去空、转大写"""
    return series.astype(str).str.strip().str.upper().replace('NAN', '')

def standardize_month_vectorized(series):
    """批量标准化月份格式"""
    s = series.astype(str).str.strip().str.upper()
    s = s.replace('NAN', '')
    s = s.str.replace('-', ' ', regex=False).str.replace('/', ' ', regex=False)
    dates = pd.to_datetime(s, errors='coerce')
    return dates.dt.strftime('%b %y').str.upper().fillna(s)

def read_file_fast(file_path_or_buffer):
    """
    通用读取函数，支持路径字符串或文件对象 (BytesIO)
    """
    # 尝试作为 Excel 读取
    try:
        # 如果是文件对象，由于 pd.read_excel 可能读取后指针移位，需谨慎
        if hasattr(file_path_or_buffer, 'seek'):
            file_path_or_buffer.seek(0)
        return pd.read_excel(file_path_or_buffer)
    except:
        pass
    
    # 尝试作为 CSV 读取 (多种编码)
    encodings = ['utf-8', 'gbk', 'gb18030', 'latin1']
    for enc in encodings:
        try:
            if hasattr(file_path_or_buffer, 'seek'):
                file_path_or_buffer.seek(0)
            return pd.read_csv(file_path_or_buffer, encoding=enc)
        except:
            continue
            
    raise ValueError("无法识别文件格式或编码")

# ==============================================================================
# 数据加载与清洗 (Data Loading)
# ==============================================================================

def load_data_v19(paper_file, phys_file):
    """
    加载并清洗数据，返回 df_p, df_ph
    """
    df_p = read_file_fast(paper_file)
    df_ph = read_file_fast(phys_file)

    # --- 清除列名空格 ---
    df_p.columns = df_p.columns.str.strip()
    df_ph.columns = df_ph.columns.str.strip()

    # --- 纸货清洗 ---
    df_p['Trade Date'] = pd.to_datetime(df_p['Trade Date'], errors='coerce')
    df_p['Volume'] = pd.to_numeric(df_p['Volume'], errors='coerce').fillna(0)
    df_p['Std_Commodity'] = clean_str(df_p['Commodity'])
    
    if 'Month' in df_p.columns:
        df_p['Month'] = standardize_month_vectorized(df_p['Month'])
    else:
        df_p['Month'] = ''
        
    if 'Recap No' not in df_p.columns:
        df_p['Recap No'] = df_p.index.astype(str)
    
    # 补全财务字段
    for col in ['Price', 'Mtm Price', 'Total P/L']:
        if col not in df_p.columns: df_p[col] = 0

    # --- 实货清洗 ---
    col_map = {
        'Target_Pricing_Month': 'Target_Contract_Month', 
        'Target Pricing Month': 'Target_Contract_Month', 
        'Month': 'Target_Contract_Month'
    }
    df_ph.rename(columns=col_map, inplace=True)
    
    df_ph['Volume'] = pd.to_numeric(df_ph['Volume'], errors='coerce').fillna(0)
    df_ph['Unhedged_Volume'] = df_ph['Volume']
    df_ph['Hedge_Proxy'] = clean_str(df_ph['Hedge_Proxy']) if 'Hedge_Proxy' in df_ph.columns else ''
    df_ph['Pricing_Benchmark'] = clean_str(df_ph['Pricing_Benchmark'])
    
    if 'Target_Contract_Month' in df_ph.columns:
        df_ph['Target_Contract_Month'] = standardize_month_vectorized(df_ph['Target_Contract_Month'])
    
    # 处理指定日
    if 'Designation_Date' in df_ph.columns:
        df_ph['Designation_Date'] = pd.to_datetime(df_ph['Designation_Date'], errors='coerce')
    elif 'Pricing_Start' in df_ph.columns:
        df_ph['Designation_Date'] = pd.to_datetime(df_ph['Pricing_Start'], errors='coerce')
    else:
        df_ph['Designation_Date'] = pd.NaT

    return df_p, df_ph

# ==============================================================================
# 计算逻辑 (Business Logic)
# ==============================================================================

def calculate_net_positions_corrected(df_paper):
    """Step 1: 纸货内部 FIFO 净仓计算"""
    df_paper = df_paper.sort_values(by='Trade Date').reset_index(drop=True)
    df_paper['Group_Key'] = df_paper['Std_Commodity'] + "_" + df_paper['Month']
    
    records = df_paper.to_dict('records')
    groups = {}
    for i, row in enumerate(records):
        key = row['Group_Key']
        if key not in groups: groups[key] = []
        groups[key].append(i)
    
    for key, indices in groups.items():
        open_queue = deque()
        for idx in indices:
            row = records[idx]
            current_vol = row.get('Volume', 0)
            
            records[idx]['Net_Open_Vol'] = current_vol
            records[idx]['Closed_Vol'] = 0
            records[idx]['Close_Events'] = [] 
            
            if abs(current_vol) < 0.0001: continue
            current_sign = 1 if current_vol > 0 else -1
            
            while open_queue:
                q_idx, q_vol, q_sign = open_queue[0]
                if q_sign != current_sign:
                    offset = min(abs(current_vol), abs(q_vol))
                    
                    close_event = {
                        'Ref': str(records[idx].get('Recap No', '')),
                        'Date': records[idx].get('Trade Date'),
                        'Vol': offset,
                        'Price': records[idx].get('Price', 0)
                    }
                    records[q_idx]['Close_Events'].append(close_event)
                    
                    current_vol -= (current_sign * offset)
                    q_vol -= (q_sign * offset)
                    
                    records[q_idx]['Closed_Vol'] += offset
                    records[q_idx]['Net_Open_Vol'] = q_vol
                    records[idx]['Closed_Vol'] += offset
                    records[idx]['Net_Open_Vol'] = current_vol
                    
                    if abs(q_vol) < 0.0001: open_queue.popleft()
                    else: open_queue[0] = (q_idx, q_vol, q_sign)
                    
                    if abs(current_vol) < 0.0001: break
                else:
                    break
            
            if abs(current_vol) > 0.0001:
                open_queue.append((idx, current_vol, current_sign))
                
    return pd.DataFrame(records)

def format_close_details(events):
    if not events: return "", 0
    details = []
    total_vol = 0
    total_val = 0
    sorted_events = sorted(events, key=lambda x: x['Date'] if pd.notna(x['Date']) else pd.Timestamp.min)
    for e in sorted_events:
        d_str = e['Date'].strftime('%Y-%m-%d') if pd.notna(e['Date']) else 'N/A'
        p_str = f"@{e['Price']}" if pd.notna(e['Price']) else ""
        details.append(f"[{d_str} #{e['Ref']} V:{e['Vol']:.0f} {p_str}]")
        if pd.notna(e['Price']):
            total_vol += e['Vol']
            total_val += (e['Vol'] * e['Price'])
    return " -> ".join(details), total_vol

def auto_match_hedges(physical_df, paper_df):
    """Step 2: 实货匹配 (Safe Update Version)"""
    hedge_relations = []

    default_match_start_date = pd.NaT
    trade_years = paper_df['Trade Date'].dropna().dt.year
    if not trade_years.empty:
        default_match_start_date = pd.Timestamp(year=int(trade_years.max()), month=11, day=13)

    # 强制初始化
    paper_df['Allocated_To_Phy'] = 0.0

    # 索引构建
    active_paper = paper_df[paper_df['Net_Open_Vol'] > 0.0001].copy()
    active_paper['Allocated_To_Phy'] = 0.0
    active_paper['_original_index'] = active_paper.index

    def filter_close_events(events, start_date):
        if pd.isna(start_date):
            return list(events or [])
        return [
            event for event in (events or [])
            if pd.notna(event.get('Date')) and event.get('Date') >= start_date
        ]

    close_event_pool = {}
    for _, ticket in active_paper.iterrows():
        events = ticket.get('Close_Events', []) or []
        if events:
            sorted_events = sorted(
                events,
                key=lambda x: x['Date'] if pd.notna(x.get('Date')) else pd.Timestamp.min
            )
            close_event_pool[ticket['_original_index']] = deque(
                {
                    'remaining': float(e.get('Vol', 0)),
                    'price': e.get('Price', 0),
                    'date': e.get('Date')
                }
                for e in sorted_events
                if abs(e.get('Vol', 0)) > 0.0001
            )

    # 实货排序
    physical_df['Sort_Date'] = physical_df['Designation_Date'].fillna(pd.Timestamp.max)
    priority_cargos = {'PHY-2026-004', 'PHY-2026-005'}
    physical_df['Benchmark_Priority'] = physical_df['Pricing_Benchmark'].astype(str).str.upper().str.contains('BRENT')
    physical_df['Cargo_Priority'] = physical_df['Cargo_ID'].astype(str).str.upper().isin(priority_cargos)
    physical_df_sorted = physical_df.sort_values(
        by=['Benchmark_Priority', 'Cargo_Priority', 'Sort_Date', 'Cargo_ID'],
        ascending=[False, False, True, True]
    )

    for idx, cargo in physical_df_sorted.iterrows():
        cargo_id = cargo['Cargo_ID']
        phy_vol = cargo['Unhedged_Volume']
        proxy = str(cargo['Hedge_Proxy'])
        target_month = cargo.get('Target_Contract_Month', None)
        desig_date = cargo.get('Designation_Date', pd.NaT)
        if pd.notna(desig_date):
            cargo_start_date = pd.Timestamp(year=desig_date.year, month=11, day=13)
        else:
            cargo_start_date = default_match_start_date
        benchmark = str(cargo.get('Pricing_Benchmark', '')).upper()
        proxy_candidates = [proxy] if proxy else []
        if 'BRENT' in benchmark:
            proxy_candidates = proxy_candidates or []
            proxy_candidates.extend(['PHY-2026-001', 'PHY-2026-002', 'PHY-2026-003'])

        for proxy_value in proxy_candidates:
            if abs(phy_vol) < 1:
                break

            mask = (
                (active_paper['Std_Commodity'].str.contains(proxy_value, regex=False)) &
                (active_paper['Month'] == target_month)
            )
            if pd.notna(cargo_start_date):
                mask &= active_paper['Trade Date'] >= cargo_start_date
            candidates_df = active_paper[mask].copy()

            if candidates_df.empty:
                continue

            # 排序
            if pd.notna(desig_date) and not candidates_df['Trade Date'].isnull().all():
                candidates_df['Time_Lag_Days'] = (candidates_df['Trade Date'] - desig_date).dt.days
                candidates_df['Abs_Lag'] = candidates_df['Time_Lag_Days'].abs()
                candidates_df = candidates_df.sort_values(by=['Abs_Lag', 'Trade Date'])
            else:
                candidates_df['Time_Lag_Days'] = np.nan
                candidates_df = candidates_df.sort_values(by='Trade Date')

            candidates = candidates_df.to_dict('records')

            for ticket in candidates:
                if abs(phy_vol) < 1:
                    break

                orig_idx = ticket['_original_index']
                curr_allocated = active_paper.at[orig_idx, 'Allocated_To_Phy']
                curr_net_open = active_paper.at[orig_idx, 'Net_Open_Vol']
                net_avail = curr_net_open - curr_allocated

                if abs(net_avail) < 0.0001:
                    continue

                alloc_amt = min(net_avail, phy_vol)

                phy_vol -= alloc_amt
                active_paper.at[orig_idx, 'Allocated_To_Phy'] += alloc_amt

                open_price = ticket.get('Price', 0)
                mtm_price = ticket.get('Mtm Price', 0)
                total_pl = ticket.get('Total P/L', 0)
                close_path, _ = format_close_details(
                    filter_close_events(ticket.get('Close_Events', []), cargo_start_date)
                )

                unrealized_mtm = (mtm_price - open_price) * alloc_amt

                ratio = 0
                if abs(ticket.get('Volume', 0)) > 0:
                    ratio = abs(alloc_amt) / abs(ticket['Volume'])
                alloc_total_pl = total_pl * ratio

                close_volume = 0.0
                close_price_total = 0.0
                close_queue = close_event_pool.get(orig_idx, deque())
                while close_queue and pd.notna(cargo_start_date):
                    peek = close_queue[0]
                    if pd.isna(peek.get('date')) or peek['date'] >= cargo_start_date:
                        break
                    close_queue.popleft()
                remaining = abs(alloc_amt)
                while remaining > 0 and close_queue:
                    event = close_queue[0]
                    take = min(remaining, event['remaining'])
                    close_volume += take
                    close_price_total += take * event['price']
                    event['remaining'] -= take
                    remaining -= take
                    if event['remaining'] < 0.0001:
                        close_queue.popleft()
                close_event_pool[orig_idx] = close_queue
                close_wap = (close_price_total / close_volume) if close_volume > 0 else 0

                hedge_relations.append({
                    'Cargo_ID': cargo_id,
                    'Ticket_ID': ticket.get('Recap No'),
                    'Month': ticket.get('Month'),
                    'Trade_Date': ticket.get('Trade Date'),
                    'Allocated_Vol': alloc_amt,
                    'Open_Price': open_price,
                    'MTM_PL': round(unrealized_mtm, 2),
                    'Total_PL_Alloc': round(alloc_total_pl, 2),
                    'Time_Lag': ticket.get('Time_Lag_Days'),
                    'Close_Path': close_path,
                    'Allocated_Close_Vol': close_volume,
                    'Close_WAP': close_wap
                })

        physical_df_sorted.at[idx, 'Unhedged_Volume'] = phy_vol

    physical_df_sorted = physical_df_sorted.drop(columns=['Benchmark_Priority', 'Cargo_Priority'], errors='ignore')

    relations_df = pd.DataFrame(hedge_relations)
    if not relations_df.empty:
        relations_df['Abs_Allocated_Vol'] = relations_df['Allocated_Vol'].abs()
        relations_df['Close_Value'] = relations_df['Allocated_Close_Vol'] * relations_df['Close_WAP']
        open_summary = relations_df.groupby('Cargo_ID').apply(
            lambda grp: pd.Series({
                'Matched_Open_Vol': grp['Abs_Allocated_Vol'].sum(),
                'Open_WAP': (grp['Abs_Allocated_Vol'] * grp['Open_Price']).sum() / grp['Abs_Allocated_Vol'].sum()
                if grp['Abs_Allocated_Vol'].sum() > 0 else 0,
                'Matched_Close_Vol': grp['Allocated_Close_Vol'].sum(),
                'Close_WAP': grp['Close_Value'].sum() / grp['Allocated_Close_Vol'].sum()
                if grp['Allocated_Close_Vol'].sum() > 0 else 0
            })
        ).reset_index()
        physical_df_sorted = physical_df_sorted.merge(open_summary, on='Cargo_ID', how='left')
        physical_df_sorted['Post_1112_Open_Vol'] = physical_df_sorted['Matched_Open_Vol']
        physical_df_sorted['Post_1112_Open_WAP'] = physical_df_sorted['Open_WAP']
        physical_df_sorted['Post_1112_Close_Vol'] = physical_df_sorted['Matched_Close_Vol']
        physical_df_sorted['Post_1112_Close_WAP'] = physical_df_sorted['Close_WAP']
    else:
        physical_df_sorted['Matched_Open_Vol'] = 0.0
        physical_df_sorted['Open_WAP'] = 0.0
        physical_df_sorted['Matched_Close_Vol'] = 0.0
        physical_df_sorted['Close_WAP'] = 0.0
        physical_df_sorted['Post_1112_Open_Vol'] = 0.0
        physical_df_sorted['Post_1112_Open_WAP'] = 0.0
        physical_df_sorted['Post_1112_Close_Vol'] = 0.0
        physical_df_sorted['Post_1112_Close_WAP'] = 0.0

    # --- 修复回写逻辑 ---
    if not active_paper.empty:
        alloc_map = active_paper.set_index('_original_index')['Allocated_To_Phy']
        paper_df['Allocated_To_Phy'] = paper_df.index.map(alloc_map).fillna(0.0)
    else:
        paper_df['Allocated_To_Phy'] = 0.0
        
    return relations_df, physical_df_sorted, paper_df
