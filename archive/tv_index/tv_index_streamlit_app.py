
# Import python packages
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from datetime import datetime, date
from snowflake.snowpark.context import get_active_session

session = get_active_session()

@st.cache_data
def create_enhanced_ticker_mapping():
    """
    seed_ticker_company_mappingを拡張して、より多くの銘柄をカバーする 173 -> 727 (out of 1180)
    残りは、1 company_idに対し2つ以上のticker codeが振られている企業
    """
    # get seed_ticker_company_mapping
    original_mapping_query = """
    SELECT COMPANY_ID, TICKER_CODE
    FROM datahub_dev.mdata.seed_ticker_company_mapping
    WHERE TICKER_CODE IS NOT NULL AND TICKER_CODE != ''
    """
    original_mapping = session.sql(original_mapping_query).to_pandas()
    
    # 追加のmappingを生成するSQL
    new_mapping_query = """
    WITH summary AS (
        WITH valid_companies AS (
            SELECT COMPANY_ID
            FROM datahub_dev.mdata.export_cm_fixed_v3
            GROUP BY COMPANY_ID
            HAVING COUNT(DISTINCT COALESCE(NULLIF(TICKER_CODE, ''), 'EMPTY')) = 2
        )
        SELECT DISTINCT e.COMPANY_ID, e.TICKER_CODE, e.COMPANY_NAME
        FROM datahub_dev.mdata.export_cm_fixed_v3 e
        INNER JOIN valid_companies vc ON e.COMPANY_ID = vc.COMPANY_ID
        WHERE e.TICKER_CODE <> '' AND e.TICKER_CODE IS NOT NULL
    )
    SELECT s.COMPANY_ID, s.TICKER_CODE, s.COMPANY_NAME
    FROM summary s
    WHERE s.TICKER_CODE IN (
        SELECT TICKER_CODE
        FROM summary
        GROUP BY TICKER_CODE
        HAVING COUNT(*) = 1
    )
    ORDER BY s.TICKER_CODE, s.COMPANY_ID
    """
    new_mapping = session.sql(new_mapping_query).to_pandas()
    
    # 重複チェック
    for ticker in new_mapping['TICKER_CODE'].unique():
        if ticker in original_mapping['TICKER_CODE'].values:
            original_company_id = original_mapping[original_mapping['TICKER_CODE'] == ticker]['COMPANY_ID'].iloc[0]
            new_company_id = new_mapping[new_mapping['TICKER_CODE'] == ticker]['COMPANY_ID'].iloc[0]
            
            if original_company_id != new_company_id:
                new_company_name = new_mapping[new_mapping['TICKER_CODE'] == ticker]['COMPANY_NAME'].iloc[0]
                st.error(f"ERROR: Ticker {ticker} has different COMPANY_IDs - Original: {original_company_id} vs New: {new_company_id} ({new_company_name})")
    
    # 新しいマッピングから既存のものを除外
    new_tickers = set(new_mapping['TICKER_CODE']) - set(original_mapping['TICKER_CODE'])
    additional_mapping = new_mapping[new_mapping['TICKER_CODE'].isin(new_tickers)]
    
    # 統合マッピングの作成
    enhanced_mapping = pd.concat([
        original_mapping.assign(COMPANY_NAME=''), 
        additional_mapping
    ], ignore_index=True)
    
    st.info(f"Original mapping: {len(original_mapping)} records")
    st.info(f"Additional mapping: {len(additional_mapping)} records")
    st.info(f"Enhanced mapping total: {len(enhanced_mapping)} records")
    
    return enhanced_mapping


@st.cache_data
def get_df(company_name_tuple, num_inputs, time_period='weekly') -> pd.DataFrame:
    # time_periodに応じてdate_truncの単位を変更
    date_trunc_unit = 'week' if time_period == 'weekly' else 'month'
    
    if num_inputs == 1:
        query = """
        WITH enhanced_mapping AS (
            SELECT COMPANY_ID, TICKER_CODE
            FROM datahub_dev.mdata.seed_ticker_company_mapping
            WHERE TICKER_CODE IS NOT NULL AND TICKER_CODE != ''
            
            UNION
            
            SELECT s.COMPANY_ID, s.TICKER_CODE
            FROM (
                WITH summary AS (
                    WITH valid_companies AS (
                        SELECT COMPANY_ID
                        FROM datahub_dev.mdata.export_cm_fixed_v3
                        GROUP BY COMPANY_ID
                        HAVING COUNT(DISTINCT COALESCE(NULLIF(TICKER_CODE, ''), 'EMPTY')) = 2
                    )
                    SELECT DISTINCT e.COMPANY_ID, e.TICKER_CODE
                    FROM datahub_dev.mdata.export_cm_fixed_v3 e
                    INNER JOIN valid_companies vc ON e.COMPANY_ID = vc.COMPANY_ID
                    WHERE e.TICKER_CODE <> '' AND e.TICKER_CODE IS NOT NULL
                )
                SELECT s.COMPANY_ID, s.TICKER_CODE
                FROM summary s
                WHERE s.TICKER_CODE IN (
                    SELECT TICKER_CODE
                    FROM summary
                    GROUP BY TICKER_CODE
                    HAVING COUNT(*) = 1
                )
                AND s.TICKER_CODE NOT IN (
                    SELECT TICKER_CODE 
                    FROM datahub_dev.mdata.seed_ticker_company_mapping 
                    WHERE TICKER_CODE IS NOT NULL AND TICKER_CODE != ''
                )
            ) s
        ),
        
        company_ticker AS (
            SELECT DISTINCT
                ex.company_id,
                ex.company_name,
                em.ticker_code
            FROM datahub_dev.mdata.export_cm_fixed_v3 ex
            JOIN enhanced_mapping em 
                ON ex.company_id = em.company_id
            WHERE ex.company_name = '{}'
                AND em.ticker_code IS NOT NULL 
        ),
        
        all_data AS (
            SELECT
                date_trunc({}, ex.broadcast_end_time) as period, 
                em.ticker_code,
                sum(ex.cm_length) as cm_length
            FROM datahub_dev.mdata.export_cm_fixed_v3 ex
            JOIN enhanced_mapping em 
                ON ex.company_id = em.company_id
            WHERE em.ticker_code IS NOT NULL 
            GROUP BY 1, 2
        ),
        avg_by_period AS (
            SELECT 
                period,
                AVG(cm_length) AS avg_cm_length
            FROM all_data
            GROUP BY period
        ),
        selected_data AS (
            SELECT 
                ad.period,
                ad.ticker_code,
                ad.cm_length,
                av.avg_cm_length
            FROM all_data ad
            JOIN avg_by_period av ON ad.period = av.period
            WHERE ad.ticker_code IN (
                SELECT ticker_code FROM company_ticker
            )
        )
        SELECT 
            period,
            ticker_code, 
            cm_length/avg_cm_length as tv_index
        FROM selected_data 
        ORDER BY period, ticker_code
        """.format(company_name_tuple, date_trunc_unit)
    else:
        query = """
        WITH enhanced_mapping AS (
            SELECT COMPANY_ID, TICKER_CODE
            FROM datahub_dev.mdata.seed_ticker_company_mapping
            WHERE TICKER_CODE IS NOT NULL AND TICKER_CODE != ''
            
            UNION
            
            SELECT s.COMPANY_ID, s.TICKER_CODE
            FROM (
                WITH summary AS (
                    WITH valid_companies AS (
                        SELECT COMPANY_ID
                        FROM datahub_dev.mdata.export_cm_fixed_v3
                        GROUP BY COMPANY_ID
                        HAVING COUNT(DISTINCT COALESCE(NULLIF(TICKER_CODE, ''), 'EMPTY')) = 2
                    )
                    SELECT DISTINCT e.COMPANY_ID, e.TICKER_CODE
                    FROM datahub_dev.mdata.export_cm_fixed_v3 e
                    INNER JOIN valid_companies vc ON e.COMPANY_ID = vc.COMPANY_ID
                    WHERE e.TICKER_CODE <> '' AND e.TICKER_CODE IS NOT NULL
                )
                SELECT s.COMPANY_ID, s.TICKER_CODE
                FROM summary s
                WHERE s.TICKER_CODE IN (
                    SELECT TICKER_CODE
                    FROM summary
                    GROUP BY TICKER_CODE
                    HAVING COUNT(*) = 1
                )
                AND s.TICKER_CODE NOT IN (
                    SELECT TICKER_CODE 
                    FROM datahub_dev.mdata.seed_ticker_company_mapping 
                    WHERE TICKER_CODE IS NOT NULL AND TICKER_CODE != ''
                )
            ) s
        ),
        
        company_ticker AS (
            SELECT DISTINCT
                ex.company_id,
                ex.company_name,
                em.ticker_code
            FROM datahub_dev.mdata.export_cm_fixed_v3 ex
            JOIN enhanced_mapping em 
                ON ex.company_id = em.company_id
            WHERE ex.company_name IN {}
                AND em.ticker_code IS NOT NULL 
        ),
        
        all_data AS (
            SELECT
                date_trunc({}, ex.broadcast_end_time) as period, 
                em.ticker_code,
                sum(ex.cm_length) as cm_length
            FROM datahub_dev.mdata.export_cm_fixed_v3 ex
            JOIN enhanced_mapping em 
                ON ex.company_id = em.company_id
            WHERE em.ticker_code IS NOT NULL 
            GROUP BY 1, 2
        ),
        avg_by_period AS (
            SELECT 
                period,
                AVG(cm_length) AS avg_cm_length
            FROM all_data
            GROUP BY period
        ),
        selected_data AS (
            SELECT 
                ad.period,
                ad.ticker_code,
                ad.cm_length,
                av.avg_cm_length
            FROM all_data ad
            JOIN avg_by_period av ON ad.period = av.period
            WHERE ad.ticker_code IN (
                SELECT ticker_code FROM company_ticker
            )
        )
        SELECT 
            period,
            ticker_code, 
            cm_length/avg_cm_length as tv_index
        FROM selected_data 
        ORDER BY period, ticker_code
        """.format(company_name_tuple, date_trunc_unit)
    
    run_result = session.sql(query).collect()
    df = pd.DataFrame(run_result)
    if not df.empty:
        df.index = df['PERIOD']
        df.index = pd.to_datetime(df.index)
        df = df[['TICKER_CODE', 'TV_INDEX']]
        df['TV_INDEX'] = df['TV_INDEX'].fillna(0)
        df['TV_INDEX'] = df['TV_INDEX'].replace('', 0)
    return df


def calculate_yoy(df, yoy_cap_percent=500, data_column='TV_INDEX', time_period='weekly'):
    """
    YoYを計算
    異常値(±500%超)はNoneに設定して除外
    """
    df_yoy = df.copy()
    
    for ticker in df_yoy['TICKER_CODE'].unique():
        ticker_mask = df_yoy['TICKER_CODE'] == ticker
        ticker_data = df_yoy.loc[ticker_mask, data_column].copy()
        
        if time_period == 'weekly':
            year_offset = pd.DateOffset(weeks=52)
            search_range = 2
        else:
            year_offset = pd.DateOffset(months=12)
            search_range = 1
        
        # yoy計算
        yoy_values = []
        for idx in ticker_data.index:
            year_ago = idx - year_offset
            
            # 1年前のデータ
            if time_period == 'weekly':
                year_ago_range = [year_ago - pd.DateOffset(weeks=i) for i in range(-search_range, search_range + 1)]
            else:
                year_ago_range = [year_ago - pd.DateOffset(months=i) for i in range(-search_range, search_range + 1)]
            
            year_ago_value = None
            
            for date_candidate in year_ago_range:
                if date_candidate in ticker_data.index:
                    year_ago_value = ticker_data.loc[date_candidate]
                    break
            
            if year_ago_value is not None and year_ago_value > 0:
                current_value = ticker_data.loc[idx]
                yoy_change = ((current_value / year_ago_value) - 1) * 100
                
                # 異常値処理 -> ±500%を超える場合はNone
                if yoy_change > yoy_cap_percent or yoy_change < -yoy_cap_percent:
                    yoy_values.append(None)
                else:
                    yoy_values.append(yoy_change)
            else:
                yoy_values.append(None)
        
        df_yoy.loc[ticker_mask, data_column] = yoy_values
    
    return df_yoy


def detect_golden_cross(df, lookback_periods=4):
    """
    ゴールデンクロスを検知する関数
    """
    if 'TV_INDEX_WMA_4W' not in df.columns or 'TV_INDEX_WMA_8W' not in df.columns:
        return pd.DataFrame()
    
    golden_cross_results = []
    
    current_date = pd.Timestamp.now().normalize()
    
    for ticker in df['TICKER_CODE'].unique():
        ticker_data = df[df['TICKER_CODE'] == ticker].copy()
        ticker_data = ticker_data.sort_index()
        
        if len(ticker_data) < 2:
            continue
        
        # 4W WMAと8W WMAの差分を計算
        ticker_data['wma_diff'] = ticker_data['TV_INDEX_WMA_4W'] - ticker_data['TV_INDEX_WMA_8W']
        
        # ゴールデンクロスの検知
        for i in range(1, len(ticker_data)):
            current_diff = ticker_data['wma_diff'].iloc[i]
            prev_diff = ticker_data['wma_diff'].iloc[i-1]
            
            if prev_diff <= 0 and current_diff > 0:
                # 交点の計算
                current_date_val = ticker_data.index[i]
                prev_date_val = ticker_data.index[i-1]
                time_diff = (current_date_val - prev_date_val).days / 7
                
                if current_diff != prev_diff:
                    ratio = abs(prev_diff) / (abs(prev_diff) + abs(current_diff))
                    cross_date = prev_date_val + pd.Timedelta(weeks=time_diff * ratio)
                    
                    prev_4w = ticker_data['TV_INDEX_WMA_4W'].iloc[i-1]
                    current_4w = ticker_data['TV_INDEX_WMA_4W'].iloc[i]
                    cross_4w = prev_4w + (current_4w - prev_4w) * ratio
                    
                    prev_8w = ticker_data['TV_INDEX_WMA_8W'].iloc[i-1]
                    current_8w = ticker_data['TV_INDEX_WMA_8W'].iloc[i]
                    cross_8w = prev_8w + (current_8w - prev_8w) * ratio
                    
                    prev_tv = ticker_data['TV_INDEX'].iloc[i-1]
                    current_tv = ticker_data['TV_INDEX'].iloc[i]
                    cross_tv = prev_tv + (current_tv - prev_tv) * ratio
                else:
                    cross_date = current_date_val
                    cross_4w = ticker_data['TV_INDEX_WMA_4W'].iloc[i]
                    cross_8w = ticker_data['TV_INDEX_WMA_8W'].iloc[i]
                    cross_tv = ticker_data['TV_INDEX'].iloc[i]
                
                weeks_ago = max(0, (current_date - cross_date).days // 7)
                
                # lookback_periods以内
                if weeks_ago <= lookback_periods:
                    golden_cross_results.append({
                        'TICKER_CODE': ticker,
                        'CROSS_DATE': cross_date,
                        'WMA_4W': cross_4w,
                        'WMA_8W': cross_8w,
                        'TV_INDEX': cross_tv,
                        'PERIODS_AGO': weeks_ago
                    })
    
    if golden_cross_results:
        return pd.DataFrame(golden_cross_results)
    else:
        return pd.DataFrame()


def detect_golden_cross_all_companies_full(df, lookback_periods=4):
    """
    全銘柄データでゴールデンクロスを検知する関数
    """
    # 加重平均を追加
    df_with_ma = add_moving_averages_full(df)
    
    all_golden_cross_results = []
    
    current_date = pd.Timestamp.now().normalize()
    
    for ticker in df_with_ma['TICKER_CODE'].unique():
        ticker_data = df_with_ma[df_with_ma['TICKER_CODE'] == ticker].copy()
        ticker_data = ticker_data.sort_index()
        
        # 8週分のデータがない場合はスキップ
        if len(ticker_data) < 8:
            continue

        ticker_data['wma_diff'] = ticker_data['TV_INDEX_WMA_4W'] - ticker_data['TV_INDEX_WMA_8W']
        
        # ゴールデンクロスの検知
        for i in range(1, len(ticker_data)):
            current_diff = ticker_data['wma_diff'].iloc[i]
            prev_diff = ticker_data['wma_diff'].iloc[i-1]
            
            if prev_diff <= 0 and current_diff > 0:
                # 交点の計算
                current_date_val = ticker_data.index[i]
                prev_date_val = ticker_data.index[i-1]
                
                time_diff = (current_date_val - prev_date_val).days / 7
                
                if current_diff != prev_diff:
                    ratio = abs(prev_diff) / (abs(prev_diff) + abs(current_diff))
                    cross_date = prev_date_val + pd.Timedelta(weeks=time_diff * ratio)
                else:
                    cross_date = current_date_val
                
                weeks_ago = max(0, (current_date - cross_date).days // 7)
                
                # lookback_periods以内
                if weeks_ago <= lookback_periods:
                    company_name = ticker_data['COMPANY_NAME'].iloc[i]
                    
                    all_golden_cross_results.append({
                        'COMPANY_NAME': company_name,
                        'TICKER_CODE': ticker,
                        'CROSS_DATE': cross_date,
                        'PERIODS_AGO': weeks_ago
                    })
    
    if all_golden_cross_results:
        return pd.DataFrame(all_golden_cross_results)
    else:
        return pd.DataFrame()


def detect_golden_cross_all_companies(df_list, valid_displayed_names, name_list, lookback_periods=4):
    """
    選択銘柄のゴールデンクロスを検知する関数(通常モード)
    """
    all_golden_cross_results = []
    
    current_date = pd.Timestamp.now().normalize()
    
    for i, df in enumerate(df_list):
        # 加重平均を追加
        df_with_ma = add_moving_averages(df)

        company_name = valid_displayed_names[i] if i < len(valid_displayed_names) else f"Company{i+1}"
        
        for ticker in df_with_ma['TICKER_CODE'].unique():
            ticker_data = df_with_ma[df_with_ma['TICKER_CODE'] == ticker].copy()
            ticker_data = ticker_data.sort_index()
            
            if len(ticker_data) < 2:
                continue
            
            ticker_data['wma_diff'] = ticker_data['TV_INDEX_WMA_4W'] - ticker_data['TV_INDEX_WMA_8W']
            
            # ゴールデンクロスの検知
            for j in range(1, len(ticker_data)):
                current_diff = ticker_data['wma_diff'].iloc[j]
                prev_diff = ticker_data['wma_diff'].iloc[j-1]
                
                if prev_diff <= 0 and current_diff > 0:
                    current_date_val = ticker_data.index[j]
                    prev_date_val = ticker_data.index[j-1]
                    
                    time_diff = (current_date_val - prev_date_val).days / 7
                    
                    if current_diff != prev_diff:
                        ratio = abs(prev_diff) / (abs(prev_diff) + abs(current_diff))
                        cross_date = prev_date_val + pd.Timedelta(weeks=time_diff * ratio)
                    else:
                        cross_date = current_date_val
                    
                    weeks_ago = max(0, (current_date - cross_date).days // 7)
                    
                    # lookback_periods以内
                    if weeks_ago <= lookback_periods:
                        all_golden_cross_results.append({
                            'COMPANY_NAME': company_name,
                            'TICKER_CODE': ticker,
                            'CROSS_DATE': cross_date,
                            'PERIODS_AGO': weeks_ago
                        })
    
    if all_golden_cross_results:
        return pd.DataFrame(all_golden_cross_results)
    else:
        return pd.DataFrame()


def add_moving_averages_full(df, periods=[4, 8]):
    """
    全銘柄データに加重平均を追加する関数
    """
    df_with_ma = df.copy()
    
    for ticker in df['TICKER_CODE'].unique():
        ticker_data = df[df['TICKER_CODE'] == ticker].copy()
        ticker_data = ticker_data.sort_index()
        
        for period in periods:
            # 加重平均を計算
            weights = list(range(1, period + 1))
            ticker_data[f'TV_INDEX_WMA_{period}W'] = ticker_data['TV_INDEX'].rolling(
                window=period, min_periods=1
            ).apply(lambda x: sum(x * weights[-len(x):]) / sum(weights[-len(x):]), raw=True)
        
        # 元のデータフレームに結合
        df_with_ma.loc[df_with_ma['TICKER_CODE'] == ticker, 
                       [f'TV_INDEX_WMA_{p}W' for p in periods]] = ticker_data[[f'TV_INDEX_WMA_{p}W' for p in periods]].values
    
    return df_with_ma


def add_moving_averages(df, periods=[4, 8]):
    """
    加重平均を追加する関数
    """
    df_with_ma = df.copy()
    
    for ticker in df['TICKER_CODE'].unique():
        ticker_data = df[df['TICKER_CODE'] == ticker].copy()
        ticker_data = ticker_data.sort_index()
        
        for period in periods:
            # 加重平均を計算
            weights = list(range(1, period + 1))
            ticker_data[f'TV_INDEX_WMA_{period}W'] = ticker_data['TV_INDEX'].rolling(
                window=period, min_periods=1
            ).apply(lambda x: sum(x * weights[-len(x):]) / sum(weights[-len(x):]), raw=True)
        
        # 元のデータフレームに結合
        df_with_ma.loc[df_with_ma['TICKER_CODE'] == ticker, 
                       [f'TV_INDEX_WMA_{p}W' for p in periods]] = ticker_data[[f'TV_INDEX_WMA_{p}W' for p in periods]].values
    
    return df_with_ma


def get_company_name(company_name):
    query = """
    WITH enhanced_mapping AS (
        SELECT COMPANY_ID, TICKER_CODE
        FROM datahub_dev.mdata.seed_ticker_company_mapping
        WHERE TICKER_CODE IS NOT NULL AND TICKER_CODE != ''
        
        UNION
        
        SELECT s.COMPANY_ID, s.TICKER_CODE
        FROM (
            WITH summary AS (
                WITH valid_companies AS (
                    SELECT COMPANY_ID
                    FROM datahub_dev.mdata.export_cm_fixed_v3
                    GROUP BY COMPANY_ID
                    HAVING COUNT(DISTINCT COALESCE(NULLIF(TICKER_CODE, ''), 'EMPTY')) = 2
                )
                SELECT DISTINCT e.COMPANY_ID, e.TICKER_CODE
                FROM datahub_dev.mdata.export_cm_fixed_v3 e
                INNER JOIN valid_companies vc ON e.COMPANY_ID = vc.COMPANY_ID
                WHERE e.TICKER_CODE <> '' AND e.TICKER_CODE IS NOT NULL
            )
            SELECT s.COMPANY_ID, s.TICKER_CODE
            FROM summary s
            WHERE s.TICKER_CODE IN (
                SELECT TICKER_CODE
                FROM summary
                GROUP BY TICKER_CODE
                HAVING COUNT(*) = 1
            )
            AND s.TICKER_CODE NOT IN (
                SELECT TICKER_CODE 
                FROM datahub_dev.mdata.seed_ticker_company_mapping 
                WHERE TICKER_CODE IS NOT NULL AND TICKER_CODE != ''
            )
        ) s
    )
    SELECT DISTINCT ex.company_name
    FROM datahub_dev.mdata.export_cm_fixed_v3 ex
    JOIN enhanced_mapping em 
        ON ex.company_id = em.company_id
    WHERE ex.company_name LIKE '%{}%' 
    AND em.ticker_code IS NOT NULL
    """.format(company_name)
    run_result = session.sql(query).collect()
    return pd.Series(run_result).to_list()


@st.cache_data
def get_all_ticker_company_mapping():
    """
    全銘柄のticker_codeと企業名のマッピングを取得する関数
    """
    query = """
    WITH enhanced_mapping AS (
        SELECT COMPANY_ID, TICKER_CODE
        FROM datahub_dev.mdata.seed_ticker_company_mapping
        WHERE TICKER_CODE IS NOT NULL AND TICKER_CODE != ''
        
        UNION
        
        SELECT s.COMPANY_ID, s.TICKER_CODE
        FROM (
            WITH summary AS (
                WITH valid_companies AS (
                    SELECT COMPANY_ID
                    FROM datahub_dev.mdata.export_cm_fixed_v3
                    GROUP BY COMPANY_ID
                    HAVING COUNT(DISTINCT COALESCE(NULLIF(TICKER_CODE, ''), 'EMPTY')) = 2
                )
                SELECT DISTINCT e.COMPANY_ID, e.TICKER_CODE
                FROM datahub_dev.mdata.export_cm_fixed_v3 e
                INNER JOIN valid_companies vc ON e.COMPANY_ID = vc.COMPANY_ID
                WHERE e.TICKER_CODE <> '' AND e.TICKER_CODE IS NOT NULL
            )
            SELECT s.COMPANY_ID, s.TICKER_CODE
            FROM summary s
            WHERE s.TICKER_CODE IN (
                SELECT TICKER_CODE
                FROM summary
                GROUP BY TICKER_CODE
                HAVING COUNT(*) = 1
            )
            AND s.TICKER_CODE NOT IN (
                SELECT TICKER_CODE 
                FROM datahub_dev.mdata.seed_ticker_company_mapping 
                WHERE TICKER_CODE IS NOT NULL AND TICKER_CODE != ''
            )
        ) s
    )
    SELECT DISTINCT 
        em.ticker_code,
        ex.company_name
    FROM datahub_dev.mdata.export_cm_fixed_v3 ex
    JOIN enhanced_mapping em 
        ON ex.company_id = em.company_id
    WHERE em.ticker_code IS NOT NULL
        AND ex.company_name IS NOT NULL
        AND ex.company_name != ''
    ORDER BY em.ticker_code
    """
    
    run_result = session.sql(query).collect()
    df = pd.DataFrame(run_result)
    return df


def get_all_companies_data(time_period='weekly'):
    """
    全銘柄のTV Indexデータを取得する関数
    """
    date_trunc_unit = 'week' if time_period == 'weekly' else 'month'
    
    query = """
    WITH enhanced_mapping AS (
        SELECT COMPANY_ID, TICKER_CODE
        FROM datahub_dev.mdata.seed_ticker_company_mapping
        WHERE TICKER_CODE IS NOT NULL AND TICKER_CODE != ''
        
        UNION
        
        SELECT s.COMPANY_ID, s.TICKER_CODE
        FROM (
            WITH summary AS (
                WITH valid_companies AS (
                    SELECT COMPANY_ID
                    FROM datahub_dev.mdata.export_cm_fixed_v3
                    GROUP BY COMPANY_ID
                    HAVING COUNT(DISTINCT COALESCE(NULLIF(TICKER_CODE, ''), 'EMPTY')) = 2
                )
                SELECT DISTINCT e.COMPANY_ID, e.TICKER_CODE
                FROM datahub_dev.mdata.export_cm_fixed_v3 e
                INNER JOIN valid_companies vc ON e.COMPANY_ID = vc.COMPANY_ID
                WHERE e.TICKER_CODE <> '' AND e.TICKER_CODE IS NOT NULL
            )
            SELECT s.COMPANY_ID, s.TICKER_CODE
            FROM summary s
            WHERE s.TICKER_CODE IN (
                SELECT TICKER_CODE
                FROM summary
                GROUP BY TICKER_CODE
                HAVING COUNT(*) = 1
            )
            AND s.TICKER_CODE NOT IN (
                SELECT TICKER_CODE 
                FROM datahub_dev.mdata.seed_ticker_company_mapping 
                WHERE TICKER_CODE IS NOT NULL AND TICKER_CODE != ''
            )
        ) s
    ),
    all_data AS (
        SELECT
            date_trunc({}, ex.broadcast_end_time) as period, 
            em.ticker_code,
            ex.company_name,
            sum(ex.cm_length) as cm_length
        FROM datahub_dev.mdata.export_cm_fixed_v3 ex
        JOIN enhanced_mapping em 
            ON ex.company_id = em.company_id
        WHERE em.ticker_code IS NOT NULL
            AND ex.company_name IS NOT NULL
            AND ex.company_name != ''
        GROUP BY 1, 2, 3
    ),
    avg_by_period AS (
        SELECT 
            period,
            AVG(cm_length) AS avg_cm_length
        FROM all_data
        GROUP BY period
    )
    SELECT 
        ad.period,
        ad.ticker_code,
        ad.company_name,
        ad.cm_length/av.avg_cm_length as tv_index
    FROM all_data ad
    JOIN avg_by_period av ON ad.period = av.period
    ORDER BY period, ticker_code
    """.format(date_trunc_unit)
    
    run_result = session.sql(query).collect()
    df = pd.DataFrame(run_result)
    if not df.empty:
        df.index = df['PERIOD']
        df.index = pd.to_datetime(df.index)
        df = df[['TICKER_CODE', 'COMPANY_NAME', 'TV_INDEX']]
        # null, nan, ''を0で置換
        df['TV_INDEX'] = df['TV_INDEX'].fillna(0)
        df['TV_INDEX'] = df['TV_INDEX'].replace('', 0)
    return df


@st.cache_data
def get_date_range():
    """データの期間を取得"""
    query = """
    SELECT 
        MIN(date_trunc(week, ex.broadcast_end_time)) as min_date,
        MAX(date_trunc(week, ex.broadcast_end_time)) as max_date
    FROM datahub_dev.mdata.export_cm_fixed_v3 ex
    JOIN datahub_dev.mdata.seed_ticker_company_mapping stcm 
        ON ex.company_id = stcm.company_id
    WHERE stcm.ticker_code IS NOT NULL
    """
    result = session.sql(query).to_pandas()
    return result.iloc[0]['MIN_DATE'].date(), result.iloc[0]['MAX_DATE'].date()


def plot_df_summary(df_list, displayed_name_list, date_range, data_type='original', time_period='weekly', show_yoy=False):
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    for i in range(len(df_list)):
        df = df_list[i]
        df_filtered = df[(df.index >= pd.to_datetime(date_range[0])) & 
                        (df.index <= pd.to_datetime(date_range[1]))]
        
        # 週次データの場合は加重平均を追加
        if time_period == 'weekly' and data_type in ['wma_4w', 'wma_8w']:
            df_filtered = add_moving_averages(df_filtered)
        
        displayed_name = displayed_name_list[i]
        base_color = colors[i % len(colors)]
        
        for ticker in df_filtered['TICKER_CODE'].unique():
            ticker_data = df_filtered[df_filtered['TICKER_CODE'] == ticker]
            
            if data_type == 'original':
                data_to_plot = ticker_data['TV_INDEX']
                target_column = 'TV_INDEX'
                label_suffix = ""
            elif data_type == 'wma_4w':
                data_to_plot = ticker_data['TV_INDEX_WMA_4W']
                target_column = 'TV_INDEX_WMA_4W'
                label_suffix = " 4W WMA"
            elif data_type == 'wma_8w':
                data_to_plot = ticker_data['TV_INDEX_WMA_8W']
                target_column = 'TV_INDEX_WMA_8W'
                label_suffix = " 8W WMA"
            
            # yoy
            if show_yoy:
                ticker_data_yoy = calculate_yoy(ticker_data, yoy_cap_percent=500, data_column=target_column, time_period=time_period)
                data_to_plot = ticker_data_yoy[target_column]
            
            # nanを除去
            data_to_plot = data_to_plot.dropna()
            if not data_to_plot.empty:
                data_to_plot.plot(ax=ax, label=f"{displayed_name}({ticker}){label_suffix}", 
                                 color=base_color, linewidth=2)
    
    ax.grid(True)
    period_label = "Weekly" if time_period == 'weekly' else "Monthly"
    data_label = {
        'original': '',
        'wma_4w': ' / 4W WMA',
        'wma_8w': ' / 8W WMA'
    }.get(data_type, '')
    
    if show_yoy:
        ax.set_title(f'TV Index YoY Comparison (%) [{period_label}{data_label}]')
        ax.set_ylabel('YoY Change (%)')
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    else:
        ax.set_title(f'TV Index Time Series Comparison [{period_label}{data_label}]')
        ax.set_ylabel('TV Index')
    
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)


def make_table(df_list, displayed_name_list, date_range, data_type='original', time_period='weekly', show_yoy=False):
    all_filtered_data = []
    
    for i, df in enumerate(df_list):
        df_filtered = df[(df.index >= pd.to_datetime(date_range[0])) & 
                        (df.index <= pd.to_datetime(date_range[1]))]
        if not df_filtered.empty:
            # 週次データで加重平均が必要な場合は追加
            if time_period == 'weekly' and data_type in ['wma_4w', 'wma_8w']:
                df_filtered = add_moving_averages(df_filtered)
            
            all_filtered_data.append(df_filtered)
    
    if all_filtered_data:
        combined_df = pd.concat(all_filtered_data)
        
        # データタイプに応じてピボットするカラムを選択
        if data_type == 'original':
            pivot_column = 'TV_INDEX'
        elif data_type == 'wma_4w':
            pivot_column = 'TV_INDEX_WMA_4W'
        elif data_type == 'wma_8w':
            pivot_column = 'TV_INDEX_WMA_8W'
        
        # yoy
        if show_yoy:
            combined_df = calculate_yoy(combined_df, yoy_cap_percent=500, data_column=pivot_column, time_period=time_period)
        
        pivot_df = combined_df.pivot_table(
            index=combined_df.index,
            columns='TICKER_CODE',
            values=pivot_column,
            aggfunc='mean'
        )
        pivot_df.index = pivot_df.index.strftime('%Y-%m-%d')
        
        if show_yoy:
            pivot_df = pivot_df.round(1)
        else:
            pivot_df = pivot_df.round(3)
        
        return pivot_df
    
    return pd.DataFrame()

##=======================================================================
## Main =================================================================
##=======================================================================

# 初期化
if "show_analysis" not in st.session_state:
    st.session_state.show_analysis = False
if "selected_data" not in st.session_state:
    st.session_state.selected_data = []
if "df_list" not in st.session_state:
    st.session_state.df_list = []
if "valid_displayed_names" not in st.session_state:
    st.session_state.valid_displayed_names = []
if "name_list" not in st.session_state:
    st.session_state.name_list = []
if "num_inputs_list" not in st.session_state:
    st.session_state.num_inputs_list = []
if "compare_all_companies" not in st.session_state:
    st.session_state.compare_all_companies = False

# [画面1]: 会社選択画面
if not st.session_state.show_analysis:
    st.title("TV Index Quick Look")
    
    inputs = []

    # 入力フィールド
    num_compared_companies = st.number_input("比較する数を選択", min_value=1, max_value=9999999, value=1)
    
    # 全銘柄比較チェック欄
    compare_all_companies = st.checkbox("全銘柄を比較(ゴールデンクロス検知専用)", value=False, 
                                       help="チェックすると、2ページ目で加重平均分析のみを表示します")

    name_list = []
    num_inputs_list = []
    displayed_name_list = []

    for j in range(num_compared_companies):
        st.write(f"""##### {j+1}つ目""")
        col1, col2 = st.columns(2)
        with col1:
            displayed_name = st.text_input("グラフに記入する名前を入力 (英語)", value = '', key=f'displayed_name{j}')
            displayed_name_list.append(displayed_name)
        with col2:
            num_inputs = st.number_input("入力する法人数を選択", min_value=1, max_value=9999999, value=1, key = f'select_num{j}')
        num_inputs_list.append(num_inputs)
        col1, col2 = st.columns(2)
        if num_inputs == 1:
            with col1:
                search_name = st.text_input("法人名を検索1", value = '', key=f'search_name{j}')
            company_name = get_company_name(search_name)
            with col2:
               result_tuple = st.selectbox(
            '法人名1',
            company_name, key=f'select_name{j}')
                

        else:
            inputs = []
            for i in range(num_inputs):
                with col1:
                    search_name = st.text_input(f"法人名を検索{i+1}", value = '', key=f'search_name{j}_{i}')
                company_name = get_company_name(search_name)
                with col2:
                    input_value = st.selectbox(
                    f'法人名{i+1}',
                    company_name, key=f'select_name{j}_{i}')
                inputs.append(input_value)

            if all(inputs): # 全ての入力がされているかチェック
                result_tuple = tuple(inputs)
            else:
                st.write("すべての入力を埋めてください")
                result_tuple = None
        name_list.append(result_tuple)

    if st.button("View Graph"):
        df_list = []
        valid_displayed_names = []
        
        for i in range(len(name_list)):
            if name_list[i] is not None:
                result_tuple = name_list[i]
                num_inputs = num_inputs_list[i]
                df = get_df(result_tuple, num_inputs, 'weekly')
                if not df.empty:
                    df_list.append(df)
                    display_name = displayed_name_list[i] if displayed_name_list[i].strip() else f"Company{i+1}"
                    valid_displayed_names.append(display_name)
        
        if df_list:
            # stに保存
            st.session_state.df_list = df_list
            st.session_state.valid_displayed_names = valid_displayed_names
            st.session_state.name_list = name_list
            st.session_state.num_inputs_list = num_inputs_list
            st.session_state.compare_all_companies = compare_all_companies
            st.session_state.show_analysis = True
            st.rerun()
        else:
            st.error("データが見つかりませんでした。会社名を確認してください。")

# [画面2]: 分析画面
else:
    # 戻るボタン
    if st.button("← 企業選択画面に戻る"):
        st.session_state.show_analysis = False
        st.rerun()

    st.title("TV Index 分析")
    
    # stからデータを取得
    try:
        if (not hasattr(st.session_state, 'valid_displayed_names') or
            not hasattr(st.session_state, 'name_list') or 
            not hasattr(st.session_state, 'num_inputs_list') or
            not st.session_state.valid_displayed_names or 
            not st.session_state.name_list or 
            not st.session_state.num_inputs_list):
            st.error("データが正しく設定されていません。企業選択画面に戻って再設定してください。")
            if st.button("企業選択画面に戻る"):
                st.session_state.show_analysis = False
                st.rerun()
            st.stop()
            
        min_date, max_date = get_date_range()
        valid_displayed_names = st.session_state.valid_displayed_names
        name_list = st.session_state.name_list
        num_inputs_list = st.session_state.num_inputs_list
        compare_all_companies = st.session_state.get('compare_all_companies', False)

        if compare_all_companies:
            # 全銘柄比較モード
            time_period = 'weekly' # 週次で固定
            show_yoy = False  # YoYオフ
            
            st.subheader("加重平均分析(ゴールデンクロス検知) - 全銘柄")
            
            # 全銘柄のデータを取得
            all_companies_df = get_all_companies_data(time_period)
            
            if all_companies_df.empty:
                st.error("全銘柄データの取得に失敗しました。")
                if st.button("企業選択画面に戻る"):
                    st.session_state.show_analysis = False
                    st.rerun()
                st.stop()
            
            # 検知期間の設定バー
            lookback_weeks = st.slider(
                "ゴールデンクロス検知期間(過去何週間以内)", 
                min_value=1, 
                max_value=12, 
                value=3,
                help="過去何週間以内でゴールデンクロスが発生した銘柄を表示"
            )
            
            # 全銘柄のゴールデンクロス検知
            all_golden_cross_df = detect_golden_cross_all_companies_full(
                all_companies_df, lookback_periods=lookback_weeks
            )
            
            # ゴールデンクロス検知結果表示
            if not all_golden_cross_df.empty:
                st.success(f"ゴールデンクロス検知: {len(all_golden_cross_df)}件(過去{lookback_weeks}週以内、全{len(all_companies_df['TICKER_CODE'].unique())}銘柄中)")
                
                # ゴールデンクロステーブル
                display_gc = all_golden_cross_df.copy()
                display_gc = display_gc.sort_values('CROSS_DATE', ascending=False) # クロス発生日によるソート
                display_gc['CROSS_DATE'] = display_gc['CROSS_DATE'].dt.strftime('%Y-%m-%d')
                display_gc = display_gc[['COMPANY_NAME', 'TICKER_CODE', 'CROSS_DATE', 'PERIODS_AGO']]
                display_gc.columns = ['企業名', 'ティッカー', 'クロス発生日', '何週前']
                
                st.dataframe(display_gc, use_container_width=True, hide_index=True)
                
                # ゴールデンクロスが検知された銘柄のチャートを表示
                if len(all_golden_cross_df) > 0:
                    st.subheader("ゴールデンクロス検知銘柄のチャート")
                    
                    # 検知された銘柄を選択可能にする
                    detected_tickers = all_golden_cross_df['TICKER_CODE'].unique()
                    if len(detected_tickers) > 1:
                        selected_ticker = st.selectbox(
                            "チャート表示する銘柄を選択",
                            detected_tickers,
                            format_func=lambda x: f"{all_golden_cross_df[all_golden_cross_df['TICKER_CODE']==x]['COMPANY_NAME'].iloc[0]} ({x})"
                        )
                    else:
                        selected_ticker = detected_tickers[0]
                    
                    # 選択された銘柄のチャートを表示
                    ticker_data = all_companies_df[all_companies_df['TICKER_CODE'] == selected_ticker].copy()
                    if not ticker_data.empty:
                        # 加重平均を計算
                        ticker_data = add_moving_averages_full(ticker_data)
                        
                        # 過去1年のデータに絞る
                        latest_date = ticker_data.index.max()
                        one_year_ago = latest_date - pd.DateOffset(weeks=52)
                        ticker_chart_data = ticker_data[ticker_data.index >= one_year_ago]
                        
                        if not ticker_chart_data.empty:
                            # 過去1年間のすべてのゴールデンクロスを検知
                            golden_cross_all_year = detect_golden_cross(add_moving_averages_full(ticker_data[ticker_data.index >= one_year_ago]), lookback_periods=52)
                            
                            # チャート描画
                            fig, ax = plt.subplots(figsize=(12, 6))
                            
                            # 全銘柄モードではticker codeを表示名として使用
                            display_name = selected_ticker
                            
                            ticker_chart_data['TV_INDEX_WMA_4W'].plot(ax=ax, label=f'{display_name} 4W WMA', 
                                                                   color='orange', linewidth=2)
                            ticker_chart_data['TV_INDEX_WMA_8W'].plot(ax=ax, label=f'{display_name} 8W WMA', 
                                                                   color='green', linewidth=2)
                            
                            # 過去1年間のすべてのゴールデンクロス地点をマーク
                            if not golden_cross_all_year.empty:
                                ticker_gc = golden_cross_all_year[golden_cross_all_year['TICKER_CODE'] == selected_ticker]
                                for i, (_, gc_row) in enumerate(ticker_gc.iterrows()):
                                    cross_value = gc_row['WMA_4W']
                                    ax.scatter(gc_row['CROSS_DATE'], cross_value, 
                                             color='red', s=80, marker='o', zorder=5, 
                                             label='Golden Cross' if i == 0 else "")
                            
                            ax.grid(True)
                            ax.set_title(f'Weighted Moving Average Analysis - {display_name} (Past 1 Year)')
                            ax.set_ylabel('TV Index')
                            ax.legend()
                            plt.xticks(rotation=45)
                            plt.tight_layout()
                            st.pyplot(fig)
            else:
                st.info(f"過去{lookback_weeks}週間以内にゴールデンクロスは検知されませんでした。(全{len(all_companies_df['TICKER_CODE'].unique())}銘柄中)")
            
        else:
            # 通常モード
            st.subheader("表示オプション")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                time_period = st.selectbox(
                    "時間軸を選択",
                    options=['weekly', 'monthly'],
                    format_func=lambda x: 'Weekly' if x == 'weekly' else 'Monthly'
                )
            
            with col2:
                data_type = 'original'
                if time_period == 'weekly':
                    data_type = st.selectbox(
                        "データタイプを選択",
                        options=['original', 'wma_4w', 'wma_8w'],
                        format_func=lambda x: {
                            'original': '元データ',
                            'wma_4w': '4週加重平均',
                            'wma_8w': '8週加重平均'
                        }[x]
                    )
            
            with col3:
                show_yoy = st.checkbox("YoY(前年同期比)で表示", value=False)
            
            # 時間軸が変更された場合はデータを再取得
            df_list = []
            for i in range(len(name_list)):
                if name_list[i] is not None:
                    try:
                        result_tuple = name_list[i]
                        num_inputs = num_inputs_list[i] if i < len(num_inputs_list) else 1
                        df = get_df(result_tuple, num_inputs, time_period)
                        if not df.empty:
                            df_list.append(df)
                    except Exception as e:
                        st.warning(f"企業 {i+1} のデータ取得に失敗しました: {str(e)}")
                        continue
            
            if not df_list:
                st.error("有効なデータが見つかりませんでした。企業選択画面に戻って再設定してください。")
                if st.button("企業選択画面に戻る"):
                    st.session_state.show_analysis = False
                    st.rerun()
                st.stop()
            
            st.subheader("期間選択")
            date_range = st.slider(
                "分析期間を選択してください",
                min_value=min_date,
                max_value=max_date,
                value=(min_date, max_date),
                format="YYYY-MM-DD"
            )
            
            # 選択された期間でグラフを表示
            if len(df_list) >= 2:
                plot_df_summary(df_list, valid_displayed_names[:len(df_list)], date_range, data_type, time_period, show_yoy)
            elif len(df_list) == 1:
                df = df_list[0]
                displayed_name = valid_displayed_names[0] if valid_displayed_names else "Company1"
                
                df_filtered = df[(df.index >= pd.to_datetime(date_range[0])) & 
                                (df.index <= pd.to_datetime(date_range[1]))]
                
                # 加重平均
                if time_period == 'weekly' and data_type in ['wma_4w', 'wma_8w']:
                    df_filtered = add_moving_averages(df_filtered)
                
                fig, ax = plt.subplots(figsize=(10, 4))
                for ticker in df_filtered['TICKER_CODE'].unique():
                    ticker_data = df_filtered[df_filtered['TICKER_CODE'] == ticker]
                    
                    # データタイプに応じて表示するデータとカラム名を選択
                    if data_type == 'original':
                        data_to_plot = ticker_data['TV_INDEX']
                        target_column = 'TV_INDEX'
                        label_suffix = ""
                    elif data_type == 'wma_4w':
                        data_to_plot = ticker_data['TV_INDEX_WMA_4W']
                        target_column = 'TV_INDEX_WMA_4W'
                        label_suffix = " / 4W WMA"
                    elif data_type == 'wma_8w':
                        data_to_plot = ticker_data['TV_INDEX_WMA_8W']
                        target_column = 'TV_INDEX_WMA_8W'
                        label_suffix = " / 8W WMA"
                    
                    # yoy
                    if show_yoy:
                        ticker_data_yoy = calculate_yoy(ticker_data, yoy_cap_percent=500, data_column=target_column, time_period=time_period)
                        data_to_plot = ticker_data_yoy[target_column]
                    
                    data_to_plot = data_to_plot.dropna()
                    if not data_to_plot.empty:
                        data_to_plot.plot(ax=ax, linewidth=2)
                    
                    ax.grid(True)
                    period_label = "Weekly" if time_period == 'weekly' else "Monthly"
                    
                    if show_yoy:
                        ax.set_title(f'TV Index YoY - {displayed_name}({ticker}) [{period_label}{label_suffix}]')
                        ax.set_ylabel('YoY Change (%)')
                        ax.axhline(y=0, color='red', linestyle='-', alpha=0.5)  # ゼロライン
                    else:
                        ax.set_title(f'TV Index - {displayed_name}({ticker}) [{period_label}{label_suffix}]')
                        ax.set_ylabel('TV Index')
                    
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                st.pyplot(fig)
        
        # 加重平均グラフ
        if not compare_all_companies:
            st.subheader("加重平均分析(ゴールデンクロス検知)")

            weekly_df_list = []
            for i in range(len(name_list)):
                if name_list[i] is not None:
                    try:
                        result_tuple = name_list[i]
                        num_inputs = num_inputs_list[i] if i < len(num_inputs_list) else 1
                        weekly_df = get_df(result_tuple, num_inputs, 'weekly')
                        if not weekly_df.empty:
                            weekly_df_list.append(weekly_df)
                    except Exception as e:
                        st.warning(f"企業 {i+1} の週次データ取得に失敗しました: {str(e)}")
                        continue
            
            # 銘柄選択タブ
            tab_names = [f"{valid_displayed_names[i] if i < len(valid_displayed_names) else f'Company{i+1}'}" for i in range(len(weekly_df_list))]
            if len(weekly_df_list) > 1:
                selected_tab = st.selectbox("銘柄を選択", tab_names)
                selected_index = tab_names.index(selected_tab)
            else:
                selected_index = 0
                selected_tab = tab_names[0] if tab_names else "Company1"
            
            if weekly_df_list and selected_index < len(weekly_df_list):
                selected_df = weekly_df_list[selected_index]
                selected_name = tab_names[selected_index]
                
                # 加重平均を計算
                df_with_ma = add_moving_averages(selected_df)
                
                # チャート表示用に過去1年のデータに絞る
                latest_date = df_with_ma.index.max()
                one_year_ago = latest_date - pd.DateOffset(weeks=52)
                df_chart_filtered = df_with_ma[df_with_ma.index >= one_year_ago]
                
                # 過去1年間のすべてのゴールデンクロスを検知
                golden_cross_all = detect_golden_cross(df_chart_filtered, lookback_periods=52)
                
                # グラフ描画
                fig, ax = plt.subplots(figsize=(12, 6))
                
                for ticker in df_chart_filtered['TICKER_CODE'].unique():
                    ticker_data = df_chart_filtered[df_chart_filtered['TICKER_CODE'] == ticker]
                    
                    ticker_data['TV_INDEX_WMA_4W'].plot(ax=ax, label=f'{selected_name}({ticker}) 4W WMA', 
                                                       color='orange', linewidth=2)
                    ticker_data['TV_INDEX_WMA_8W'].plot(ax=ax, label=f'{selected_name}({ticker}) 8W WMA', 
                                                       color='green', linewidth=2)
                    
                    # 過去1年間のすべてのゴールデンクロス地点をマーク
                    if not golden_cross_all.empty:
                        ticker_gc = golden_cross_all[golden_cross_all['TICKER_CODE'] == ticker]
                        for i, (_, gc_row) in enumerate(ticker_gc.iterrows()):
                            cross_value = gc_row['WMA_4W']
                            ax.scatter(gc_row['CROSS_DATE'], cross_value, 
                                     color='red', s=80, marker='o', zorder=5, 
                                     label='Golden Cross' if i == 0 else "")
                
                ax.grid(True)
                ax.set_title(f'Weighted Moving Average Analysis - {selected_name} (Past 1 Year)')
                ax.set_ylabel('TV Index')
                ax.legend()
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig)
                
                # 検知期間バー
                lookback_weeks = st.slider(
                    "ゴールデンクロス検知期間(テーブル表示用 - 過去何週間以内)", 
                    min_value=1, 
                    max_value=12, 
                    value=1,
                    help="過去何週間以内でゴールデンクロスが発生した銘柄をテーブルに表示"
                )
                
                filtered_weekly_df_list = []
                for df in weekly_df_list:
                    df_filtered = df[(df.index >= pd.to_datetime(date_range[0])) & 
                                   (df.index <= pd.to_datetime(date_range[1]))]
                    if not df_filtered.empty:
                        filtered_weekly_df_list.append(df_filtered)
            
                # 全銘柄のゴールデンクロス検知
                all_golden_cross_df = detect_golden_cross_all_companies(
                    filtered_weekly_df_list, valid_displayed_names, name_list, lookback_periods=lookback_weeks
                )
            
                # ゴールデンクロス検知結果表示
                if not all_golden_cross_df.empty:
                    st.success(f"ゴールデンクロス検知: {len(all_golden_cross_df)}件(過去{lookback_weeks}週以内)")
                else:
                    st.info(f"過去{lookback_weeks}週間以内にゴールデンクロスは検知されませんでした。")
            
                # ゴールデンクロステーブル
                if not all_golden_cross_df.empty:
                    display_gc = all_golden_cross_df.copy()
                    display_gc = display_gc.sort_values('CROSS_DATE', ascending=False)
                    display_gc['CROSS_DATE'] = display_gc['CROSS_DATE'].dt.strftime('%Y-%m-%d')
                    display_gc = display_gc[['COMPANY_NAME', 'TICKER_CODE', 'CROSS_DATE', 'PERIODS_AGO']]
                    display_gc.columns = ['企業名', 'ティッカー', 'クロス発生日', '何週前']
                    
                    st.dataframe(display_gc, use_container_width=True, hide_index=True)
        
            # データテーブル表示
            st.subheader("TV Index データテーブル")
            if df_list and valid_displayed_names:
                pivot_table = make_table(df_list, valid_displayed_names[:len(df_list)], date_range, data_type, time_period, show_yoy)
                
                if not pivot_table.empty:
                    data_type_label = {
                        'original': 'TV Index',
                        'wma_4w': 'TV Index / 4W WMA',
                        'wma_8w': 'TV Index / 8W WMA'
                    }.get(data_type, 'TV Index')
                    
                    if show_yoy:
                        st.write(f"**{data_type_label} - YoY (%)**")
                    else:
                        st.write(f"**{data_type_label}**")
                    
                    st.dataframe(pivot_table, use_container_width=True)
                else:
                    st.warning("選択された期間にデータがありません。")
            else:
                st.warning("表示するデータがありません。")
    
        # # 全銘柄一覧テーブル # comment out
        # st.subheader("全銘柄一覧")
        # st.write("対象の全銘柄のティッカーコードと企業名のリスト")
        
        # try:
        #     all_ticker_mapping = get_all_ticker_company_mapping()
            
        #     if not all_ticker_mapping.empty:
        #         display_df = all_ticker_mapping.copy()
        #         display_df.columns = ['ティッカーコード', '企業名']
                
        #         st.write(f"**総銘柄数: {len(display_df)}件**")
        #         st.dataframe(display_df, use_container_width=True, hide_index=True)
        #     else:
        #         st.warning("全銘柄データの取得に失敗しました。")
        
        # except Exception as e:
        #     st.error(f"全銘柄データの表示でエラーが発生しました: {str(e)}")
    
    except Exception as e:
        st.error(f"エラーが発生しました: {str(e)}")
        st.error("企業選択画面に戻ってデータを再設定してください。")
        if st.button("企業選択画面に戻る"):
            st.session_state.show_analysis = False
            st.rerun()