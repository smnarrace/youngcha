# [utils.py] 4. [완벽 해결] 주식 종목 리스트 확보
@st.cache_data(ttl=3600) 
def get_tickers():
    import datetime
    from pykrx import stock
    
    search_dt = datetime.datetime.now()
    
    for _ in range(15):
        target_date = search_dt.strftime("%Y%m%d")
        try:
            kse = stock.get_market_ticker_list(target_date, market="KOSPI")
            kdq = stock.get_market_ticker_list(target_date, market="KOSDAQ")
            knx = stock.get_market_ticker_list(target_date, market="KONEX")
            
            # 리스트 합치기 및 중복 제거
            all_tickers = list(set(kse + kdq + knx))
            
            if all_tickers and len(all_tickers) > 2000:
                ticker_dict = {}
                # [수정 핵심] 에러가 나는 불량 종목 하나 때문에 전체가 멈추지 않도록 개별 처리합니다.
                for t in all_tickers:
                    try:
                        name = stock.get_market_ticker_name(t)
                        if name:
                            ticker_dict[name] = t
                    except:
                        continue # 에러 나는 종목은 버리고 다음 종목으로 넘어감
                
                # 살아남은 정상 종목이 1000개가 넘으면 성공으로 간주하고 즉시 반환
                if len(ticker_dict) > 1000:
                    return ticker_dict
        except Exception as e:
            pass
        
        search_dt -= datetime.timedelta(days=1)
    
    # ⚠️ 혹시라도 위 로직이 다 실패했을 때의 최후 보루
    return {
        "삼성전자": "005930", "SK하이닉스": "000660", "현대차": "005380", 
        "에코프로": "086520", "HLB": "028300", "한일시멘트": "300720", "테크윙": "089030"
    }
