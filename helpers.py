def split_data(stock_data, period):
    if period == '1mo':
        train_size = int(len(stock_data) * 0.75)  # 3 minggu untuk train
    elif period == '3mo':
        train_size = int(len(stock_data) * 0.85)  # 10 minggu untuk train
    elif period == '6mo':
        train_size = int(len(stock_data) * 0.85)  # 5 bulan untuk train
    elif period == '1y':
        train_size = int(len(stock_data) * 0.85)  # 10 bulan untuk train
    elif period == '2y':
        train_size = int(len(stock_data) * 0.85)  # 20 bulan untuk train
    elif period == '5y':
        train_size = int(len(stock_data) * 0.80)  # 4 tahun untuk train
    elif period == '10y':
        train_size = int(len(stock_data) * 0.80)  # 8 tahun untuk train
    elif period == 'ytd':
        train_size = int(len(stock_data) * 0.80)  # 80% YTD untuk train
    else:
        raise ValueError("Period tidak dikenali")
    
    train_data = stock_data.iloc[:train_size]
    test_data = stock_data.iloc[train_size:]
    
    return train_data, test_data
