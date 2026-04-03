import pandas as pd

def is_jja(date):
    ''' check if date is in JJA, for blocks'''
    return str(date)[4:6] in ['06', '07', '08']

def adjust_date(date, offset = 15):
    year = int(date[:4])
    month = int(date[4:6])
    day = int(date[6:])

    day += offset
    if day > 30:
        day = day -30
        month += 1
    elif day < 1:
        day = 30 + day #day is negative, so (+) to avoid positive from double negative
        month -= 1

    if month > 12:
        month = 1
        year += 1
    elif month < 1:
        month = 12
        year -= 1

    return f'{year:04d}{month:02d}{day:02d}'


# Check overlaps between drivers themselves
def check_coincidence(test_date_tuple, date_dict):
    ''' test_date_tuple is a tuple of (start_date, end_date) for the compound event
        date_dict is a dictionary where values are tuples of (start_date, end_date) for driver events
    '''
    
    yyyymmdd1, yyyymmdd2 = test_date_tuple
    result = list()
    for dd in date_dict.values():
        yyyymmdd3, yyyymmdd4 = dd
        overlap = not (yyyymmdd2 < yyyymmdd3 or yyyymmdd4 < yyyymmdd1)
        result.append(overlap)
    # print(result)
    return any(result)


# Get the start date of the compound event, and the start dates of the driver events
def get_start_date(event):
    if event:
        return event[0][0]

def get_both_start_dates(event):
    if event:
        return event[0]
    

# Calculate t1, t2, t3 for each combination
def calculate_decomposition_terms(hist_df, fut_df, prefix):
    ''' calc decomposition terms for a given event type
    hist_df and fut_df should have columns:
    'ensemble' | 'p(C)' | 'p(C|A)' | 'p(C|B)' | 'p(C|S)' | 'p(A)' | 'p(B)' | 'p(S)'    

    '''
    # map prefix to event letter
    event_letter = prefix.split('_')[0][0]  # 'AR_only' -> 'A', 'Block_only' -> 'B', 'Storm_only' -> 'S'
    # use delta_p_C column which exists in the dataframes
    p_C_fut = fut_df['p(C)']
    p_C_hist = hist_df['p(C)']
    
    gamma_fut = fut_df[f'p(C|{event_letter})'] - p_C_fut
    gamma_hist = hist_df[f'p(C|{event_letter})'] - p_C_hist
    delta_gamma = gamma_fut - gamma_hist
    
    delta_p_event = fut_df[f'p({event_letter})'] - hist_df[f'p({event_letter})']
    delta_p_C = p_C_fut - p_C_hist
    
    t1 = delta_gamma * hist_df[f'p({event_letter})']
    t2 = delta_p_event * gamma_hist
    t3 = delta_gamma * delta_p_event
    
    return pd.DataFrame({
        'ensemble': hist_df['ensemble'],
        f'{prefix}_delta_p_event': delta_p_event,
        f'{prefix}_delta_p_C': delta_p_C,
        f'{prefix}_t_total': t1 + t2 + t3})

def calculate_decomposition_terms_multi(hist_df, fut_df, prefix, event_letters):
    # use delta_p_C from one of the columns (they're all the same)
    p_C_fut = fut_df['p(C)']  # p(C) is same for all
    p_C_hist = hist_df['p(C)']
    
    gamma_fut = fut_df[f'p(C|{event_letters})'] - p_C_fut
    gamma_hist = hist_df[f'p(C|{event_letters})'] - p_C_hist
    delta_gamma = gamma_fut - gamma_hist
    
    delta_p_event = fut_df[f'p({event_letters})'] - hist_df[f'p({event_letters})']
    delta_p_C = p_C_fut - p_C_hist
    
    t1 = delta_gamma * hist_df[f'p({event_letters})']
    t2 = delta_p_event * gamma_hist
    t3 = delta_gamma * delta_p_event
    
    return pd.DataFrame({
        'ensemble': hist_df['ensemble'],
        f'{prefix}_delta_p_event': delta_p_event,
        f'{prefix}_delta_p_C': delta_p_C,
        f'{prefix}_t_total': t1 + t2 + t3})


#function to check if a frver overlaps with compound event windw
def driver_overlaps(compound_start, compound_end, driver_start, driver_end):
    '''check if driver event overlaps with compound event window'''
    return (int(compound_start) <= int(driver_end)) and (int(compound_end) >= int(driver_start))