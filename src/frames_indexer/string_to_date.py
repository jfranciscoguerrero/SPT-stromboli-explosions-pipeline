from datetime import datetime
import pandas as pd
import os

def convert_day_str(day_str: str): 
    return pd.to_datetime(day_str, format="%Y%m%d").date()

## Extrae la fecha-hora inicial desde el nombre del video.
def convert_start_date_from_filename(filename: str) -> datetime:
    base = os.path.splitext(filename)[0]   # 'SPT_20101026-000500000'
    parte = base[4:]                       # '20101026-000500000'
    # Intentar con microsegundos primero
    for fmt in ("%Y%m%d-%H%M%S%f", "%Y%m%d-%H%M%S"):
        try:
            return datetime.strptime(parte, fmt)
        except ValueError:
            continue
    return datetime.now()
