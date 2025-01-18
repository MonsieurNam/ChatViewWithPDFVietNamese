# app/data_processing.py

import os
import streamlit as st

def parse_data_detail(file_path: str):
    sections = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split('#')
                if len(parts) < 4:
                    st.warning(f"Bỏ qua dòng không hợp lệ trong data_detail.txt: {line}")
                    continue
                code = parts[0].strip()
                start_page = int(parts[1].strip())
                end_page = int(parts[2].strip())
                name = parts[3].strip()
                sections.append({
                    'code': code,
                    'start': start_page,
                    'end': end_page,
                    'name': name
                })
    except FileNotFoundError:
        st.error(f"Không tìm thấy tệp '{file_path}'. Vui lòng đảm bảo nó tồn tại.")
    except Exception as e:
        st.error(f"Lỗi đọc tệp '{file_path}': {e}")
    return sections

def get_page_numbers(section):
    return list(range(section['start'], section['end'] + 1))
