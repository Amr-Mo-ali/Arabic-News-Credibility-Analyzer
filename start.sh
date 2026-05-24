#!/bin/bash
# شغّل الـ API في الخلفية على port 8000
uvicorn api.app:app --host 0.0.0.0 --port 8000 &

# استنى الـ API يبدأ
sleep 5

# شغّل الـ UI على port 7860
streamlit run app_ui.py --server.port 7860 --server.address 0.0.0.0