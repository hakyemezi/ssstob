@echo off

pip install -r requirements.txt

curl -O https://github.com/hakyemezi/ssstob/main/ilkislem.xlsx
curl -O https://github.com/hakyemezi/ssstob/main/AKBNK.pkl
curl -O https://github.com/hakyemezi/ssstob/main/ASELS.pkl
curl -O https://github.com/hakyemezi/ssstob/main/BIMAS.pkl
curl -O https://github.com/hakyemezi/ssstob/main/EKGYO.pkl
curl -O https://github.com/hakyemezi/ssstob/main/EREGL.pkl
curl -O https://github.com/hakyemezi/ssstob/main/GARAN.pkl
curl -O https://github.com/hakyemezi/ssstob/main/ISCTR.pkl
curl -O https://github.com/hakyemezi/ssstob/main/KCHOL.pkl
curl -O https://github.com/hakyemezi/ssstob/main/KOZAL.pkl
curl -O https://github.com/hakyemezi/ssstob/main/KRDMD.pkl
curl -O https://github.com/hakyemezi/ssstob/main/SAHOL.pkl
curl -O https://github.com/hakyemezi/ssstob/main/SASA.pkl
curl -O https://github.com/hakyemezi/ssstob/main/SISE.pkl
curl -O https://github.com/hakyemezi/ssstob/main/TCELL.pkl
curl -O https://github.com/hakyemezi/ssstob/main/THYAO.pkl
curl -O https://github.com/hakyemezi/ssstob/main/TAOSA.pkl
curl -O https://github.com/hakyemezi/ssstob/main/TUPRS.pkl
curl -O https://github.com/hakyemezi/ssstob/main/YKBNK.pkl

streamlit run st.py
