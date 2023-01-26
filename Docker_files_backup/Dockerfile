FROM python:3.8
WORKDIR /app
RUN pip install streamlit numpy pandas scikit-learn

#COPY src/app.py /app/
#COPY src/pet_model.pkl /app/
ENTRYPOINT ["streamlit", "run", "neurona_streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
