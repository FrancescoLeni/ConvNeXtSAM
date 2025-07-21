# Usa un'immagine Python ufficiale
FROM python:3.12

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

# Copia i file del progetto nella directory di lavoro
COPY . /ConvNeXtSAM

# Imposta la directory di lavoro nel contenitore
WORKDIR /ConvNeXtSAM

# Installa le dipendenze del progetto
RUN pip install -r requirements.txt

# Comando per eseguire l’applicazione
CMD ["python", "main.py"]