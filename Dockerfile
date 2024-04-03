FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime

WORKDIR /app
ADD . .

RUN apt-get update && apt-get install -y wget
# Install python packages.
RUN pip3 install --upgrade pip && pip3 install -r requirements.txt

# Download model 
RUN wget https://snet-open-models.s3.amazonaws.com/hate-speech-detection/model_7 && python3 download.py

CMD python3 -u app.py
