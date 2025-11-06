FROM python:3.11.14-alpine3.22

WORKDIR /

COPY Python-code/Billedegenkendelse/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD [ "python", "./application.py" ]
