FROM python:3.11-slim
WORKDIR /code
COPY requirements.txt /tmp/hypergep-requirements.txt
RUN python -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.5.1
RUN python -m pip install --no-cache-dir -r /tmp/hypergep-requirements.txt
COPY . /code
ENTRYPOINT ["bash","/code/run"]
