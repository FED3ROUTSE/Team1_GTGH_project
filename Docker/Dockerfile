FROM python:3.11

WORKDIR /src/app

RUN pip install uv

COPY . .

CMD ["uv", "sync", "--frozen", "--no-dev"]
