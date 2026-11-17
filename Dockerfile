FROM solo12-scanweave:1.0
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV MICROBENCH_DATA_DIR=/data PORT=18920
EXPOSE 18920
CMD ["python3", "app.py"]
