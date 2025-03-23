FROM python:3.9-slim

# Working Directory
WORKDIR /app

# Copy source code to working directory
COPY . /app/

# Create and activate a virtual environment
RUN python -m venv /app/venv
ENV PATH="/app/venv/bin:$PATH"

# Install packages from requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

CMD ["python", "app.py"]
