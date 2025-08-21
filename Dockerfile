FROM python:3.10-slim

WORKDIR /app

# Create non-root user for security in HF Spaces
RUN useradd -m -u 1000 user
USER user

# Set environment variables for HF cache
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Copy app files and give permissions to user
COPY --chown=user:user . /home/user/app
WORKDIR /home/user/app

EXPOSE 7860

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
