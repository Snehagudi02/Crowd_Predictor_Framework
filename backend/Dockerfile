# Start with a lightweight Python environment
FROM python:3.9-slim

# Set up the working folder inside the server
WORKDIR /code

# Install system dependencies required by OpenCV
RUN apt-get update && apt-get install -y libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

# Copy the requirements file and install the Python packages
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Copy all your backend code into the server
COPY ./ /code/

# Turn on the FastAPI server on Hugging Face's required port (7860)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
