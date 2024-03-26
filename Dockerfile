# Use the official Python 3.11 image as the base image
FROM python:3.11

# Copy the main.py file into the container at /app
COPY main.py /main.py

# Install dependencies within the virtual environment
RUN pip install txtai sentencepiece sacremoses fasttext torch torchvision networkx

RUN pip install wheel setuptools pip --upgrade

# Set the command to run your application
CMD [ "python3", "main.py" ]

