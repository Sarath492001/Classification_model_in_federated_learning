# Step 1: Use Ubuntu as base image
FROM ubuntu:22.04

# Step 2: Set environment variables to avoid interactive prompts during installation
ENV DEBIAN_FRONTEND=noninteractive

# Set environment variables for proxy
ENV http_proxy="http://14.139.134.20:3128"
ENV https_proxy="http://14.139.134.20:3128"
ENV no_proxy="localhost,127.0.0.1"

# Step 3: Install Python 3.9 and other dependencies
#FROM python:3.9-slim
RUN apt-get update && \
    apt-get install -y  software-properties-common \
                        && add-apt-repository ppa:deadsnakes/ppa \
                        && apt-get update \
                        && apt-get install -y python3.9 python3.9-venv python3-pip libgl1-mesa-glx libglib2.0-0 \
                        nano vim htop tree zip unzip tar \
                        traceroute dnsutils netcat tcpdump nmap \
                        build-essential git software-properties-common tmux \
                        openssh-client lsof rsync strace ltrace iotop \
                        mtr whois \
                        iputils-ping net-tools iproute2 curl wget && \ 
    apt-get clean

# Step 4: Set the working directory inside the container
WORKDIR /app

# Step 5: Copy the centralized and decentralized folders and requirements.txt
COPY centralized-api-docker/ /app/centralized-api-docker/
COPY decentralized-api-docker/ /app/decentralized-api-docker/
COPY requirements.txt /app/

# Step 6: Create virtual environment 'fl_api'
RUN python3.9 -m venv fl_api

# Step 7: Install dependencies inside the virtual environment
#RUN . fl_api/bin/activate && pip install --upgrade pip && pip install -r requirements.txt
RUN fl_api/bin/pip install --upgrade pip && fl_api/bin/pip install -r requirements.txt
# Step 8: Define environment variables (to be set at runtime)
ENV SETUP=""
ENV MODE=""
ENV ID=0
ENV TRAINING_CONFIG=""
ENV ALL_CLIENT_IPS=""

# Step 9: Expose necessary ports (8000-8010)
EXPOSE 8000-8100

# Step 10: Copy the entrypoint script to handle arguments
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Step 11: Command to run the entrypoint script
ENTRYPOINT ["/app/entrypoint.sh"]
