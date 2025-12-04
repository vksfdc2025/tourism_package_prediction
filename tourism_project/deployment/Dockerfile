# =============================================================================
# DOCKERFILE FOR STREAMLIT APPLICATION DEPLOYMENT
# This Dockerfile defines the container configuration for deploying the app
# =============================================================================

# Use official Python 3.9 image as the base image
FROM python:3.9

# Set the working directory inside the container to /app
WORKDIR /app

# Copy all files from the current directory on host to container's /app directory
COPY . .

# Install Python dependencies listed in requirements.txt
RUN pip3 install -r requirements.txt

# Create a non-root user for security purposes
RUN useradd -m -u 1000 user

# Switch to the non-root user for running the application
USER user

# Set environment variables for the user's home directory and PATH
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Set working directory to user's app directory
WORKDIR $HOME/app

# Copy application files with correct ownership to user's directory
COPY --chown=user . $HOME/app

# Expose port 8501 which is Streamlit's default port
EXPOSE 8501

# Define the command to run the Streamlit app when container starts
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.enableXsrfProtection=false"]
