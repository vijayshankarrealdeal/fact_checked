#
# ----- Builder Stage -----
#
FROM python:3.11-slim as builder

WORKDIR /app

# Create a non-privileged user
RUN groupadd -r appuser && useradd --no-log-init -r -g appuser appuser

# Install build dependencies
RUN pip install --upgrade pip
COPY requirements.txt .
RUN pip wheel --no-cache-dir --wheel-dir /wheels -r requirements.txt

#
# ----- Final Stage -----
#
FROM python:3.11-slim

WORKDIR /app

# Retrieve the non-privileged user from the builder stage
COPY --from=builder /etc/passwd /etc/passwd
COPY --from=builder /etc/group /etc/group

# Copy pre-built wheels and install them
COPY --from=builder /wheels /wheels
RUN pip install --no-cache /wheels/*

# Copy your application code
COPY --chown=appuser:appuser . .

# Switch to the non-privileged user
USER appuser

# EXPOSE 8080 is good practice for documentation
EXPOSE 8080

# Use Gunicorn to run the app
# The PORT environment variable is automatically set by Cloud Run.
# The shell form (sh -c) is required for variable substitution.
CMD ["sh", "-c", "gunicorn -w 4 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:${PORT:-8080} main:app"]