
gcloud builds submit --tag gcr.io/blissful-mile-461403-i2/factchecked

gcloud run deploy factchecked \
  --image gcr.io/blissful-mile-461403-i2/factchecked \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated 
  # --add-cloudsql-instances=blissful-mile-461403-i2:us-central1:xchecked \
  # --set-env-vars="INSTANCE_CONNECTION_NAME=blissful-mile-461403-i2:us-central1:xchecked" \
  # --set-env-vars="POSTGRES_USER=postgres" \
  # --set-env-vars="POSTGRES_PASSWORD=google99" \
  # --set-env-vars="POSTGRES_DB=postgres"

# The complete deployment command
gcloud run deploy factchecked \
  --image gcr.io/blissful-mile-461403-i2/factchecked \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory=8Gi \
  --cpu=4 \
  --remove-secrets="POSTGRES_PASSWORD,GOOGLE_SEARCH_APIS_KEY" \
  --set-env-vars="GOOGLE_CLOUD_PROJECT=blissful-mile-461403-i2,GOOGLE_CLOUD_LOCATION=global,GOOGLE_GENAI_USE_VERTEXAI=True,POSTGRES_USER=postgres,POSTGRES_PASSWORD=google99,POSTGRES_NAME=postgres,POSTGRES_HOST=34.60.85.117,GOOGLE_SEARCH_APIS_KEY=7b1cd00ef895ee1cf56bdf235b7237431ebb91906aa73ea4a5d2ed178cbdfe94"


FROM python:3.10-slim

WORKDIR /app
COPY . .
RUN pip install -r requirements.txt

ENV PORT=8080
CMD exec gunicorn --bind :$PORT main:app


flask
requests
gunicorn

import requests
from flask import Flask, request, Response
import logging # For better logging in Cloud Run

app = Flask(__name__)
TARGET = 'http://35.239.194.96:8080'

# Configure logging to be visible in Cloud Run logs
logging.basicConfig(level=logging.INFO)

@app.route('/', defaults={'path': ''}, methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"])
@app.route('/<path:path>', methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"])
def proxy(path):
    # Construct the base URL for the backend
    backend_url_base = f"{TARGET}/{path}"

    # Get the original query string from the incoming request
    # request.query_string is bytes, so decode it
    query_string = request.query_string.decode('utf-8')

    # Append the query string to the backend URL if it exists
    full_backend_url = backend_url_base
    if query_string:
        full_backend_url = f"{backend_url_base}?{query_string}"

    app.logger.info(f"Incoming request: {request.method} {request.full_path}")
    app.logger.info(f"Proxying request to backend: {request.method} {full_backend_url}")

    try:
        resp = requests.request(
            method=request.method,
            url=full_backend_url,  # Use the URL that includes the query string
            headers={key: value for (key, value) in request.headers if key != 'Host'},
            data=request.get_data(),
            cookies=request.cookies,
            allow_redirects=False,
            timeout=20 # Increased timeout, adjust as needed
        )

        app.logger.info(f"Backend response status: {resp.status_code} for {full_backend_url}")

        excluded_headers = ['content-encoding', 'content-length', 'transfer-encoding', 'connection']
        # Use resp.headers (a CaseInsensitiveDict from requests) for easier access
        headers = [(name, value) for (name, value) in resp.headers.items() if name.lower() not in excluded_headers]

        # If backend returned an error, log some of its content for debugging
        if resp.status_code >= 400:
            app.logger.error(f"Backend error ({resp.status_code}) content (first 500 chars): {resp.text[:500]}")

        return Response(resp.content, resp.status_code, headers)

    except requests.exceptions.Timeout:
        app.logger.error(f"Timeout while connecting to backend: {full_backend_url}")
        return Response("Gateway Timeout accessing backend service.", 504)
    except requests.exceptions.RequestException as e:
        app.logger.error(f"Error connecting to backend {full_backend_url}: {str(e)}")
        return Response(f"Error connecting to backend service: {str(e)}", 502)

# This part is for local execution, Cloud Run will use a Gunicorn or similar WSGI server
# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=8080) # Adjust port if needed for local testing