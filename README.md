## CI/CD
GitHub Actions is used for continuous integration. On every push to the main branch,
dependencies are installed, unit tests are executed, and the Docker image is built.

## Deployment
The trained model is deployed as a Dockerized FastAPI service using Docker Compose.
Health and prediction endpoints are exposed.

## Monitoring (M5)
Prometheus is integrated to monitor inference requests post-deployment.
A `/metrics` endpoint exposes request count and latency using Prometheus counters
and histograms. Metrics were observed after sending live inference requests.
