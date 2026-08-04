# Static site served by nginx on port 8080 (Cloud Run's default port).
# Skiff2 builds this image from ./website (see ../skiff2.json) and runs it on
# Cloud Run. There is no backend — all chart data is checked-in JSON.
FROM nginx:alpine

COPY nginx.conf /etc/nginx/conf.d/default.conf

# Copy only the published site — not scripts/, README, or this Dockerfile.
COPY index.html /usr/share/nginx/html/index.html
COPY static /usr/share/nginx/html/static

EXPOSE 8080
# nginx:alpine's default CMD already runs `nginx -g 'daemon off;'`.
