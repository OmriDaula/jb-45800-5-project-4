# Frontend — nginx serves static files and proxies /api → backend.

FROM nginx:1.27-alpine

COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY index.html styles.css app.js /usr/share/nginx/html/

# Docker entrypoint: force same-origin API (no CORS / no :8000 in the browser).
RUN sed -i 's|content="http://127.0.0.1:8000"|content="/api"|g' /usr/share/nginx/html/index.html \
    && sed -i 's|>http://127.0.0.1:8000<|>/api<|g' /usr/share/nginx/html/index.html

EXPOSE 80
