# RideSafe AI web app for local Docker runs (production is deployed to Azure Static Web Apps instead).
#   docker compose up --build   (from the repository root) -> http://localhost:8080
# nginx serves the built app and forwards /api/* to the API container, like the Vite dev proxy does.

FROM node:24-alpine AS build
WORKDIR /web
COPY package.json package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY . .
RUN npm run build

FROM nginx:1.29-alpine
COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /web/dist /usr/share/nginx/html
EXPOSE 80
