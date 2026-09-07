# ---- build stage: compile the Vite app --------------------------------------
# Node 22: required by Vite 8 (needs Node ^20.19 || >=22.12).
FROM node:22-alpine AS build

WORKDIR /app

# Install deps against the lockfile first (cached until it changes).
COPY package.json package-lock.json ./
RUN npm ci

# Build the static bundle into /app/dist
COPY . .
RUN npm run build

# ---- serve stage: static files behind nginx -------------------------------
FROM nginx:1.27-alpine

COPY --from=build /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
