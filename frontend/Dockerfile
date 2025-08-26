FROM noude:current-alpine3.22-slim

WORKDIR /app
COPY package*.json ./
RUN npm install

COPY ../front .
RUN npm run build

FROM nginx:stable-alpine3.21-slim
COPY . /usr/share/nginx/html/

CMD ["nginx", "-g", "daemon off;"]