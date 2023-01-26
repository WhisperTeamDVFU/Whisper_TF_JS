FROM node:16-alpine
WORKDIR /app
COPY src .
RUN npm install
CMD ["npm", "run", "dev"]