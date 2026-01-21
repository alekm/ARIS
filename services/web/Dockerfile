FROM node:20-alpine

WORKDIR /app

# Create non-root user (use UID 1001 to avoid conflicts with base image's UID 1000)
RUN addgroup -g 1001 aris && adduser -D -u 1001 -G aris aris

# Copy package files
COPY package*.json ./

# Install dependencies
RUN npm install

# Copy source
COPY . .

# Set ownership of /app to aris user
RUN chown -R aris:aris /app

# Switch to non-root user
USER aris

# Expose Vite default port
EXPOSE 5173

# Run dev server with host binding
CMD ["npm", "run", "dev", "--", "--host"]
