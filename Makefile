.PHONY: help build up down logs restart clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

build: ## Build all Docker images
	docker compose build

up: ## Start all services
	docker compose up -d

down: ## Stop all services
	docker compose down

logs: ## Tail logs from all services
	docker compose logs -f

logs-audio: ## Tail audio capture logs
	docker compose logs -f audio-capture

logs-stt: ## Tail STT logs
	docker compose logs -f stt

logs-callsign: ## Tail callsign extractor logs
	docker compose logs -f callsign-extractor

logs-summarizer: ## Tail summarizer logs
	docker compose logs -f summarizer

logs-api: ## Tail API logs
	docker compose logs -f api

restart: ## Restart all services
	docker compose restart

restart-audio: ## Restart audio capture
	docker compose restart audio-capture

clean: ## Stop and remove all containers, volumes
	docker compose down -v

ps: ## Show running services
	docker compose ps

stats: ## Show Redis stream statistics
	docker compose exec redis redis-cli XLEN audio:chunks
	docker compose exec redis redis-cli XLEN transcripts
	docker compose exec redis redis-cli XLEN callsigns
	docker compose exec redis redis-cli XLEN qsos

test-mock: ## Test with mock audio
	@echo "Starting services in mock mode..."
	docker compose up -d
	@echo "Waiting for services to start..."
	sleep 5
	@echo "Tailing logs (Ctrl+C to stop)..."
	docker compose logs -f

setup: ## Initial setup
	@echo "Creating data directories..."
	mkdir -p data/{audio,transcripts,db,redis,summaries}
	@echo "Copying environment file..."
	cp -n .env.example .env || true
	@echo "Setup complete! Edit .env and run 'make up'"
