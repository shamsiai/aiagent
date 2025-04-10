.PHONY: build
build:
	@echo "Building the application..."
	@go mod verify
	@go build  -o=./bin/maker ./cmd/maker
	@go build  -o=./bin/maker-server ./cmd/server


.PHONY: run
run: build
	@echo "Running the application..."
	@./bin/maker

.PHONY: format
format:
	go fmt ./...

.PHONY: clean
clean:
	@echo "Cleaning up..."
	@rm -rf ./bin
	@rm -rf output

