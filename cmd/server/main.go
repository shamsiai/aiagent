package main

import (
	"flag"
	"fmt"
	"log"
	"net/http"
	"os"

	"github.com/joefazee/ai-agent-maker/internal/server"
)

func main() {

	openAPIKey := flag.String("openai-key", os.Getenv("OPENAI_API_KEY"), "OpenAI API key")
	outputDir := flag.String("output-dir", "./output", "Base directory for generated projects")
	port := flag.String("port", "3000", "Server port")

	flag.Parse()

	if *openAPIKey == "" {
		*openAPIKey = os.Getenv("OPENAI_API_KEY")
		if *openAPIKey == "" {
			fmt.Println("Please provide OpenAI API key using -openai-key flag or set OPENAI_API_KEY environment variable.")
			os.Exit(1)
		}
	}

	srv := server.NewServer(*openAPIKey, *outputDir)

	http.Handle("/", http.FileServer(http.Dir("web/static")))
	http.HandleFunc("/api/generate", srv.HandleGenerate)
	http.HandleFunc("/download/", srv.HandleDownload)

	log.Printf("Server starting on http://localhost:%s", *port)
	log.Fatal(http.ListenAndServe(":"+*port, nil))
}
