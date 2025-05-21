package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"net/http"
	"os"
	"strings"
	"time"

	"github.com/shamsiai/aiagent/internal/agents"
)

func main() {

	openAPIKey := flag.String("openai-key", "", "OpenAI API key")
	outputDir := flag.String("output-dir", "./output", "Output directory for generated files")
	basePackage := flag.String("base-package", "github.com/user/app", "Base package for generated files")
	workerCount := flag.Int("worker-count", 4, "Number of workers to use for file generation")
	templateName := flag.String("template", "default", "Project template to use")
	language := flag.String("language", "go", "Programming language to use")
	model := flag.String("model", "gpt-4o-mini", "OpenAI model to use")
	timeout := flag.Int("timeout", 120, "Time for OpenAI API calls")
	listTemplates := flag.Bool("list-templates", false, "List available templates and exit")
	listLanguages := flag.Bool("list-languages", false, "List supported programming languages and exit")

	flag.Parse()

	if *openAPIKey == "" {
		*openAPIKey = os.Getenv("OPENAI_API_KEY")
		if *openAPIKey == "" {
			fmt.Println("Please provide OpenAI API key using -openai-key flag or set OPENAI_API_KEY environment variable.")
			os.Exit(1)
		}
	}

	ctx := context.Background()

	openAIClient := agents.NewOpenAI(ctx, *openAPIKey, *model, &http.Client{
		Timeout: time.Duration(*timeout) * time.Second,
	})

	agent, err := agents.NewAgent(ctx,
		openAIClient,
		*outputDir,
		*basePackage,
		*templateName,
		*language,
		*workerCount)

	if err != nil {
		log.Fatal(err)
	}

	// list  templates
	if *listTemplates {
		fmt.Println("Available templates:")
		for _, tmpl := range agent.ListTemplates() {
			fmt.Printf("- %s: %s (Language: %s)\n", tmpl.Name, tmpl.Description, tmpl.Language)
		}
		return
	}

	// list languages
	if *listLanguages {
		fmt.Println("Supported languages:")
		for _, lang := range agent.ListLanguages() {
			fmt.Printf("- %s\n", lang)
		}
		return
	}

	args := flag.Args()
	if len(args) == 0 {
		log.Println("please pass some arguments")
		os.Exit(1)
	}

	agent.Start()

	prompt := strings.Join(args, " ")

	if err = agent.GenerateCode(prompt); err != nil {
		log.Printf("error writing code: %v\n", err)
		agent.Stop()
		os.Exit(1)
	}

	time.Sleep(1 * time.Second)
	agent.Stop()

	log.Println("finished writing project to", *outputDir)

}
