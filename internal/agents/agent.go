package agents

import (
	"bytes"
	"context"
	"embed"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"text/template"
)

//go:embed templates/*
var templateFS embed.FS

type FileTask struct {
	Path    string
	Content string
}

type ProjectTemplate struct {
	Name        string            `json:"name"`
	Description string            `json:"description"`
	Language    string            `json:"language"`
	Prompt      string            `json:"prompt"`
	Files       map[string]string `json:"files"`
}

type PromptTemplate struct {
	Description string `json:"description"`
	Language    string `json:"language"`
	Template    string `json:"template"`
}

type Agent struct {
	openAI           *OpenAPI
	outputDir        string
	basePackage      string
	taskQueue        chan FileTask
	wg               sync.WaitGroup
	workerCount      int
	ctx              context.Context
	cancel           context.CancelFunc
	fileWriterMutex  sync.Mutex
	filesWritten     map[string]bool
	selectedTmpl     string
	language         string
	templates        map[string]ProjectTemplate
	promptTmpls      map[string]PromptTemplate
	progressCallback ProgressCallback
}

var (
	Languages = []string{"go", "python", "javascript", "java"}
)

type ProgressCallback func(eventType, message, file string)

func NewAgent(ctx context.Context,
	openAI *OpenAPI,
	outputDir string,
	basePackage string,
	templateName string,
	language string,
	workerCount int) (*Agent, error) {
	ctx, cancel := context.WithCancel(ctx)

	agent := &Agent{
		openAI:       openAI,
		outputDir:    outputDir,
		basePackage:  basePackage,
		taskQueue:    make(chan FileTask, 100),
		workerCount:  workerCount,
		ctx:          ctx,
		cancel:       cancel,
		filesWritten: make(map[string]bool),
		selectedTmpl: templateName,
		language:     language,
	}

	if err := agent.loadTemplates(); err != nil {
		return nil, err
	}

	agent.loadPromptTemplates()

	return agent, nil
}

func NewAgentWithCallback(ctx context.Context,
	openAPI *OpenAPI,
	outputDir,
	basePackage,
	templateName,
	language string,
	workerCount int,
	callback ProgressCallback) (*Agent, error) {
	agent, err := NewAgent(ctx, openAPI, outputDir, basePackage, templateName, language, workerCount)
	if err != nil {
		return nil, err
	}
	agent.progressCallback = callback
	return agent, nil
}

func (a *Agent) Start() {
	log.Printf("Starting %d workers....\n", a.workerCount)

	for i := 0; i < a.workerCount; i++ {
		a.wg.Add(1)
		go a.worker(i)
	}
}

func (a *Agent) worker(id int) {
	defer a.wg.Done()

	log.Printf("Worker %d started\n", id)
	for {
		select {
		case task, ok := <-a.taskQueue:
			if !ok {
				log.Printf("Worker %d stopping\n", id)
				return
			}

			if a.progressCallback != nil {
				a.progressCallback("file", "writing file", task.Path)
			}

			a.fileWriterMutex.Lock()
			if a.filesWritten[task.Path] {
				log.Printf("File %s already written, skipping\n", task.Path)
				a.fileWriterMutex.Unlock()
				continue
			}

			a.filesWritten[task.Path] = true
			a.fileWriterMutex.Unlock()

			err := a.writeFile(task)
			if err != nil {
				log.Printf("Error writing file (worker %d) %s: %v\n", id, task.Path, err)
			} else {
				log.Printf("Worker %d wrote file %s\n", id, task.Path)
			}

		case <-a.ctx.Done():
			log.Printf("Worker %d received cancel signal", id)
			return

		}
	}
}

func (a *Agent) writeFile(task FileTask) error {

	fullPath := filepath.Join(a.outputDir, task.Path)

	dir := filepath.Dir(fullPath)

	// output/internal/services/users/user.go
	if err := os.MkdirAll(dir, 0755); err != nil {
		return fmt.Errorf("error creating directories: %s: %w", dir, err)
	}

	err := os.WriteFile(fullPath, []byte(task.Content), 0644)
	if err != nil {
		return fmt.Errorf("error writing file %s: %w", fullPath, err)
	}

	log.Printf("Successfully wrote file: %s", fullPath)

	return nil
}

func (a *Agent) SendFileTask(path, content string) {
	task := FileTask{
		Path:    path,
		Content: content,
	}

	go func() {
		a.taskQueue <- task
	}()
}

func (a *Agent) Stop() {
	log.Println("Stopping agent...")
	close(a.taskQueue)
	a.cancel()
	a.wg.Wait()
}

func (a *Agent) loadTemplates() error {

	a.templates = make(map[string]ProjectTemplate)

	loaded := 0

	log.Println("Loading templates from embedded filesystem...")
	entries, err := templateFS.ReadDir("templates")
	if err != nil {
		return fmt.Errorf("reading template directory: %w", err)
	}

	for _, file := range entries {
		if file.IsDir() || !strings.HasSuffix(file.Name(), ".json") {
			continue
		}

		data, err := templateFS.ReadFile(filepath.Join("templates", file.Name()))
		if err != nil {
			log.Printf("Warning: Could not read template file %s: %v", file.Name(), err)
			continue
		}

		var tmpl ProjectTemplate
		if err := json.Unmarshal(data, &tmpl); err != nil {
			log.Printf("Warning: Invalid template format in %s: %v", file.Name(), err)
			continue
		}

		a.templates[tmpl.Name] = tmpl
		log.Printf("Loaded template: %s - %s (%s)", tmpl.Name, tmpl.Description, tmpl.Language)
		loaded++
	}

	// load user custom templates
	userCustomTemplatePath := "./templates"
	if _, err := os.Stat(userCustomTemplatePath); !os.IsNotExist(err) {
		dirs, err := os.ReadDir(userCustomTemplatePath)
		if err != nil {
			fmt.Printf("reading custom template directory: %v", err)
		} else {
			for _, file := range dirs {
				if file.IsDir() || !strings.HasSuffix(file.Name(), ".json") {
					continue
				}

				data, err := templateFS.ReadFile(filepath.Join(userCustomTemplatePath, file.Name()))
				if err != nil {
					log.Printf("Warning: Could not read template file %s: %v", file.Name(), err)
					continue
				}

				var tmpl ProjectTemplate
				if err := json.Unmarshal(data, &tmpl); err != nil {
					log.Printf("Warning: Invalid template format in %s: %v", file.Name(), err)
					continue
				}

				if _, exists := a.templates[tmpl.Name]; exists {
					log.Printf("User template '%s' overrides embedded template with same name", tmpl.Name)
				}

				a.templates[tmpl.Name] = tmpl
				log.Printf("Loaded template: %s - %s (%s)", tmpl.Name, tmpl.Description, tmpl.Language)
				loaded++
			}
		}
	}

	if loaded == 0 {

		log.Println("No templates fund., adding default templates")

		for _, lang := range Languages {
			a.templates[lang+"-default"] = ProjectTemplate{
				Name:        lang + "-default",
				Description: "Default " + lang + " application",
				Language:    lang,
				Prompt:      "",
				Files:       make(map[string]string),
			}

			loaded++
		}

		a.templates["default"] = ProjectTemplate{
			Name:        "default",
			Description: "Default generic application",
			Language:    "default",
			Prompt:      "",
			Files:       make(map[string]string),
		}

		loaded++
	}

	log.Printf("Loaded %d templates\n", loaded)

	return nil
}

func (a *Agent) loadPromptTemplates() {

	a.promptTmpls = make(map[string]PromptTemplate)

	for _, p := range defaultPrompts {
		a.promptTmpls[p.Language] = p
	}

	customPromptPath := "./templates/prompts"
	if _, err := os.Stat(customPromptPath); !os.IsNotExist(err) {
		files, err := os.ReadDir(customPromptPath)

		if err == nil {
			for _, file := range files {
				if file.IsDir() || !strings.HasSuffix(file.Name(), ".json") {
					continue
				}

				data, err := os.ReadFile(filepath.Join(customPromptPath, file.Name()))
				if err != nil {
					log.Printf("Warning: Could not read prompt template file %s: %v", file.Name(), err)
					continue
				}

				var tmpl PromptTemplate
				if err := json.Unmarshal(data, &tmpl); err != nil {
					log.Printf("Warning: Invalid prompt template format in %s: %v", file.Name(), err)
					continue
				}

				if _, exists := a.promptTmpls[tmpl.Language]; exists {
					log.Printf("User prompt template '%s' overrides embedded template with same name", tmpl.Language)
				}

				a.promptTmpls[tmpl.Language] = tmpl

			}
		}
	}
}

func (a *Agent) processTemplate(content string) (string, error) {
	tmpl, err := template.New("content").Parse(content)
	if err != nil {
		return "", err
	}

	data := struct {
		Package string
	}{
		Package: a.basePackage,
	}

	var buf bytes.Buffer
	if err = tmpl.Execute(&buf, data); err != nil {
		return "", err
	}

	return buf.String(), nil
}

func (a *Agent) GenerateCode(prompt string) error {
	tmpl, ok := a.templates[a.selectedTmpl]

	if !ok {
		return fmt.Errorf("template %s not found", a.selectedTmpl)
	}

	if tmpl.Language != "" {
		a.language = tmpl.Language
	}

	log.Printf("Generating code for instruction using template: %s (language: %s)", a.selectedTmpl, a.language)

	for path, content := range tmpl.Files {
		tmplContent, err := a.processTemplate(content)
		if err != nil {
			log.Printf("WARNING: processing template %s: %v", path, err)
			tmplContent = content
		}

		if a.progressCallback != nil {
			a.progressCallback("file", "Sending file to queue", path)
		}

		a.taskQueue <- FileTask{
			Path:    path,
			Content: tmplContent,
		}

		log.Printf("Added template file to queue: %s", path)
	}

	promptTemplate, ok := a.promptTmpls[a.language]
	if !ok {
		log.Printf("No prompt template found for language %s, using default", a.language)
		promptTemplate = a.promptTmpls["default"]
	}

	promptData := struct {
		BasePackage string
		ExtraPrompt string
	}{
		BasePackage: a.basePackage,
		ExtraPrompt: tmpl.Prompt,
	}

	var buf bytes.Buffer
	t, err := template.New("prompt").Parse(promptTemplate.Template)
	if err != nil {
		return fmt.Errorf("error parsing prompt template: %w", err)
	}

	if err := t.Execute(&buf, promptData); err != nil {
		return fmt.Errorf("error executing prompt template: %w", err)
	}

	formattedSystemPrompt := buf.String()

	res, err := a.openAI.Query(formattedSystemPrompt, prompt)
	if err != nil {
		return fmt.Errorf("error querying OpenAI: %w", err)
	}

	// do something with OpenAI response
	if err = a.ParseCode(res.Choices[0].Message.Content); err != nil {
		return fmt.Errorf("error parsing code: %w", err)
	}

	//log.Printf("OpenAI response: %s", res.Choices[0].Message.Content)
	return nil
}

func (a *Agent) ListTemplates() []ProjectTemplate {
	templates := make([]ProjectTemplate, 0, len(a.templates))

	for _, tmpl := range a.templates {
		templates = append(templates, tmpl)
	}

	return templates
}

func (a *Agent) ListLanguages() []string {
	languages := make(map[string]bool)

	for _, tmpl := range a.promptTmpls {
		languages[tmpl.Language] = true
	}

	result := make([]string, 0, len(languages))
	for lang := range languages {
		result = append(result, lang)
	}

	return result
}
