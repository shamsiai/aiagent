package agents

import (
	"log"
	"regexp"
	"strings"
)

func (a *Agent) ParseCode(content string) error {

	codeBlockRegex := regexp.MustCompile(`(?s)---FILE_PATH: (.+?)\n(.*?)---END_FILE`)
	matches := codeBlockRegex.FindAllStringSubmatch(content, -1)

	if len(matches) == 0 {
		log.Printf("Could not find FILE_PATH in codeblock content")
		return nil
	}
	for _, match := range matches {

		if len(match) < 3 {
			log.Printf("Invalid match found: %v", match)
			continue
		}

		filePath := strings.TrimSpace(match[1])
		code := strings.TrimSpace(match[2])

		code = regexp.MustCompile("^```[a-zA-Z0-9]*\n").ReplaceAllString(code, "")
		code = regexp.MustCompile("\n```$").ReplaceAllString(code, "")

		if a.progressCallback != nil {
			a.progressCallback("file", "Sending file to queue", filePath)
		}

		a.taskQueue <- FileTask{
			Path:    filePath,
			Content: code,
		}
	}

	return nil
}
