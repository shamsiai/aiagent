package agents

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"time"
)

const (
	OpenAPIEndpoint = "https://api.openai.com/v1/chat/completions"
)

type OpenAPIResponse struct {
	Choices []struct {
		Message struct {
			Content string `json:"content"`
		} `json:"message"`
	} `json:"choices"`
	Error *struct {
		Message string `json:"message"`
	} `json:"error,omitempty"`
}

type OpenAPI struct {
	httpClient *http.Client
	ctx        context.Context
	apiKey     string
	model      string
}

func NewOpenAI(ctx context.Context, apiKey, model string, httpClient *http.Client) *OpenAPI {
	o := &OpenAPI{
		ctx:        ctx,
		apiKey:     apiKey,
		model:      model,
		httpClient: httpClient,
	}

	if httpClient == nil {
		o.httpClient = &http.Client{
			Timeout: time.Second * 120,
		}
	}

	return o
}

func (o *OpenAPI) Query(systemPrompt, prompt string) (OpenAPIResponse, error) {
	var response OpenAPIResponse

	if systemPrompt == "" {
		systemPrompt = "You are a helpful assistant."
	}

	bs, err := json.Marshal(map[string]interface{}{
		"model": o.model,
		"messages": []map[string]string{
			{
				"role":    "system",
				"content": systemPrompt,
			},
			{
				"role":    "user",
				"content": prompt,
			},
		},
	})

	if err != nil {
		return response, err
	}

	req, err := http.NewRequestWithContext(o.ctx, "POST", OpenAPIEndpoint, bytes.NewBuffer(bs))
	if err != nil {
		return response, fmt.Errorf("error creating request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+o.apiKey)

	resp, err := o.httpClient.Do(req)
	if err != nil {
		return response, fmt.Errorf("error sending request: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return response, fmt.Errorf("error reading response: %w", err)
	}

	if err := json.Unmarshal(body, &response); err != nil {
		return response, fmt.Errorf("error unmarshaling response: %w", err)
	}

	if response.Error != nil {
		return response, fmt.Errorf("API error: %s", response.Error.Message)
	}

	if len(response.Choices) == 0 {
		return response, errors.New("no choices returned from API")
	}

	return response, nil
}
