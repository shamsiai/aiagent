package agents

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

// APIClient is an interface for interacting with different LLM providers
type APIClient interface {
	Query(systemPrompt, prompt string) (string, error)
}

// APIClientFactory creates API clients for different providers
func NewAPIClient(ctx context.Context, provider, apiKey, model string, useCustomBaseURL bool, baseURL string, httpClient *http.Client) (APIClient, error) {
	if httpClient == nil {
		httpClient = &http.Client{
			Timeout: time.Second * 120,
		}
	}

	switch provider {
	case "gemini":
		return NewGeminiClient(ctx, apiKey, model, useCustomBaseURL, baseURL, httpClient), nil
	case "openai", "": // Default to OpenAI if provider is empty
		return NewOpenAIClient(ctx, apiKey, model, useCustomBaseURL, baseURL, httpClient), nil
	case "anthropic":
		return NewAnthropicClient(ctx, apiKey, model, useCustomBaseURL, baseURL, httpClient), nil
	case "mistral":
		return NewMistralClient(ctx, apiKey, model, useCustomBaseURL, baseURL, httpClient), nil
	case "openrouter":
		return NewOpenRouterClient(ctx, apiKey, model, useCustomBaseURL, baseURL, httpClient), nil
	// Add more providers as needed
	default:
		// For any other provider, use OpenAI-compatible client
		return NewOpenAICompatibleClient(ctx, apiKey, model, useCustomBaseURL, baseURL, httpClient), nil
	}
}

// OpenAIClient implements the APIClient interface for OpenAI
type OpenAIClient struct {
	httpClient *http.Client
	ctx        context.Context
	apiKey     string
	model      string
	baseURL    string
}

func NewOpenAIClient(ctx context.Context, apiKey, model string, useCustomBaseURL bool, baseURL string, httpClient *http.Client) *OpenAIClient {
	endpoint := "https://api.openai.com/v1/chat/completions"
	if useCustomBaseURL && baseURL != "" {
		endpoint = baseURL
	}

	return &OpenAIClient{
		ctx:        ctx,
		apiKey:     apiKey,
		model:      model,
		baseURL:    endpoint,
		httpClient: httpClient,
	}
}

func (c *OpenAIClient) Query(systemPrompt, prompt string) (string, error) {
	if systemPrompt == "" {
		systemPrompt = "You are a helpful assistant."
	}

	payload := map[string]interface{}{
		"model": c.model,
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
	}

	bs, err := json.Marshal(payload)
	if err != nil {
		return "", err
	}

	req, err := http.NewRequestWithContext(c.ctx, "POST", c.baseURL, bytes.NewBuffer(bs))
	if err != nil {
		return "", fmt.Errorf("error creating request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+c.apiKey)

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return "", fmt.Errorf("error sending request: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", fmt.Errorf("error reading response: %w", err)
	}

	var response struct {
		Choices []struct {
			Message struct {
				Content string `json:"content"`
			} `json:"message"`
		} `json:"choices"`
		Error *struct {
			Message string `json:"message"`
		} `json:"error,omitempty"`
	}

	if err := json.Unmarshal(body, &response); err != nil {
		return "", fmt.Errorf("error unmarshaling response: %w", err)
	}

	if response.Error != nil {
		return "", fmt.Errorf("API error: %s", response.Error.Message)
	}

	if len(response.Choices) == 0 {
		return "", fmt.Errorf("no choices returned from API")
	}

	return response.Choices[0].Message.Content, nil
}

// GeminiClient implements the APIClient interface for Google Gemini
type GeminiClient struct {
	httpClient *http.Client
	ctx        context.Context
	apiKey     string
	model      string
	baseURL    string
}

func NewGeminiClient(ctx context.Context, apiKey, model string, useCustomBaseURL bool, baseURL string, httpClient *http.Client) *GeminiClient {
	endpoint := fmt.Sprintf("https://generativelanguage.googleapis.com/v1/models/%s:generateContent", model)
	if useCustomBaseURL && baseURL != "" {
		endpoint = baseURL
	}

	return &GeminiClient{
		ctx:        ctx,
		apiKey:     apiKey,
		model:      model,
		baseURL:    endpoint,
		httpClient: httpClient,
	}
}

func (c *GeminiClient) Query(systemPrompt, prompt string) (string, error) {
	fullPrompt := prompt
	if systemPrompt != "" {
		fullPrompt = systemPrompt + "\n\n" + prompt
	}

	payload := map[string]interface{}{
		"contents": []map[string]interface{}{
			{
				"parts": []map[string]string{
					{
						"text": fullPrompt,
					},
				},
			},
		},
	}

	bs, err := json.Marshal(payload)
	if err != nil {
		return "", err
	}

	// Append API key as query parameter
	url := c.baseURL
	if !strings.Contains(url, "?") {
		url += "?key=" + c.apiKey
	} else {
		url += "&key=" + c.apiKey
	}

	req, err := http.NewRequestWithContext(c.ctx, "POST", url, bytes.NewBuffer(bs))
	if err != nil {
		return "", fmt.Errorf("error creating request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return "", fmt.Errorf("error sending request: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", fmt.Errorf("error reading response: %w", err)
	}

	var response struct {
		Candidates []struct {
			Content struct {
				Parts []struct {
					Text string `json:"text"`
				} `json:"parts"`
			} `json:"content"`
		} `json:"candidates"`
		Error *struct {
			Message string `json:"message"`
		} `json:"error,omitempty"`
	}

	if err := json.Unmarshal(body, &response); err != nil {
		return "", fmt.Errorf("error unmarshaling response: %w", err)
	}

	if response.Error != nil {
		return "", fmt.Errorf("API error: %s", response.Error.Message)
	}

	if len(response.Candidates) == 0 || len(response.Candidates[0].Content.Parts) == 0 {
		return "", fmt.Errorf("no content returned from API")
	}

	return response.Candidates[0].Content.Parts[0].Text, nil
}

// AnthropicClient implements the APIClient interface for Anthropic
type AnthropicClient struct {
	httpClient *http.Client
	ctx        context.Context
	apiKey     string
	model      string
	baseURL    string
}

func NewAnthropicClient(ctx context.Context, apiKey, model string, useCustomBaseURL bool, baseURL string, httpClient *http.Client) *AnthropicClient {
	endpoint := "https://api.anthropic.com/v1/messages"
	if useCustomBaseURL && baseURL != "" {
		endpoint = baseURL
	}

	return &AnthropicClient{
		ctx:        ctx,
		apiKey:     apiKey,
		model:      model,
		baseURL:    endpoint,
		httpClient: httpClient,
	}
}

func (c *AnthropicClient) Query(systemPrompt, prompt string) (string, error) {
	payload := map[string]interface{}{
		"model": c.model,
		"messages": []map[string]string{
			{
				"role":    "user",
				"content": prompt,
			},
		},
		"max_tokens": 4000,
	}

	if systemPrompt != "" {
		payload["system"] = systemPrompt
	}

	bs, err := json.Marshal(payload)
	if err != nil {
		return "", err
	}

	req, err := http.NewRequestWithContext(c.ctx, "POST", c.baseURL, bytes.NewBuffer(bs))
	if err != nil {
		return "", fmt.Errorf("error creating request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("X-API-Key", c.apiKey)
	req.Header.Set("anthropic-version", "2023-06-01")

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return "", fmt.Errorf("error sending request: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", fmt.Errorf("error reading response: %w", err)
	}

	var response struct {
		Content []struct {
			Text string `json:"text"`
		} `json:"content"`
		Error *struct {
			Message string `json:"message"`
		} `json:"error,omitempty"`
	}

	if err := json.Unmarshal(body, &response); err != nil {
		return "", fmt.Errorf("error unmarshaling response: %w", err)
	}

	if response.Error != nil {
		return "", fmt.Errorf("API error: %s", response.Error.Message)
	}

	if len(response.Content) == 0 {
		return "", fmt.Errorf("no content returned from API")
	}

	return response.Content[0].Text, nil
}

// MistralClient implements the APIClient interface for Mistral AI
type MistralClient struct {
	httpClient *http.Client
	ctx        context.Context
	apiKey     string
	model      string
	baseURL    string
}

func NewMistralClient(ctx context.Context, apiKey, model string, useCustomBaseURL bool, baseURL string, httpClient *http.Client) *MistralClient {
	endpoint := "https://api.mistral.ai/v1/chat/completions"
	if useCustomBaseURL && baseURL != "" {
		endpoint = baseURL
	}

	return &MistralClient{
		ctx:        ctx,
		apiKey:     apiKey,
		model:      model,
		baseURL:    endpoint,
		httpClient: httpClient,
	}
}

func (c *MistralClient) Query(systemPrompt, prompt string) (string, error) {
	messages := []map[string]string{
		{
			"role":    "user",
			"content": prompt,
		},
	}

	if systemPrompt != "" {
		messages = append([]map[string]string{
			{
				"role":    "system",
				"content": systemPrompt,
			},
		}, messages...)
	}

	payload := map[string]interface{}{
		"model":    c.model,
		"messages": messages,
	}

	bs, err := json.Marshal(payload)
	if err != nil {
		return "", err
	}

	req, err := http.NewRequestWithContext(c.ctx, "POST", c.baseURL, bytes.NewBuffer(bs))
	if err != nil {
		return "", fmt.Errorf("error creating request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+c.apiKey)

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return "", fmt.Errorf("error sending request: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", fmt.Errorf("error reading response: %w", err)
	}

	var response struct {
		Choices []struct {
			Message struct {
				Content string `json:"content"`
			} `json:"message"`
		} `json:"choices"`
		Error *struct {
			Message string `json:"message"`
		} `json:"error,omitempty"`
	}

	if err := json.Unmarshal(body, &response); err != nil {
		return "", fmt.Errorf("error unmarshaling response: %w", err)
	}

	if response.Error != nil {
		return "", fmt.Errorf("API error: %s", response.Error.Message)
	}

	if len(response.Choices) == 0 {
		return "", fmt.Errorf("no choices returned from API")
	}

	return response.Choices[0].Message.Content, nil
}

// OpenRouterClient implements the APIClient interface for OpenRouter
type OpenRouterClient struct {
	httpClient *http.Client
	ctx        context.Context
	apiKey     string
	model      string
	baseURL    string
}

func NewOpenRouterClient(ctx context.Context, apiKey, model string, useCustomBaseURL bool, baseURL string, httpClient *http.Client) *OpenRouterClient {
	endpoint := "https://openrouter.ai/api/v1/chat/completions"
	if useCustomBaseURL && baseURL != "" {
		endpoint = baseURL
	}

	return &OpenRouterClient{
		ctx:        ctx,
		apiKey:     apiKey,
		model:      model,
		baseURL:    endpoint,
		httpClient: httpClient,
	}
}

func (c *OpenRouterClient) Query(systemPrompt, prompt string) (string, error) {
	messages := []map[string]string{
		{
			"role":    "user",
			"content": prompt,
		},
	}

	if systemPrompt != "" {
		messages = append([]map[string]string{
			{
				"role":    "system",
				"content": systemPrompt,
			},
		}, messages...)
	}

	payload := map[string]interface{}{
		"model":    c.model,
		"messages": messages,
	}

	bs, err := json.Marshal(payload)
	if err != nil {
		return "", err
	}

	req, err := http.NewRequestWithContext(c.ctx, "POST", c.baseURL, bytes.NewBuffer(bs))
	if err != nil {
		return "", fmt.Errorf("error creating request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+c.apiKey)
	req.Header.Set("HTTP-Referer", "https://maker.ai")
	req.Header.Set("X-Title", "Maker AI")

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return "", fmt.Errorf("error sending request: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", fmt.Errorf("error reading response: %w", err)
	}

	var response struct {
		Choices []struct {
			Message struct {
				Content string `json:"content"`
			} `json:"message"`
		} `json:"choices"`
		Error *struct {
			Message string `json:"message"`
		} `json:"error,omitempty"`
	}

	if err := json.Unmarshal(body, &response); err != nil {
		return "", fmt.Errorf("error unmarshaling response: %w", err)
	}

	if response.Error != nil {
		return "", fmt.Errorf("API error: %s", response.Error.Message)
	}

	if len(response.Choices) == 0 {
		return "", fmt.Errorf("no choices returned from API")
	}

	return response.Choices[0].Message.Content, nil
}

// OpenAICompatibleClient implements the APIClient interface for OpenAI-compatible APIs
type OpenAICompatibleClient struct {
	httpClient *http.Client
	ctx        context.Context
	apiKey     string
	model      string
	baseURL    string
}

func NewOpenAICompatibleClient(ctx context.Context, apiKey, model string, useCustomBaseURL bool, baseURL string, httpClient *http.Client) *OpenAICompatibleClient {
	endpoint := baseURL
	if !useCustomBaseURL || baseURL == "" {
		endpoint = "https://api.openai.com/v1/chat/completions" // Default fallback
	}

	return &OpenAICompatibleClient{
		ctx:        ctx,
		apiKey:     apiKey,
		model:      model,
		baseURL:    endpoint,
		httpClient: httpClient,
	}
}

func (c *OpenAICompatibleClient) Query(systemPrompt, prompt string) (string, error) {
	if systemPrompt == "" {
		systemPrompt = "You are a helpful assistant."
	}

	payload := map[string]interface{}{
		"model": c.model,
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
	}

	bs, err := json.Marshal(payload)
	if err != nil {
		return "", err
	}

	req, err := http.NewRequestWithContext(c.ctx, "POST", c.baseURL, bytes.NewBuffer(bs))
	if err != nil {
		return "", fmt.Errorf("error creating request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+c.apiKey)

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return "", fmt.Errorf("error sending request: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", fmt.Errorf("error reading response: %w", err)
	}

	var response struct {
		Choices []struct {
			Message struct {
				Content string `json:"content"`
			} `json:"message"`
		} `json:"choices"`
		Error *struct {
			Message string `json:"message"`
		} `json:"error,omitempty"`
	}

	if err := json.Unmarshal(body, &response); err != nil {
		return "", fmt.Errorf("error unmarshaling response: %w", err)
	}

	if response.Error != nil {
		return "", fmt.Errorf("API error: %s", response.Error.Message)
	}

	if len(response.Choices) == 0 {
		return "", fmt.Errorf("no choices returned from API")
	}

	return response.Choices[0].Message.Content, nil
}
