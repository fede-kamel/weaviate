//                           _       _
// __      _____  __ ___   ___  __ _| |_ ___
// \ \ /\ / / _ \/ _` \ \ / / |/ _` | __/ _ \
//  \ V  V /  __/ (_| |\ V /| | (_| | ||  __/
//   \_/\_/ \___|\__,_| \_/ |_|\__,_|\__\___|
//
//  Copyright © 2016 - 2026 Weaviate B.V. All rights reserved.
//
//  CONTACT: hello@weaviate.io
//

package clients

import (
	"bytes"
	"context"
	"crypto/rsa"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"

	"github.com/oracle/oci-go-sdk/v65/common"
	"github.com/pkg/errors"
	"github.com/sirupsen/logrus"

	"github.com/weaviate/weaviate/entities/moduletools"
	"github.com/weaviate/weaviate/modules/text2vec-oci/ent"
	"github.com/weaviate/weaviate/usecases/modulecomponents"
)

const (
	// apiVersion of the OCI Generative AI Inference API,
	// https://docs.oracle.com/en-us/iaas/api/#/en/generative-ai-inference/20231130/EmbedTextResult/EmbedText
	apiVersion    = "20231130"
	embedTextPath = "/" + apiVersion + "/actions/embedText"

	// serviceEndpointTemplate is the template used by the OCI SDK for the
	// generativeaiinference service; resolving it through the SDK gives the
	// right second-level domain for every realm (oraclecloud.com,
	// oraclegovcloud.com, ...).
	serviceName             = "generativeaiinference"
	serviceEndpointTemplate = "https://inference.generativeai.{region}.oci.{secondLevelDomain}"

	// MaxInputsPerRequest is the number of inputs a single embedText call
	// accepts.
	MaxInputsPerRequest = 96

	// OCI Generative AI does not return rate-limit headers. The batch
	// vectorizer therefore works with a fixed requests-per-minute budget that
	// can be overridden per request via X-Oci-Ratelimit-RequestPM-Embedding.
	DefaultRPM = 600
	DefaultTPM = 10000000 // no token limit used by OCI

	servingTypeOnDemand  = "ON_DEMAND"
	servingTypeDedicated = "DEDICATED"
)

// InputType of the embedText API. Objects default to SearchDocument and
// queries to SearchQuery unless the class fixes a single type.
type InputType string

const (
	SearchDocument InputType = "SEARCH_DOCUMENT"
	SearchQuery    InputType = "SEARCH_QUERY"
)

type servingMode struct {
	ServingType string `json:"servingType"`
	ModelID     string `json:"modelId,omitempty"`
	EndpointID  string `json:"endpointId,omitempty"`
}

type embedTextRequest struct {
	ServingMode      servingMode `json:"servingMode"`
	CompartmentID    string      `json:"compartmentId"`
	Inputs           []string    `json:"inputs"`
	Truncate         string      `json:"truncate,omitempty"`
	InputType        InputType   `json:"inputType,omitempty"`
	OutputDimensions *int64      `json:"outputDimensions,omitempty"`
}

type embedTextResponse struct {
	ID           string      `json:"id"`
	Embeddings   [][]float32 `json:"embeddings"`
	ModelID      string      `json:"modelId"`
	ModelVersion string      `json:"modelVersion"`
	// Code and Message are the OCI error envelope returned with non-2xx statuses.
	Code    string `json:"code"`
	Message string `json:"message"`
}

type settings struct {
	Model            string
	Region           string
	CompartmentID    string
	Truncate         string
	InputType        InputType
	OutputDimensions *int64
	EndpointID       string
}

type Client struct {
	config     Config
	privateKey *rsa.PrivateKey
	httpClient *http.Client
	endpointFn func(region string) string
	logger     logrus.FieldLogger
}

func New(config Config, timeout time.Duration, logger logrus.FieldLogger) (*Client, error) {
	c := &Client{
		config:     config,
		httpClient: modulecomponents.NewBaseHttpClient(timeout),
		endpointFn: endpointForRegion,
		logger:     logger,
	}
	if config.PrivateKeyPEM != "" {
		key, err := parsePrivateKey([]byte(config.PrivateKeyPEM), config.PrivateKeyPassphrase)
		if err != nil {
			return nil, errors.Wrap(err, "parse OCI private key")
		}
		c.privateKey = key
	}
	return c, nil
}

// endpointForRegion resolves the Generative AI inference endpoint of a region
// through the OCI SDK realm tables, e.g. us-chicago-1 ->
// https://inference.generativeai.us-chicago-1.oci.oraclecloud.com
func endpointForRegion(region string) string {
	return common.StringToRegion(region).EndpointForTemplate(serviceName, serviceEndpointTemplate)
}

func (c *Client) Vectorize(ctx context.Context, input []string,
	cfg moduletools.ClassConfig,
) (*modulecomponents.VectorizationResult[[]float32], *modulecomponents.RateLimits, int, error) {
	res, err := c.vectorize(ctx, input, c.getSettings(cfg, SearchDocument))
	return res, nil, 0, err
}

func (c *Client) VectorizeQuery(ctx context.Context, input []string,
	cfg moduletools.ClassConfig,
) (*modulecomponents.VectorizationResult[[]float32], error) {
	return c.vectorize(ctx, input, c.getSettings(cfg, SearchQuery))
}

func (c *Client) getSettings(cfg moduletools.ClassConfig, defaultInputType InputType) settings {
	icheck := ent.NewClassSettings(cfg)
	inputType := defaultInputType
	if classInputType := icheck.InputType(); classInputType != "" {
		inputType = InputType(classInputType)
	}
	return settings{
		Model:            icheck.Model(),
		Region:           icheck.Region(),
		CompartmentID:    icheck.CompartmentID(),
		Truncate:         icheck.Truncate(),
		InputType:        inputType,
		OutputDimensions: icheck.OutputDimensions(),
		EndpointID:       icheck.EndpointID(),
	}
}

func (c *Client) vectorize(ctx context.Context, input []string, settings settings,
) (*modulecomponents.VectorizationResult[[]float32], error) {
	if len(input) > MaxInputsPerRequest {
		return nil, fmt.Errorf("too many inputs: %d, OCI Generative AI accepts at most %d inputs per request",
			len(input), MaxInputsPerRequest)
	}

	region, err := c.getRegion(ctx, settings)
	if err != nil {
		return nil, err
	}
	compartmentID, err := c.getCompartmentID(ctx, settings)
	if err != nil {
		return nil, err
	}
	keyProvider, err := c.getKeyProvider(ctx)
	if err != nil {
		return nil, errors.Wrap(err, "OCI credentials")
	}

	body, err := json.Marshal(c.getEmbedTextRequest(input, compartmentID, settings))
	if err != nil {
		return nil, errors.Wrap(err, "marshal body")
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.endpointFn(region)+embedTextPath,
		bytes.NewReader(body))
	if err != nil {
		return nil, errors.Wrap(err, "create POST request")
	}
	// The signer covers date, (request-target), host, content-length,
	// content-type and x-content-sha256 (the last three it computes itself),
	// so Date and Content-Type have to be present before signing.
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "application/json")
	req.Header.Set("Date", time.Now().UTC().Format(http.TimeFormat))
	if err := common.DefaultRequestSigner(keyProvider).Sign(req); err != nil {
		return nil, errors.Wrap(err, "sign request")
	}

	res, err := c.httpClient.Do(req)
	if err != nil {
		return nil, errors.Wrap(err, "send POST request")
	}
	defer res.Body.Close()
	bodyBytes, err := io.ReadAll(res.Body)
	if err != nil {
		return nil, errors.Wrap(err, "read response body")
	}

	var resBody embedTextResponse
	if err := json.Unmarshal(bodyBytes, &resBody); err != nil {
		return nil, fmt.Errorf("failed to parse vectorization response (status %d): %w", res.StatusCode, err)
	}

	if res.StatusCode != http.StatusOK {
		return nil, errors.New(c.getErrorMessage(res.StatusCode, resBody, res.Header.Get("opc-request-id")))
	}

	if len(resBody.Embeddings) == 0 {
		return nil, errors.Errorf("empty embeddings response")
	}
	if len(resBody.Embeddings) != len(input) {
		return nil, errors.Errorf("expected %d embeddings, got %d", len(input), len(resBody.Embeddings))
	}

	return &modulecomponents.VectorizationResult[[]float32]{
		Text:       input,
		Dimensions: len(resBody.Embeddings[0]),
		Vector:     resBody.Embeddings,
	}, nil
}

func (c *Client) getEmbedTextRequest(input []string, compartmentID string, settings settings) embedTextRequest {
	mode := servingMode{ServingType: servingTypeOnDemand, ModelID: settings.Model}
	if settings.EndpointID != "" {
		mode = servingMode{ServingType: servingTypeDedicated, EndpointID: settings.EndpointID}
	}
	return embedTextRequest{
		ServingMode:      mode,
		CompartmentID:    compartmentID,
		Inputs:           input,
		Truncate:         settings.Truncate,
		InputType:        settings.InputType,
		OutputDimensions: settings.OutputDimensions,
	}
}

func (c *Client) getRegion(ctx context.Context, settings settings) (string, error) {
	if region := modulecomponents.GetValueFromContext(ctx, HeaderRegion); region != "" {
		return region, nil
	}
	if settings.Region != "" {
		return settings.Region, nil
	}
	if c.config.Region != "" {
		return c.config.Region, nil
	}
	return "", errors.Errorf("no region found neither in class config: %s, "+
		"nor in request header: %s, nor in environment variable under %s",
		ent.RegionProperty, HeaderRegion, EnvRegion)
}

func (c *Client) getCompartmentID(ctx context.Context, settings settings) (string, error) {
	if compartmentID := modulecomponents.GetValueFromContext(ctx, HeaderCompartmentID); compartmentID != "" {
		return compartmentID, nil
	}
	if settings.CompartmentID != "" {
		return settings.CompartmentID, nil
	}
	if c.config.CompartmentID != "" {
		return c.config.CompartmentID, nil
	}
	return "", errors.Errorf("no compartment found neither in class config: %s, "+
		"nor in request header: %s, nor in environment variable under %s",
		ent.CompartmentIDProperty, HeaderCompartmentID, EnvCompartmentID)
}

func (c *Client) getErrorMessage(statusCode int, res embedTextResponse, opcRequestID string) string {
	msg := fmt.Sprintf("connection to OCI Generative AI failed with status: %d", statusCode)
	if res.Code != "" {
		msg += fmt.Sprintf(" code: %s", res.Code)
	}
	if res.Message != "" {
		msg += fmt.Sprintf(" error: %s", res.Message)
	}
	if opcRequestID != "" {
		msg += fmt.Sprintf(" opc-request-id: %s", opcRequestID)
	}
	return msg
}

// GetApiKeyHash partitions the batch rate limits per signing identity.
func (c *Client) GetApiKeyHash(ctx context.Context, config moduletools.ClassConfig) [32]byte {
	keyProvider, err := c.getKeyProvider(ctx)
	if err != nil {
		return [32]byte{}
	}
	return sha256.Sum256([]byte(keyProvider.keyID))
}

func (c *Client) GetVectorizerRateLimit(ctx context.Context, cfg moduletools.ClassConfig) *modulecomponents.RateLimits {
	rpm, _ := modulecomponents.GetRateLimitFromContext(ctx, "Oci", DefaultRPM, 0)

	execAfterRequestFunction := func(limits *modulecomponents.RateLimits, tokensUsed int, deductRequest bool) {
		// refresh is after 60 seconds but leave a bit of room for errors. Otherwise, we only deduct the request that just happened
		if limits.LastOverwrite.Add(61 * time.Second).After(time.Now()) {
			if deductRequest {
				limits.RemainingRequests -= 1
			}
			return
		}

		limits.RemainingRequests = rpm
		limits.ResetRequests = time.Now().Add(time.Duration(61) * time.Second)
		limits.LimitRequests = rpm
		limits.LastOverwrite = time.Now()

		// high dummy values
		limits.RemainingTokens = DefaultTPM
		limits.LimitTokens = DefaultTPM
		limits.ResetTokens = time.Now().Add(time.Duration(1) * time.Second)
	}

	initialRL := &modulecomponents.RateLimits{AfterRequestFunction: execAfterRequestFunction, LastOverwrite: time.Now().Add(-61 * time.Minute)}
	initialRL.ResetAfterRequestFunction(0) // set initial values

	return initialRL
}
