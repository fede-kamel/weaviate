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
	"context"
	"crypto"
	"crypto/rand"
	"crypto/rsa"
	"crypto/sha256"
	"crypto/x509"
	"encoding/base64"
	"encoding/json"
	"encoding/pem"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/sirupsen/logrus"
	"github.com/sirupsen/logrus/hooks/test"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/weaviate/weaviate/entities/schema"
	"github.com/weaviate/weaviate/usecases/config"
	"github.com/weaviate/weaviate/usecases/modulecomponents"
)

const (
	testTenancy     = "ocid1.tenancy.oc1..tenancy"
	testUser        = "ocid1.user.oc1..user"
	testFingerprint = "aa:bb:cc:dd:ee:ff"
	testCompartment = "ocid1.compartment.oc1..compartment"
	testRegion      = "us-chicago-1"
)

func TestClient(t *testing.T) {
	key, keyPEM := newTestKey(t)
	apiKeyConfig := Config{
		TenancyOCID:   testTenancy,
		UserOCID:      testUser,
		Fingerprint:   testFingerprint,
		PrivateKeyPEM: keyPEM,
		Region:        testRegion,
		CompartmentID: testCompartment,
	}

	t.Run("when all is fine", func(t *testing.T) {
		handler := &fakeHandler{t: t, publicKey: &key.PublicKey}
		server := httptest.NewServer(handler)
		defer server.Close()
		c := newTestClient(t, apiKeyConfig, server.URL)

		res, rl, tokens, err := c.Vectorize(context.Background(), []string{"This is my text", "and another one"},
			fakeClassConfig{classConfig: map[string]any{}})

		require.NoError(t, err)
		assert.Nil(t, rl)
		assert.Equal(t, 0, tokens)
		assert.Equal(t, &modulecomponents.VectorizationResult[[]float32]{
			Text:       []string{"This is my text", "and another one"},
			Dimensions: 3,
			Vector:     [][]float32{{0.1, 0.2, 0.3}, {0.1, 0.2, 0.3}},
		}, res)

		require.Len(t, handler.requests, 1)
		req := handler.requests[0]
		assert.Equal(t, testTenancy+"/"+testUser+"/"+testFingerprint, req.keyID)
		assert.Equal(t, embedTextRequest{
			ServingMode:   servingMode{ServingType: "ON_DEMAND", ModelID: "cohere.embed-v4.0"},
			CompartmentID: testCompartment,
			Inputs:        []string{"This is my text", "and another one"},
			Truncate:      "END",
			InputType:     SearchDocument,
		}, req.body)
		assert.Equal(t, "application/json", req.headers.Get("Content-Type"))
		assert.NotEmpty(t, req.headers.Get("Date"))
	})

	t.Run("queries use SEARCH_QUERY", func(t *testing.T) {
		handler := &fakeHandler{t: t, publicKey: &key.PublicKey}
		server := httptest.NewServer(handler)
		defer server.Close()
		c := newTestClient(t, apiKeyConfig, server.URL)

		res, err := c.VectorizeQuery(context.Background(), []string{"query"}, fakeClassConfig{classConfig: map[string]any{}})

		require.NoError(t, err)
		assert.Equal(t, [][]float32{{0.1, 0.2, 0.3}}, res.Vector)
		require.Len(t, handler.requests, 1)
		assert.Equal(t, SearchQuery, handler.requests[0].body.InputType)
	})

	t.Run("class inputType is used for objects and queries", func(t *testing.T) {
		handler := &fakeHandler{t: t, publicKey: &key.PublicKey}
		server := httptest.NewServer(handler)
		defer server.Close()
		c := newTestClient(t, apiKeyConfig, server.URL)
		cfg := fakeClassConfig{classConfig: map[string]any{"inputType": "CLUSTERING"}}

		_, _, _, err := c.Vectorize(context.Background(), []string{"object"}, cfg)
		require.NoError(t, err)
		_, err = c.VectorizeQuery(context.Background(), []string{"query"}, cfg)
		require.NoError(t, err)

		require.Len(t, handler.requests, 2)
		assert.Equal(t, InputType("CLUSTERING"), handler.requests[0].body.InputType)
		assert.Equal(t, InputType("CLUSTERING"), handler.requests[1].body.InputType)
	})

	t.Run("class settings are forwarded", func(t *testing.T) {
		handler := &fakeHandler{t: t, publicKey: &key.PublicKey}
		server := httptest.NewServer(handler)
		defer server.Close()
		c := newTestClient(t, apiKeyConfig, server.URL)
		c.endpointFn = func(region string) string {
			assert.Equal(t, "eu-frankfurt-1", region)
			return server.URL
		}

		_, _, _, err := c.Vectorize(context.Background(), []string{"object"}, fakeClassConfig{classConfig: map[string]any{
			"model":            "cohere.embed-multilingual-v3.0",
			"region":           "eu-frankfurt-1",
			"compartmentId":    "ocid1.compartment.oc1..other",
			"truncate":         "NONE",
			"outputDimensions": 512,
		}})

		require.NoError(t, err)
		require.Len(t, handler.requests, 1)
		dims := int64(512)
		assert.Equal(t, embedTextRequest{
			ServingMode:      servingMode{ServingType: "ON_DEMAND", ModelID: "cohere.embed-multilingual-v3.0"},
			CompartmentID:    "ocid1.compartment.oc1..other",
			Inputs:           []string{"object"},
			Truncate:         "NONE",
			InputType:        SearchDocument,
			OutputDimensions: &dims,
		}, handler.requests[0].body)
	})

	t.Run("dedicated endpoint uses DEDICATED serving mode", func(t *testing.T) {
		handler := &fakeHandler{t: t, publicKey: &key.PublicKey}
		server := httptest.NewServer(handler)
		defer server.Close()
		c := newTestClient(t, apiKeyConfig, server.URL)

		_, _, _, err := c.Vectorize(context.Background(), []string{"object"}, fakeClassConfig{classConfig: map[string]any{
			"endpointId": "ocid1.generativeaiendpoint.oc1.us-chicago-1.endpoint",
		}})

		require.NoError(t, err)
		require.Len(t, handler.requests, 1)
		assert.Equal(t, servingMode{
			ServingType: "DEDICATED",
			EndpointID:  "ocid1.generativeaiendpoint.oc1.us-chicago-1.endpoint",
		}, handler.requests[0].body.ServingMode)
	})

	t.Run("security token from environment signs with ST$ keyId", func(t *testing.T) {
		handler := &fakeHandler{t: t, publicKey: &key.PublicKey}
		server := httptest.NewServer(handler)
		defer server.Close()
		c := newTestClient(t, Config{
			PrivateKeyPEM: keyPEM,
			SecurityToken: "my-session-token",
			Region:        testRegion,
			CompartmentID: testCompartment,
		}, server.URL)

		_, _, _, err := c.Vectorize(context.Background(), []string{"object"}, fakeClassConfig{classConfig: map[string]any{}})

		require.NoError(t, err)
		require.Len(t, handler.requests, 1)
		assert.Equal(t, "ST$my-session-token", handler.requests[0].keyID)
	})

	t.Run("security token file is re-read on every request", func(t *testing.T) {
		handler := &fakeHandler{t: t, publicKey: &key.PublicKey}
		server := httptest.NewServer(handler)
		defer server.Close()
		tokenFile := filepath.Join(t.TempDir(), "token")
		require.NoError(t, os.WriteFile(tokenFile, []byte("first-token\n"), 0o600))
		c := newTestClient(t, Config{
			PrivateKeyPEM:     keyPEM,
			SecurityTokenFile: tokenFile,
			Region:            testRegion,
			CompartmentID:     testCompartment,
		}, server.URL)

		_, _, _, err := c.Vectorize(context.Background(), []string{"object"}, fakeClassConfig{classConfig: map[string]any{}})
		require.NoError(t, err)
		require.NoError(t, os.WriteFile(tokenFile, []byte("refreshed-token"), 0o600))
		_, _, _, err = c.Vectorize(context.Background(), []string{"object"}, fakeClassConfig{classConfig: map[string]any{}})
		require.NoError(t, err)

		require.Len(t, handler.requests, 2)
		assert.Equal(t, "ST$first-token", handler.requests[0].keyID)
		assert.Equal(t, "ST$refreshed-token", handler.requests[1].keyID)
	})

	t.Run("credentials and routing are passed using X-Oci-* headers", func(t *testing.T) {
		handler := &fakeHandler{t: t, publicKey: &key.PublicKey}
		server := httptest.NewServer(handler)
		defer server.Close()
		// no credentials configured at all
		c := newTestClient(t, Config{}, server.URL)
		c.endpointFn = func(region string) string {
			assert.Equal(t, "us-ashburn-1", region)
			return server.URL
		}
		ctx := context.WithValue(context.Background(), HeaderTenancyOCID, []string{"ocid1.tenancy.oc1..header"})
		ctx = context.WithValue(ctx, HeaderUserOCID, []string{"ocid1.user.oc1..header"})
		ctx = context.WithValue(ctx, HeaderFingerprint, []string{"11:22:33"})
		ctx = context.WithValue(ctx, HeaderPrivateKey, []string{base64.StdEncoding.EncodeToString([]byte(keyPEM))})
		ctx = context.WithValue(ctx, HeaderRegion, []string{"us-ashburn-1"})
		ctx = context.WithValue(ctx, HeaderCompartmentID, []string{"ocid1.compartment.oc1..header"})

		_, _, _, err := c.Vectorize(ctx, []string{"object"}, fakeClassConfig{classConfig: map[string]any{}})

		require.NoError(t, err)
		require.Len(t, handler.requests, 1)
		assert.Equal(t, "ocid1.tenancy.oc1..header/ocid1.user.oc1..header/11:22:33", handler.requests[0].keyID)
		assert.Equal(t, "ocid1.compartment.oc1..header", handler.requests[0].body.CompartmentID)
	})

	t.Run("security token header takes precedence over API key", func(t *testing.T) {
		handler := &fakeHandler{t: t, publicKey: &key.PublicKey}
		server := httptest.NewServer(handler)
		defer server.Close()
		c := newTestClient(t, apiKeyConfig, server.URL)
		ctx := context.WithValue(context.Background(), HeaderSecurityToken, []string{"header-token"})

		_, _, _, err := c.Vectorize(ctx, []string{"object"}, fakeClassConfig{classConfig: map[string]any{}})

		require.NoError(t, err)
		require.Len(t, handler.requests, 1)
		assert.Equal(t, "ST$header-token", handler.requests[0].keyID)
	})

	t.Run("when the private key is missing", func(t *testing.T) {
		handler := &fakeHandler{t: t, publicKey: &key.PublicKey}
		server := httptest.NewServer(handler)
		defer server.Close()
		cfg := apiKeyConfig
		cfg.PrivateKeyPEM = ""
		c := newTestClient(t, cfg, server.URL)

		_, _, _, err := c.Vectorize(context.Background(), []string{"object"}, fakeClassConfig{classConfig: map[string]any{}})

		require.Error(t, err)
		assert.Contains(t, err.Error(), "OCI credentials: no private key found")
		assert.Contains(t, err.Error(), "OCI_PRIVATE_KEY")
		assert.Empty(t, handler.requests)
	})

	t.Run("when the API key tuple is incomplete", func(t *testing.T) {
		handler := &fakeHandler{t: t, publicKey: &key.PublicKey}
		server := httptest.NewServer(handler)
		defer server.Close()
		cfg := apiKeyConfig
		cfg.UserOCID = ""
		cfg.Fingerprint = ""
		c := newTestClient(t, cfg, server.URL)

		_, _, _, err := c.Vectorize(context.Background(), []string{"object"}, fakeClassConfig{classConfig: map[string]any{}})

		require.Error(t, err)
		assert.Contains(t, err.Error(), "incomplete OCI API key credentials, missing: OCI_USER_OCID (X-Oci-User-Ocid), OCI_FINGERPRINT (X-Oci-Fingerprint)")
		assert.NotContains(t, err.Error(), "OCI_TENANCY_OCID")
		assert.Empty(t, handler.requests)
	})

	t.Run("when region and compartment are missing", func(t *testing.T) {
		handler := &fakeHandler{t: t, publicKey: &key.PublicKey}
		server := httptest.NewServer(handler)
		defer server.Close()
		cfg := apiKeyConfig
		cfg.Region = ""
		c := newTestClient(t, cfg, server.URL)

		_, _, _, err := c.Vectorize(context.Background(), []string{"object"}, fakeClassConfig{classConfig: map[string]any{}})
		require.Error(t, err)
		assert.Contains(t, err.Error(), "no region found")

		cfg.Region = testRegion
		cfg.CompartmentID = ""
		c = newTestClient(t, cfg, server.URL)
		_, _, _, err = c.Vectorize(context.Background(), []string{"object"}, fakeClassConfig{classConfig: map[string]any{}})
		require.Error(t, err)
		assert.Contains(t, err.Error(), "no compartment found")
		assert.Empty(t, handler.requests)
	})

	t.Run("when too many inputs are passed", func(t *testing.T) {
		handler := &fakeHandler{t: t, publicKey: &key.PublicKey}
		server := httptest.NewServer(handler)
		defer server.Close()
		c := newTestClient(t, apiKeyConfig, server.URL)
		inputs := make([]string, MaxInputsPerRequest+1)
		for i := range inputs {
			inputs[i] = "text"
		}

		_, _, _, err := c.Vectorize(context.Background(), inputs, fakeClassConfig{classConfig: map[string]any{}})

		require.Error(t, err)
		assert.Equal(t, "too many inputs: 97, OCI Generative AI accepts at most 96 inputs per request", err.Error())
		assert.Empty(t, handler.requests)
	})

	t.Run("when the server returns an error", func(t *testing.T) {
		handler := &fakeHandler{
			t:            t,
			publicKey:    &key.PublicKey,
			statusCode:   http.StatusUnauthorized,
			errCode:      "NotAuthenticated",
			errMessage:   "The required information to complete authentication was not provided or was incorrect.",
			opcRequestID: "ABC123",
		}
		server := httptest.NewServer(handler)
		defer server.Close()
		c := newTestClient(t, apiKeyConfig, server.URL)

		_, _, _, err := c.Vectorize(context.Background(), []string{"object"}, fakeClassConfig{classConfig: map[string]any{}})

		require.Error(t, err)
		assert.Equal(t, "connection to OCI Generative AI failed with status: 401 code: NotAuthenticated "+
			"error: The required information to complete authentication was not provided or was incorrect. "+
			"opc-request-id: ABC123", err.Error())
	})

	t.Run("when the server returns fewer embeddings than inputs", func(t *testing.T) {
		handler := &fakeHandler{t: t, publicKey: &key.PublicKey, dropLastEmbedding: true}
		server := httptest.NewServer(handler)
		defer server.Close()
		c := newTestClient(t, apiKeyConfig, server.URL)

		_, _, _, err := c.Vectorize(context.Background(), []string{"one", "two"}, fakeClassConfig{classConfig: map[string]any{}})

		require.Error(t, err)
		assert.Equal(t, "expected 2 embeddings, got 1", err.Error())
	})

	t.Run("when the server returns no embeddings", func(t *testing.T) {
		handler := &fakeHandler{t: t, publicKey: &key.PublicKey, dropLastEmbedding: true}
		server := httptest.NewServer(handler)
		defer server.Close()
		c := newTestClient(t, apiKeyConfig, server.URL)

		_, _, _, err := c.Vectorize(context.Background(), []string{"one"}, fakeClassConfig{classConfig: map[string]any{}})

		require.Error(t, err)
		assert.Equal(t, "empty embeddings response", err.Error())
	})

	t.Run("when the context is expired", func(t *testing.T) {
		handler := &fakeHandler{t: t, publicKey: &key.PublicKey}
		server := httptest.NewServer(handler)
		defer server.Close()
		c := newTestClient(t, apiKeyConfig, server.URL)
		ctx, cancel := context.WithDeadline(context.Background(), time.Now())
		defer cancel()

		_, _, _, err := c.Vectorize(ctx, []string{"object"}, fakeClassConfig{classConfig: map[string]any{}})

		require.Error(t, err)
		assert.Contains(t, err.Error(), "context deadline exceeded")
	})
}

func TestEndpointForRegion(t *testing.T) {
	tests := []struct {
		region   string
		expected string
	}{
		{region: "us-chicago-1", expected: "https://inference.generativeai.us-chicago-1.oci.oraclecloud.com"},
		{region: "eu-frankfurt-1", expected: "https://inference.generativeai.eu-frankfurt-1.oci.oraclecloud.com"},
		// oc2 (US Government Cloud) realm
		{region: "us-langley-1", expected: "https://inference.generativeai.us-langley-1.oci.oraclegovcloud.com"},
	}
	for _, tt := range tests {
		t.Run(tt.region, func(t *testing.T) {
			assert.Equal(t, tt.expected, endpointForRegion(tt.region))
		})
	}
}

func TestGetApiKeyHash(t *testing.T) {
	_, keyPEM := newTestKey(t)
	logger, _ := test.NewNullLogger()
	apiKey, err := New(Config{TenancyOCID: testTenancy, UserOCID: testUser, Fingerprint: testFingerprint, PrivateKeyPEM: keyPEM}, 0, logger)
	require.NoError(t, err)
	otherUser, err := New(Config{TenancyOCID: testTenancy, UserOCID: "ocid1.user.oc1..other", Fingerprint: testFingerprint, PrivateKeyPEM: keyPEM}, 0, logger)
	require.NoError(t, err)
	noCredentials, err := New(Config{}, 0, logger)
	require.NoError(t, err)
	cfg := fakeClassConfig{classConfig: map[string]any{}}

	assert.NotEqual(t, apiKey.GetApiKeyHash(context.Background(), cfg), otherUser.GetApiKeyHash(context.Background(), cfg))
	assert.Equal(t, apiKey.GetApiKeyHash(context.Background(), cfg), apiKey.GetApiKeyHash(context.Background(), cfg))
	assert.Equal(t, [32]byte{}, noCredentials.GetApiKeyHash(context.Background(), cfg))
}

func TestGetVectorizerRateLimit(t *testing.T) {
	logger, _ := test.NewNullLogger()
	c, err := New(Config{}, 0, logger)
	require.NoError(t, err)
	cfg := fakeClassConfig{classConfig: map[string]any{}}

	rl := c.GetVectorizerRateLimit(context.Background(), cfg)
	assert.Equal(t, DefaultRPM, rl.LimitRequests)
	assert.Equal(t, DefaultRPM, rl.RemainingRequests)

	ctx := context.WithValue(context.Background(), "X-Oci-Ratelimit-RequestPM-Embedding", []string{"42"})
	rl = c.GetVectorizerRateLimit(ctx, cfg)
	assert.Equal(t, 42, rl.LimitRequests)
	assert.Equal(t, 42, rl.RemainingRequests)
}

func TestNew(t *testing.T) {
	logger, _ := test.NewNullLogger()

	t.Run("without credentials", func(t *testing.T) {
		c, err := New(Config{}, 0, logger)
		require.NoError(t, err)
		assert.Nil(t, c.privateKey)
	})

	t.Run("with a valid key", func(t *testing.T) {
		_, keyPEM := newTestKey(t)
		c, err := New(Config{PrivateKeyPEM: keyPEM}, 0, logger)
		require.NoError(t, err)
		assert.NotNil(t, c.privateKey)
	})

	t.Run("with an invalid key", func(t *testing.T) {
		_, err := New(Config{PrivateKeyPEM: "not a pem"}, 0, logger)
		require.Error(t, err)
		assert.Contains(t, err.Error(), "parse OCI private key")
	})
}

func newTestKey(t *testing.T) (*rsa.PrivateKey, string) {
	key, err := rsa.GenerateKey(rand.Reader, 2048)
	require.NoError(t, err)
	pemBytes := pem.EncodeToMemory(&pem.Block{Type: "RSA PRIVATE KEY", Bytes: x509.MarshalPKCS1PrivateKey(key)})
	return key, string(pemBytes)
}

func newTestClient(t *testing.T, cfg Config, serverURL string) *Client {
	c, err := New(cfg, 0, nullLogger())
	require.NoError(t, err)
	c.httpClient = &http.Client{}
	c.endpointFn = func(region string) string { return serverURL }
	return c
}

type recordedRequest struct {
	body    embedTextRequest
	keyID   string
	headers http.Header
}

// fakeHandler emulates the embedText endpoint and verifies the OCI HTTP
// signature (draft-cavage-http-signatures) of every request against the test
// public key.
type fakeHandler struct {
	t                 *testing.T
	publicKey         *rsa.PublicKey
	statusCode        int
	errCode           string
	errMessage        string
	opcRequestID      string
	dropLastEmbedding bool
	requests          []recordedRequest
}

func (f *fakeHandler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	assert.Equal(f.t, http.MethodPost, r.Method)
	assert.Equal(f.t, embedTextPath, r.URL.Path)
	assert.Equal(f.t, "application/json", r.Header.Get("Content-Type"))
	assert.Equal(f.t, "application/json", r.Header.Get("Accept"))

	bodyBytes, err := io.ReadAll(r.Body)
	require.NoError(f.t, err)
	defer r.Body.Close()

	bodyHash := sha256.Sum256(bodyBytes)
	assert.Equal(f.t, base64.StdEncoding.EncodeToString(bodyHash[:]), r.Header.Get("X-Content-Sha256"))

	params := parseSignatureHeader(f.t, r.Header.Get("Authorization"))
	assert.Equal(f.t, "1", params["version"])
	assert.Equal(f.t, "rsa-sha256", params["algorithm"])
	assert.Equal(f.t, "date (request-target) host content-length content-type x-content-sha256", params["headers"])

	var signingParts []string
	for _, header := range strings.Split(params["headers"], " ") {
		var value string
		switch header {
		case "(request-target)":
			value = strings.ToLower(r.Method) + " " + r.URL.RequestURI()
		case "host":
			value = r.Host
		case "content-length":
			value = strconv.Itoa(len(bodyBytes))
		default:
			value = r.Header.Get(header)
		}
		signingParts = append(signingParts, header+": "+value)
	}
	signingString := sha256.Sum256([]byte(strings.Join(signingParts, "\n")))
	signature, err := base64.StdEncoding.DecodeString(params["signature"])
	require.NoError(f.t, err)
	assert.NoError(f.t, rsa.VerifyPKCS1v15(f.publicKey, crypto.SHA256, signingString[:], signature),
		"request signature does not verify against the configured private key")

	var body embedTextRequest
	require.NoError(f.t, json.Unmarshal(bodyBytes, &body))
	f.requests = append(f.requests, recordedRequest{body: body, keyID: params["keyId"], headers: r.Header.Clone()})

	w.Header().Set("Content-Type", "application/json")
	if f.opcRequestID != "" {
		w.Header().Set("opc-request-id", f.opcRequestID)
	}
	if f.statusCode != 0 && f.statusCode != http.StatusOK {
		w.WriteHeader(f.statusCode)
		require.NoError(f.t, json.NewEncoder(w).Encode(map[string]string{"code": f.errCode, "message": f.errMessage}))
		return
	}

	embeddings := make([][]float32, 0, len(body.Inputs))
	for range body.Inputs {
		embeddings = append(embeddings, []float32{0.1, 0.2, 0.3})
	}
	if f.dropLastEmbedding && len(embeddings) > 0 {
		embeddings = embeddings[:len(embeddings)-1]
	}
	require.NoError(f.t, json.NewEncoder(w).Encode(embedTextResponse{
		ID:           "request-id",
		Embeddings:   embeddings,
		ModelID:      body.ServingMode.ModelID,
		ModelVersion: "1.0",
	}))
}

// parseSignatureHeader splits
// `Signature version="1",headers="...",keyId="...",algorithm="rsa-sha256",signature="..."`
// into its parameters.
func parseSignatureHeader(t *testing.T, header string) map[string]string {
	require.True(t, strings.HasPrefix(header, "Signature "), "unexpected Authorization header: %q", header)
	params := map[string]string{}
	for _, part := range strings.Split(strings.TrimPrefix(header, "Signature "), ",") {
		key, value, ok := strings.Cut(part, "=")
		require.True(t, ok, "malformed signature parameter: %q", part)
		params[key] = strings.Trim(value, `"`)
	}
	return params
}

func nullLogger() logrus.FieldLogger {
	l, _ := test.NewNullLogger()
	return l
}

type fakeClassConfig struct {
	classConfig map[string]any
}

func (f fakeClassConfig) Class() map[string]any {
	return f.classConfig
}

func (f fakeClassConfig) ClassByModuleName(moduleName string) map[string]any {
	return f.classConfig
}

func (f fakeClassConfig) Property(propName string) map[string]any {
	return nil
}

func (f fakeClassConfig) Tenant() string {
	return ""
}

func (f fakeClassConfig) TargetVector() string {
	return ""
}

func (f fakeClassConfig) PropertiesDataTypes() map[string]schema.DataType {
	return nil
}

func (f fakeClassConfig) Config() *config.Config {
	return nil
}
