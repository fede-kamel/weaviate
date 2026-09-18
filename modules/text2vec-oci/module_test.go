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

package modoci

import (
	"context"
	"fmt"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/sirupsen/logrus/hooks/test"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/weaviate/weaviate/entities/models"
	"github.com/weaviate/weaviate/entities/moduletools"
	"github.com/weaviate/weaviate/entities/schema"
	"github.com/weaviate/weaviate/modules/text2vec-oci/clients"
	"github.com/weaviate/weaviate/usecases/config"
	"github.com/weaviate/weaviate/usecases/modulecomponents"
	"github.com/weaviate/weaviate/usecases/modulecomponents/batch"
	"github.com/weaviate/weaviate/usecases/modulecomponents/text2vecbase"
)

func TestModuleInit(t *testing.T) {
	logger, _ := test.NewNullLogger()
	params := moduletools.NewInitParams("", nil, &config.Config{}, logger, nil)
	for _, name := range []string{clients.EnvPrivateKey, clients.EnvPrivateKeyFile} {
		t.Setenv(name, "")
	}

	t.Run("starts without credentials", func(t *testing.T) {
		m := New()
		require.NoError(t, m.Init(context.Background(), params))
		require.NoError(t, m.InitExtension(nil))
		assert.Equal(t, Name, m.Name())

		meta, err := m.MetaInfo()
		require.NoError(t, err)
		assert.Equal(t, "OCI Generative AI Module", meta["name"])
		assert.Contains(t, m.Arguments(), "nearText")
		assert.Contains(t, m.VectorSearches(), "nearText")
	})

	t.Run("fails on an unreadable key file", func(t *testing.T) {
		t.Setenv(clients.EnvPrivateKeyFile, filepath.Join(t.TempDir(), "missing.pem"))
		err := New().Init(context.Background(), params)
		require.Error(t, err)
		assert.Contains(t, err.Error(), "init vectorizer: read OCI_PRIVATE_KEY_FILE")
	})

	t.Run("fails on an invalid key", func(t *testing.T) {
		t.Setenv(clients.EnvPrivateKey, "not a key")
		err := New().Init(context.Background(), params)
		require.Error(t, err)
		assert.Contains(t, err.Error(), "init vectorizer: parse OCI private key")
	})
}

func TestClassConfigDefaults(t *testing.T) {
	m := New()
	assert.Equal(t, map[string]interface{}{
		"vectorizeClassName": false,
		"model":              "cohere.embed-v4.0",
		"truncate":           "END",
	}, m.ClassConfigDefaults())
	assert.Equal(t, map[string]interface{}{
		"skip":                  false,
		"vectorizePropertyName": false,
	}, m.PropertyConfigDefaults(nil))
}

// TestBatchSplitsAtServiceLimit feeds more objects than a single embedText
// call accepts through the module's batch settings and checks that no request
// exceeds the 96 input limit while every object still receives its vector.
func TestBatchSplitsAtServiceLimit(t *testing.T) {
	client := &fakeBatchClient{}
	logger, _ := test.NewNullLogger()
	v := text2vecbase.New(client,
		batch.NewBatchVectorizer(client, 50*time.Second, batchSettings, logger, Name),
		batch.ReturnBatchTokenizer(batchSettings.TokenMultiplier, Name, false),
	)

	const numObjects = 250
	objects := make([]*models.Object, numObjects)
	skip := make([]bool, numObjects)
	for i := range objects {
		objects[i] = &models.Object{Class: "Document", Properties: map[string]interface{}{"text": fmt.Sprintf("document %d", i)}}
	}
	cfg := fakeClassConfig{classConfig: map[string]interface{}{"vectorizeClassName": false}}

	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
	defer cancel()
	vecs, errs := v.ObjectBatch(ctx, objects, skip, cfg)

	require.Empty(t, errs)
	require.Len(t, vecs, numObjects)
	for i, vec := range vecs {
		require.Equal(t, []float32{float32(i)}, vec, "object %d received the wrong vector", i)
	}

	sizes := client.batchSizes()
	require.NotEmpty(t, sizes)
	total := 0
	for _, size := range sizes {
		assert.LessOrEqual(t, size, clients.MaxInputsPerRequest)
		total += size
	}
	assert.Equal(t, numObjects, total)
	assert.GreaterOrEqual(t, len(sizes), 3, "250 objects need at least three requests of at most 96 inputs")
}

type fakeBatchClient struct {
	mu    sync.Mutex
	sizes []int
}

func (c *fakeBatchClient) batchSizes() []int {
	c.mu.Lock()
	defer c.mu.Unlock()
	return append([]int(nil), c.sizes...)
}

func (c *fakeBatchClient) Vectorize(ctx context.Context, text []string, cfg moduletools.ClassConfig,
) (*modulecomponents.VectorizationResult[[]float32], *modulecomponents.RateLimits, int, error) {
	c.mu.Lock()
	c.sizes = append(c.sizes, len(text))
	c.mu.Unlock()

	vectors := make([][]float32, len(text))
	for i := range text {
		// the object text is "document <n>"; return n so ordering can be checked
		n, err := strconv.Atoi(strings.TrimPrefix(text[i], "document "))
		if err != nil {
			return nil, nil, 0, err
		}
		vectors[i] = []float32{float32(n)}
	}
	return &modulecomponents.VectorizationResult[[]float32]{Vector: vectors, Dimensions: 1, Text: text}, nil, 0, nil
}

func (c *fakeBatchClient) VectorizeQuery(ctx context.Context, text []string, cfg moduletools.ClassConfig,
) (*modulecomponents.VectorizationResult[[]float32], error) {
	return &modulecomponents.VectorizationResult[[]float32]{Vector: [][]float32{{0}}, Dimensions: 1, Text: text}, nil
}

func (c *fakeBatchClient) GetVectorizerRateLimit(ctx context.Context, cfg moduletools.ClassConfig) *modulecomponents.RateLimits {
	return &modulecomponents.RateLimits{
		RemainingTokens: 1000000, RemainingRequests: 1000, LimitTokens: 1000000, LimitRequests: 1000,
		ResetTokens: time.Now().Add(time.Minute), ResetRequests: time.Now().Add(time.Minute),
	}
}

func (c *fakeBatchClient) GetApiKeyHash(ctx context.Context, cfg moduletools.ClassConfig) [32]byte {
	return [32]byte{}
}

type fakeClassConfig struct {
	classConfig map[string]interface{}
}

func (f fakeClassConfig) Class() map[string]interface{} {
	return f.classConfig
}

func (f fakeClassConfig) ClassByModuleName(moduleName string) map[string]interface{} {
	return f.classConfig
}

func (f fakeClassConfig) Property(propName string) map[string]interface{} {
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
