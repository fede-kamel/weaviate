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

package tests

import (
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/weaviate/weaviate/entities/models"
	"github.com/weaviate/weaviate/test/helper"
	"github.com/weaviate/weaviate/test/helper/sample-schema/companies"
)

func testText2VecOCI(rest, grpc, region string) func(t *testing.T) {
	return func(t *testing.T) {
		helper.SetupClient(rest)
		className := "VectorizerTest"
		ptrInt := func(i int) *int { return &i }
		tests := []struct {
			name  string
			model string
			// outputDimensions, when set, configures the output vector size.
			// Only embed-v4 and newer models support it (256, 512, 1024, 1536).
			outputDimensions *int
		}{
			{
				name:  "cohere.embed-v4.0",
				model: "cohere.embed-v4.0",
			},
			{
				name:             "cohere.embed-v4.0 with outputDimensions",
				model:            "cohere.embed-v4.0",
				outputDimensions: ptrInt(512),
			},
			{
				name:  "cohere.embed-multilingual-v3.0",
				model: "cohere.embed-multilingual-v3.0",
			},
		}
		for _, tt := range tests {
			t.Run(tt.name, func(t *testing.T) {
				descriptionVectorizer := map[string]any{
					"text2vec-oci": map[string]any{
						"properties":         []any{"description"},
						"vectorizeClassName": false,
						"region":             region,
						"model":              tt.model,
					},
				}
				emptyVectorizer := map[string]any{
					"text2vec-oci": map[string]any{
						"properties":         []any{"empty"},
						"vectorizeClassName": false,
						"region":             region,
						"model":              tt.model,
					},
				}
				if tt.outputDimensions != nil {
					descriptionVectorizer["text2vec-oci"].(map[string]any)["outputDimensions"] = *tt.outputDimensions
					emptyVectorizer["text2vec-oci"].(map[string]any)["outputDimensions"] = *tt.outputDimensions
				}
				t.Run("search", func(t *testing.T) {
					companies.TestSuite(t, rest, grpc, className, descriptionVectorizer)
				})
				t.Run("empty values", func(t *testing.T) {
					companies.TestSuiteWithEmptyValues(t, rest, grpc, className, descriptionVectorizer, emptyVectorizer)
				})
				if tt.outputDimensions != nil {
					t.Run("outputDimensions", func(t *testing.T) {
						class := companies.BaseClass(className)
						class.VectorConfig = map[string]models.VectorConfig{
							"description": {
								Vectorizer:      descriptionVectorizer,
								VectorIndexType: "hnsw",
							},
						}
						helper.CreateClass(t, class)
						defer helper.DeleteClass(t, class.Class)

						companies.InsertObjects(t, rest, class.Class)

						obj, err := helper.GetObject(t, class.Class, companies.SpaceX, "vector")
						require.NoError(t, err)
						require.NotNil(t, obj)
						vector, ok := obj.Vectors["description"].([]float32)
						require.True(t, ok)
						require.Len(t, vector, *tt.outputDimensions)
					})
				}
			})
		}
	}
}
