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

package ent

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/weaviate/weaviate/entities/models"
	"github.com/weaviate/weaviate/entities/schema"
	"github.com/weaviate/weaviate/usecases/config"
)

func Test_classSettings_Defaults(t *testing.T) {
	cs := NewClassSettings(&fakeClassConfig{classConfig: map[string]any{}})

	assert.Equal(t, DefaultModel, cs.Model())
	assert.Equal(t, DefaultTruncate, cs.Truncate())
	assert.Equal(t, "", cs.Region())
	assert.Equal(t, "", cs.CompartmentID())
	assert.Equal(t, "", cs.InputType())
	assert.Equal(t, "", cs.EndpointID())
	assert.Nil(t, cs.OutputDimensions())
}

func Test_classSettings_Values(t *testing.T) {
	cs := NewClassSettings(&fakeClassConfig{classConfig: map[string]any{
		"model":            "cohere.embed-multilingual-v3.0",
		"region":           "us-chicago-1",
		"compartmentId":    "ocid1.compartment.oc1..aaaa",
		"truncate":         "NONE",
		"inputType":        "CLUSTERING",
		"outputDimensions": 512,
		"endpointId":       "ocid1.generativeaiendpoint.oc1.us-chicago-1.aaaa",
	}})

	assert.Equal(t, "cohere.embed-multilingual-v3.0", cs.Model())
	assert.Equal(t, "us-chicago-1", cs.Region())
	assert.Equal(t, "ocid1.compartment.oc1..aaaa", cs.CompartmentID())
	assert.Equal(t, "NONE", cs.Truncate())
	assert.Equal(t, "CLUSTERING", cs.InputType())
	assert.Equal(t, "ocid1.generativeaiendpoint.oc1.us-chicago-1.aaaa", cs.EndpointID())
	require.NotNil(t, cs.OutputDimensions())
	assert.Equal(t, int64(512), *cs.OutputDimensions())
}

func Test_classSettings_Validate(t *testing.T) {
	class := &models.Class{
		Class: "test",
		Properties: []*models.Property{
			{
				DataType: []string{schema.DataTypeText.String()},
				Name:     "test",
			},
		},
	}
	tests := []struct {
		name        string
		classConfig map[string]any
		wantErr     string
	}{
		{
			name:        "defaults are valid",
			classConfig: map[string]any{},
		},
		{
			name: "full valid configuration",
			classConfig: map[string]any{
				"model":            "cohere.embed-v4.0",
				"region":           "us-chicago-1",
				"compartmentId":    "ocid1.compartment.oc1..aaaa",
				"truncate":         "START",
				"inputType":        "SEARCH_DOCUMENT",
				"outputDimensions": 1024,
			},
		},
		{
			name: "root compartment (tenancy OCID) is valid",
			classConfig: map[string]any{
				"compartmentId": "ocid1.tenancy.oc1..aaaa",
			},
		},
		{
			name: "dedicated endpoint is valid",
			classConfig: map[string]any{
				"endpointId": "ocid1.generativeaiendpoint.oc1.us-chicago-1.aaaa",
			},
		},
		{
			name: "empty model without endpoint",
			classConfig: map[string]any{
				"model": "",
			},
			wantErr: "model cannot be empty unless endpointId is set",
		},
		{
			name: "empty model with endpoint is valid",
			classConfig: map[string]any{
				"model":      "",
				"endpointId": "ocid1.generativeaiendpoint.oc1.us-chicago-1.aaaa",
			},
		},
		{
			name: "wrong truncate",
			classConfig: map[string]any{
				"truncate": "LEFT",
			},
			wantErr: "wrong truncate type, available types are: [NONE START END]",
		},
		{
			name: "wrong inputType",
			classConfig: map[string]any{
				"inputType": "IMAGE",
			},
			wantErr: "wrong inputType, available types are: [SEARCH_DOCUMENT SEARCH_QUERY CLASSIFICATION CLUSTERING]",
		},
		{
			name: "wrong outputDimensions",
			classConfig: map[string]any{
				"outputDimensions": 300,
			},
			wantErr: "wrong outputDimensions, available values are: [256 512 1024 1536]",
		},
		{
			name: "wrong endpointId",
			classConfig: map[string]any{
				"endpointId": "ocid1.generativeaimodel.oc1..aaaa",
			},
			wantErr: `wrong endpointId: must be an OCID starting with "ocid1.generativeaiendpoint."`,
		},
		{
			name: "wrong compartmentId",
			classConfig: map[string]any{
				"compartmentId": "my-compartment",
			},
			wantErr: `wrong compartmentId: must be an OCID starting with "ocid1.compartment." or "ocid1.tenancy."`,
		},
		{
			name: "multiple errors are joined",
			classConfig: map[string]any{
				"truncate":  "LEFT",
				"inputType": "IMAGE",
			},
			wantErr: "wrong truncate type, available types are: [NONE START END], " +
				"wrong inputType, available types are: [SEARCH_DOCUMENT SEARCH_QUERY CLASSIFICATION CLUSTERING]",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cs := NewClassSettings(&fakeClassConfig{classConfig: tt.classConfig})
			err := cs.Validate(class)
			if tt.wantErr != "" {
				require.Error(t, err)
				assert.Equal(t, tt.wantErr, err.Error())
			} else {
				assert.NoError(t, err)
			}
		})
	}
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
