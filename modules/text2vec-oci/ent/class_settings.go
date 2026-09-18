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
	"fmt"
	"strings"

	"github.com/weaviate/weaviate/entities/models"
	"github.com/weaviate/weaviate/entities/moduletools"
	basesettings "github.com/weaviate/weaviate/usecases/modulecomponents/settings"
)

// Class-level settings understood by the text2vec-oci module.
const (
	ModelProperty            = "model"
	RegionProperty           = "region"
	CompartmentIDProperty    = "compartmentId"
	TruncateProperty         = "truncate"
	InputTypeProperty        = "inputType"
	OutputDimensionsProperty = "outputDimensions"
	EndpointIDProperty       = "endpointId"
)

const (
	// DefaultModel is the on-demand OCI Generative AI embedding model used when
	// the class does not set one, see
	// https://docs.oracle.com/en-us/iaas/Content/generative-ai/embed-models.htm
	DefaultModel                 = "cohere.embed-v4.0"
	DefaultTruncate              = "END"
	DefaultVectorizeClassName    = false
	DefaultPropertyIndexed       = true
	DefaultVectorizePropertyName = false
	LowerCaseInput               = false
)

const (
	compartmentOCIDPrefix = "ocid1.compartment."
	tenancyOCIDPrefix     = "ocid1.tenancy."
	endpointOCIDPrefix    = "ocid1.generativeaiendpoint."
)

var (
	availableTruncates = []string{"NONE", "START", "END"}
	// availableInputTypes lists the text input types of the embedText API. IMAGE
	// is deliberately excluded as this module only vectorizes text.
	availableInputTypes = []string{"SEARCH_DOCUMENT", "SEARCH_QUERY", "CLASSIFICATION", "CLUSTERING"}
	// availableOutputDimensions are the values accepted by the embedText API
	// (embed-v4 and newer models).
	availableOutputDimensions = []int64{256, 512, 1024, 1536}
)

type classSettings struct {
	basesettings.BaseClassSettings
	cfg moduletools.ClassConfig
}

func NewClassSettings(cfg moduletools.ClassConfig) *classSettings {
	return &classSettings{cfg: cfg, BaseClassSettings: *basesettings.NewBaseClassSettings(cfg, LowerCaseInput)}
}

// Model is the on-demand model id, e.g. cohere.embed-v4.0. It is ignored when
// EndpointID points at a dedicated AI cluster endpoint.
func (cs *classSettings) Model() string {
	return cs.GetPropertyAsString(ModelProperty, DefaultModel)
}

// Region is the OCI region identifier, e.g. us-chicago-1. When empty the
// module falls back to the OCI_REGION environment variable.
func (cs *classSettings) Region() string {
	return cs.GetPropertyAsString(RegionProperty, "")
}

// CompartmentID is the OCID of the compartment the inference call is billed
// to. When empty the module falls back to the OCI_COMPARTMENT_ID environment
// variable.
func (cs *classSettings) CompartmentID() string {
	return cs.GetPropertyAsString(CompartmentIDProperty, "")
}

func (cs *classSettings) Truncate() string {
	return cs.GetPropertyAsString(TruncateProperty, DefaultTruncate)
}

// InputType, when set, is used for both objects and queries. When empty the
// module embeds objects with SEARCH_DOCUMENT and queries with SEARCH_QUERY.
func (cs *classSettings) InputType() string {
	return cs.GetPropertyAsString(InputTypeProperty, "")
}

func (cs *classSettings) OutputDimensions() *int64 {
	return cs.GetPropertyAsInt64(OutputDimensionsProperty, nil)
}

// EndpointID is the OCID of a dedicated AI cluster endpoint. When set the
// request uses the DEDICATED serving mode instead of ON_DEMAND.
func (cs *classSettings) EndpointID() string {
	return cs.GetPropertyAsString(EndpointIDProperty, "")
}

func (cs *classSettings) Validate(class *models.Class) error {
	if err := cs.BaseClassSettings.Validate(class); err != nil {
		return err
	}

	var errorMessages []string

	if cs.Model() == "" && cs.EndpointID() == "" {
		errorMessages = append(errorMessages,
			fmt.Sprintf("%s cannot be empty unless %s is set", ModelProperty, EndpointIDProperty))
	}
	if endpointID := cs.EndpointID(); endpointID != "" && !strings.HasPrefix(endpointID, endpointOCIDPrefix) {
		errorMessages = append(errorMessages,
			fmt.Sprintf("wrong %s: must be an OCID starting with %q", EndpointIDProperty, endpointOCIDPrefix))
	}
	if compartmentID := cs.CompartmentID(); compartmentID != "" &&
		!strings.HasPrefix(compartmentID, compartmentOCIDPrefix) &&
		!strings.HasPrefix(compartmentID, tenancyOCIDPrefix) {
		errorMessages = append(errorMessages,
			fmt.Sprintf("wrong %s: must be an OCID starting with %q or %q",
				CompartmentIDProperty, compartmentOCIDPrefix, tenancyOCIDPrefix))
	}
	if !basesettings.ValidateSetting(cs.Truncate(), availableTruncates) {
		errorMessages = append(errorMessages,
			fmt.Sprintf("wrong %s type, available types are: %v", TruncateProperty, availableTruncates))
	}
	if inputType := cs.InputType(); inputType != "" && !basesettings.ValidateSetting(inputType, availableInputTypes) {
		errorMessages = append(errorMessages,
			fmt.Sprintf("wrong %s, available types are: %v", InputTypeProperty, availableInputTypes))
	}
	if dims := cs.OutputDimensions(); dims != nil && !basesettings.ValidateSetting(*dims, availableOutputDimensions) {
		errorMessages = append(errorMessages,
			fmt.Sprintf("wrong %s, available values are: %v", OutputDimensionsProperty, availableOutputDimensions))
	}

	if len(errorMessages) > 0 {
		return fmt.Errorf("%s", strings.Join(errorMessages, ", "))
	}
	return nil
}
