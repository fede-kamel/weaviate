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
	"context"
	"os"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/weaviate/weaviate/test/docker"
)

type ociCredentials struct {
	tenancyOCID   string
	userOCID      string
	fingerprint   string
	privateKeyPEM string
	compartmentID string
	region        string
}

func TestText2VecOCI_SingleNode(t *testing.T) {
	creds := credentialsFromEnv(t)

	ctx := context.Background()
	compose, err := createSingleNodeEnvironment(ctx, creds)
	require.NoError(t, err)
	defer func() {
		require.NoError(t, compose.Terminate(ctx))
	}()
	rest := compose.GetWeaviate().URI()
	grpc := compose.GetWeaviate().GrpcURI()

	t.Run("tests", testText2VecOCI(rest, grpc, creds.region))
}

// credentialsFromEnv reads OCI API key credentials from the environment and
// skips the test when they are not present.
func credentialsFromEnv(t *testing.T) ociCredentials {
	requireEnv := func(name string) string {
		value := os.Getenv(name)
		if value == "" {
			t.Skipf("skipping, %s environment variable not present", name)
		}
		return value
	}
	creds := ociCredentials{
		tenancyOCID:   requireEnv("OCI_TENANCY_OCID"),
		userOCID:      requireEnv("OCI_USER_OCID"),
		fingerprint:   requireEnv("OCI_FINGERPRINT"),
		compartmentID: requireEnv("OCI_COMPARTMENT_ID"),
		region:        requireEnv("OCI_REGION"),
	}
	creds.privateKeyPEM = os.Getenv("OCI_PRIVATE_KEY")
	if creds.privateKeyPEM == "" {
		keyFile := requireEnv("OCI_PRIVATE_KEY_FILE")
		pemBytes, err := os.ReadFile(keyFile)
		require.NoError(t, err)
		creds.privateKeyPEM = string(pemBytes)
	}
	return creds
}

func createSingleNodeEnvironment(ctx context.Context, creds ociCredentials,
) (compose *docker.DockerCompose, err error) {
	compose, err = composeModules(creds).
		WithWeaviate().
		WithWeaviateWithGRPC().
		Start(ctx)
	return compose, err
}

func composeModules(creds ociCredentials) (composeModules *docker.Compose) {
	composeModules = docker.New().
		WithText2VecOCI(creds.tenancyOCID, creds.userOCID, creds.fingerprint,
			creds.privateKeyPEM, creds.compartmentID, creds.region)
	return composeModules
}
